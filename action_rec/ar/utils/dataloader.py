import copy
import os
import pickle
import torch.utils.data as data
import random
import numpy as np
import cv2
from action_rec.ar.utils.configuration import ubuntu


# https://rose1.ntu.edu.sg/dataset/actionRecognition/


def flat_last(x):
    return x.reshape(x.shape[:-2] + (-1,))


def worker_init_fn(worker_id):
    import torch
    torch_seed = torch.initial_seed()
    random.seed(torch_seed + worker_id)
    if torch_seed >= 2**30:  # make sure torch_seed + workder_id < 2**32
        torch_seed = torch_seed % 2**30
    np.random.seed(torch_seed + worker_id)


def rotate_y(points, angle):  # Created by Chat GPT
    """Rotate an array of 3D points around the y-axis by a given angle (in degrees).

  Args:
    points: An Nx3 array of 3D points.
    angle: The angle (in degrees) to rotate the points around the y-axis.

  Returns:
    An Nx3 array of rotated points.
  """
    # Convert the angle to radians
    angle = np.radians(angle)

    # Create the rotation matrix
    R = np.array([[np.cos(angle), 0, np.sin(angle)],
                  [0, 1, 0],
                  [-np.sin(angle), 0, np.cos(angle)]])

    # Rotate the points
    rotated_points = np.dot(points, R.T)

    return rotated_points


class MyLoader(data.Dataset):
    def __init__(self, queries_path, k=5, n=5, n_task=10000, input_type="hybrid",
                 exemplars_path=None, support_classes=None, query_class=None,
                 skeleton="smpl+head_30", given_122=False, do_augmentation=True,
                 unknown_class=None):
        """
        Loader class that provides all the functionality needed for training and testing
        @param queries_path: path to main dataset
        @param k: dimension of support set (ways)
        @param n: number of element inside the support set for each class (shots)
        @param n_task: number of task for each epoch, if query_class is not provided
        @param max_l: expected maximum number of frame for each instance
        @param l: number of frame to load for each instance
        @param input_type: one between ["skeleton", "rgb", "hybrid"]
        @param exemplars_path: if provided, support set elements will be loaded from this folder
        @param support_classes: if provided, the support set will always contain these classes
        @param query_class: if provided, queries will belong only from this class
        To compute FSOS metric, give the support_classes to fix the support classes together with
         exemplars_path to load exemplars, and give a query_class to fix the queries classes
        """
        self.queries_path = queries_path
        self.k = k if not support_classes else len(support_classes)
        self.n = n
        self.l = 16 if input_type == "skeleton" else 8
        self.input_type = input_type
        self.all_classes = next(os.walk(self.queries_path))[1]  # Get list of directories
        self.do_augmentation = do_augmentation
        self.unknown_class = unknown_class

        self.support_classes = support_classes  # Optional, to load always same classes in support set
        self.exemplars_path = exemplars_path  # Optional, to use exemplars when loading support set

        self.n_task = n_task
        self.query_class = query_class
        self.queries = None
        if self.query_class:
            self.queries = []
            for class_dir in next(os.walk(os.path.join(queries_path, query_class)))[1]:
                self.queries.append(os.path.join(queries_path, query_class, class_dir))
            self.n_task = len(self.queries)
        self.default_sample = None

        self.skeleton = skeleton
        with open(f'action_rec/hpe/assets/skeleton_types.pkl', "rb") as input_file:
            skeleton_types = pickle.load(input_file)
        self.edges = skeleton_types[skeleton]['edges']
        self.indices = skeleton_types[skeleton]['indices'] if given_122 else np.array(range(0, 30))

        # Create dict of file to save time when sampling random sequence of a class
        self.sequences = {}
        for c in self.all_classes:
            self.sequences[c] = next(os.walk(os.path.join(self.queries_path, c)))[1]

    def get_sample(self, class_name, ss=False, path=None):
        if not path:
            if ss and self.exemplars_path:  # Need to load a support from the exemplars
                path = random.sample(next(os.walk(os.path.join(self.exemplars_path, class_name)))[1], 1)
            else:  # Need to load a query from the given class
                path = random.sample(self.sequences[class_name], 1)[0]
            path = os.path.join(self.queries_path, class_name, path)  # Create full path

        sample = {}

        # Load poses
        if self.input_type in ["hybrid", "skeleton"]:
            with open(os.path.join(path, f"poses.pkl"), 'rb') as file:
                poses = pickle.load(file)
            poses = poses[:, self.indices]  # Get true indices
            poses = poses - poses[:, 0][:, None]  # Center w.r.t. pelvis
            if self.do_augmentation:
                poses = rotate_y(poses, random.uniform(0, 360))
            sample["sk"] = poses if self.l == 16 else poses[np.arange(0, 16, 2)]

        # Load images
        if self.input_type in ["rgb", "hybrid"]:
            imgs = []
            for i in range(0, 16, 2):
                img = cv2.imread(os.path.join(path, f"{i}.png"))
                img = cv2.resize(img, (224, 224))
                img = img / 255.
                img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                imgs.append(img.swapaxes(-1, -2).swapaxes(-2, -3))
            sample["rgb"] = np.stack(imgs)  # if self.l == self.max_l else np.stack(imgs)[np.arange(0, 16, 2)]

        return sample

    def __getitem__(self, _):  # Must return complete, imp_x and impl_y
        support_classes = random.sample(self.all_classes, self.k) if not self.support_classes else self.support_classes
        target_class = random.sample(support_classes, 1)[0]
        if self.unknown_class is None:
            unknown_class = random.sample([x for x in self.all_classes if x not in support_classes], 1)[0]
        else:
            unknown_class = self.unknown_class

        support_set = [[self.get_sample(cl, ss=True) for _ in range(self.n)] for cl in support_classes]

        fin = {}
        for t in support_set[0][0].keys():
            res = []
            for support in support_set:
                res.append({t: np.stack([elem[t] for elem in support])})
            fin[t] = np.stack([elem[t] for elem in res])
        support_set = fin

        target_set = self.get_sample(target_class, path=self.queries[_] if self.queries else None)
        unknown_set = self.get_sample(unknown_class)

        return {'support_set': support_set,
                'target_set': target_set,
                'unknown_set': unknown_set,
                'support_classes': np.stack([self.all_classes.index(elem) for elem in support_classes]),
                'target_class': self.all_classes.index(target_class),
                'unknown_class': self.all_classes.index(unknown_class),
                'known': target_class in support_classes}

    def __len__(self):
        return self.n_task


if __name__ == "__main__":
    from action_rec.hpe.utils.matplotlib_visualizer import MPLPosePrinter
    from action_rec.ar.utils.configuration import TRXTrainConfig

    input_type = "rgb"
    data_path = TRXTrainConfig().data_path
    data_path = "/media/sberti/Data/datasets/NTURGBD_to_YOLO_METRO_122"

    loader = MyLoader(TRXTrainConfig().data_path, input_type=input_type, given_122=not ubuntu,
                      do_augmentation=False, support_classes=['cross_toe_touch', 'cutting_paper_(using_scissors)', 'drink_water', 'eat_meal-snack', 'drop'],
                      query_class='apply_cream_on_face',
                      unknown_class='apply_cream_on_face')
    if input_type in ["skeleton", "hybrid"]:
        vis = MPLPosePrinter()

    for asd in loader:
        sup = asd['support_set']
        trg = asd['target_set']
        unk = asd['unknown_set']
        lab = asd["support_classes"]

        print(asd['support_classes'])
        n_classes, n_examples, n_frames = sup[list(sup.keys())[0]].shape[:3]
        for c in range(n_classes):
            for n in range(n_examples):
                for k in range(n_frames):
                    if input_type in ["rgb", "hybrid"]:
                        cv2.imshow("sup", sup["rgb"][c][n][k].swapaxes(0, 1).swapaxes(1, 2))
                        cv2.waitKey(1)
                    if input_type in ["skeleton", "hybrid"]:
                        vis.set_title(f"{loader.all_classes[lab[c]]}, {n}, {k}")
                        vis.clear()
                        vis.print_pose(sup["sk"][c][n][k], loader.edges)
                        vis.sleep(0.001)
