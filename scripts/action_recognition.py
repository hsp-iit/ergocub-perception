import tensorrt  # Avoid Myelin error (DO NOT REMOVE)
from configs.action_rec_config import Network, AR, Logging
import pycuda.autoinit  # Create context on GPU (DO NOT REMOVE)
from utils.logging import setup_logger


setup_logger(**Logging.Logger.Params.to_dict())


class ActionRecognition(Network.node):
    def __init__(self, input_type, window_size, skeleton_scale, acquisition_time,
                 consistency_window_length, os_score_thr):
        super().__init__(**Network.Args.to_dict())
        self.input_type = input_type
        self.window_size = window_size
        self.fps_s = []
        self.last_poses = []
        self.last_n_actions = []
        self.skeleton_scale = skeleton_scale
        self.consistency_window_length = consistency_window_length
        self.os_score_thr = os_score_thr
        self.ar = None

    def startup(self):
        self.ar = AR.model(**AR.Args.to_dict())
        self.ar.load()

    def loop(self, data):
        elements = {}

        # Human Console Commands
        train_data = data["train"] if "train" in data.keys() else None
        if train_data is not None:
            elements["log"] = self.ar.train(train_data)

        remove_data = data["remove_action"] if "remove_action" in data.keys() else None
        if remove_data is not None:
            elements["log"] = self.ar.remove_action(remove_data)

        remove_example = data["remove_example"] if "remove_example" in data.keys() else None
        if remove_example is not None:
            elements["log"] = self.ar.remove_example(remove_example[0], remove_example[1])

        debug_data = data["debug"] if "debug" in data.keys() else None
        if debug_data is not None:
            elements["log"] = self.ar.save_ss_image()
        save_data = data["save"] if "save" in data.keys() else None
        if save_data is not None:
            elements["log"] = self.ar.save()
        load_data = data["load"] if "load" in data.keys() else None
        if load_data is not None:
            elements["log"] = self.ar.load()

        ar_input = {}
        pose = data["pose"]
        if pose is None:
            return elements

        ar_input["sk"] = pose.reshape(-1)

        # img = data["rgb"]
        # ar_input = {}
        # elements = {}

        # Cap fps
        # if self.last_time is not None:
        #     while (time.time() - self.last_time) < 1 / self.fps and cap_fps:
        #         time.sleep(0.01)
        #     self.fps_s.append(1. / (time.time() - self.last_time))
        #     fps_s = self.fps_s[-10:]
        #     fps = sum(fps_s) / len(fps_s)
        #     elements["fps_ar"] = fps
        # self.last_time = time.time()

        # RGB CASE
        # hpe_res = self.hpe.estimate(img)
        # if self.input_type == "hybrid" or self.input_type == "rgb":
        #     elements["bbox"] = None
        #     elements["img_preprocessed"] = None
        #     if hpe_res is not None:
        #         x1, y1, x2, y2 = hpe_res['bbox']
        #         elements["bbox"] = x1, x2, y1, y2
        #         xm = int((x1 + x2) / 2)
        #         ym = int((y1 + y2) / 2)
        #         l = max(xm - x1, ym - y1)
        #         img_ = img[(ym - l if ym - l > 0 else 0):(ym + l), (xm - l if xm - l > 0 else 0):(xm + l)]
        #         img_ = cv2.resize(img_, (224, 224))
        #         img_ = img_ / 255.
        #         img_ = img_ * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        #         img_ = img_.swapaxes(-1, -3).swapaxes(-1, -2)
        #         ar_input["rgb"] = img_
        #         elements["img_preprocessed"] = img_

        # SKELETON CASE
        # if self.input_type == "hybrid" or self.input_type == "skeleton":
        #     # elements["human_distance"] = None
        #     elements["pose"] = None
        #     # elements["edges"] = None
        #     # elements["bbox"] = None
        #     if hpe_res is not None:
        #         pose, edges, bbox = hpe_res['pose'], hpe_res['edges'], hpe_res['bbox']
        #         if self.edges is None:
        #             self.edges = edges
        #         if pose is not None:
        #             elements["human_distance"] = np.sqrt(
        #                 np.sum(np.square(np.array([0, 0, 0]) - np.array(pose[0])))) * 2.5
        #             pose = pose - pose[0, :]
        #             elements["pose"] = pose
        #             ar_input["sk"] = pose.reshape(-1)
        #         elements["edges"] = edges
        #         if bbox is not None:
        #             elements["bbox"] = bbox

        # Make inference
        results = self.ar.inference(ar_input)
        actions, is_true = results
        elements["actions"] = actions
        elements["is_true"] = is_true

        # Filter action with os and consistency window
        elements["action"] = -1
        if len(elements["actions"]) > 0:
            best_action = max(elements["actions"], key=elements["actions"].get)
            best_index = list(elements["actions"].keys()).index(best_action)
            # Reject low os
            if is_true < self.os_score_thr:
                best_index = -1
            # Consistency window
            if len(self.last_n_actions) > self.consistency_window_length:
                self.last_n_actions = self.last_n_actions[1:]
            self.last_n_actions.append(best_index)

            # BEFORE it was considering an action only all the n detected action was that action
            if all([elem == self.last_n_actions[-1] for elem in self.last_n_actions]):
                elements["action"] = best_index
            # NOW it takes the action higher frequency in last n frames
            # max_f = 0
            # for i in self.last_n_actions:
            #     freq = self.last_n_actions.count(i)
            #     if freq > max_f:
            #         max_f = freq
            #         best_index = i
            # elements["action"] = best_index
        elements["fps_ar"] = self.fps()
        return elements


if __name__ == "__main__":
    m = ActionRecognition(input_type=AR.Main.input_type,
                          window_size=AR.Main.window_size,
                          skeleton_scale=AR.Main.skeleton_scale,
                          acquisition_time=AR.Main.acquisition_time,
                          consistency_window_length=AR.Main.consistency_window_length,
                          os_score_thr=AR.Main.os_score_thr)
    m.run()
