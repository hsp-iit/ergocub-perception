import pickle
from action_rec.hpe.utils.misc import homography, is_within_fov, reconstruct_absolute
import einops
import numpy as np
from utils.human_runner import Runner
from tqdm import tqdm
import cv2
from action_rec.hpe.utils.matplotlib_visualizer import MPLPosePrinter


class HumanPoseEstimator:
    def __init__(self, skeleton=None, image_transformation_engine_path=None, bbone_engine_path=None,
                 heads_engine_path=None, skeleton_types_path=None, expand_joints_path=None, fx=None, fy=None, ppx=None,
                 ppy=None, necessary_percentage_visible_joints=None, width=None, height=None):

        # Intrinsics and K matrix of RealSense
        self.K = np.zeros((3, 3), np.float32)
        self.K[0][0] = fx
        self.K[0][2] = ppx
        self.K[1][1] = fy
        self.K[1][2] = ppy
        self.K[2][2] = 1

        # Load conversions
        self.skeleton = skeleton
        self.expand_joints = np.load(expand_joints_path)
        with open(skeleton_types_path, "rb") as input_file:
            self.skeleton_types = pickle.load(input_file)

        # Load modules
        self.image_transformation = Runner(image_transformation_engine_path)
        self.bbone = Runner(bbone_engine_path)
        self.heads = Runner(heads_engine_path)

        self.necessary_percentage_visible_joints = necessary_percentage_visible_joints

    def estimate(self, rgb, bbox):

        x1, y1, x2, y2 = bbox

        new_K, homo_inv = homography(x1, x2, y1, y2, self.K, 256)

        # Apply homography
        H = self.K @ np.linalg.inv(new_K @ homo_inv)
        bbone_in = self.image_transformation(rgb.astype(int), H.astype(np.float32))

        bbone_in = bbone_in[0].reshape(1, 256, 256, 3)  # [..., ::-1]
        bbone_in_ = (bbone_in / 255.0).astype(np.float32)

        # BackBone
        outputs = self.bbone(bbone_in_)

        # Heads
        logits = self.heads(outputs[0])

        # Get logits 3d
        logits = logits[0].reshape(1, 8, 8, 288)
        _, logits2d, logits3d = np.split(logits, [0, 32], axis=3)
        current_format = 'b h w (d j)'
        logits3d = einops.rearrange(logits3d, f'{current_format} -> b h w d j', j=32)  # 5, 8, 8, 9, 32

        # 3D Softmax
        heatmap_axes = (2, 1, 3)
        max_along_axis = logits3d.max(axis=heatmap_axes, keepdims=True)
        exponential = np.exp(logits3d - max_along_axis)
        denominator = np.sum(exponential, axis=heatmap_axes, keepdims=True)
        res = exponential / denominator

        # 3D Decode Heatmap
        result = []
        for ax in heatmap_axes:
            other_heatmap_axes = tuple(other_ax for other_ax in heatmap_axes if other_ax != ax)
            summed_over_other_heatmap_axes = np.sum(res, axis=other_heatmap_axes, keepdims=True)
            coords = np.linspace(0.0, 1.0, res.shape[ax])
            decoded = np.tensordot(summed_over_other_heatmap_axes, coords, axes=[[ax], [0]])
            result.append(np.squeeze(np.expand_dims(decoded, ax), axis=heatmap_axes))
        pred3d = np.stack(result, axis=-1)

        # 2D Softmax
        heatmap_axes = (2, 1)
        max_along_axis = logits2d.max(axis=heatmap_axes, keepdims=True)
        exponential = np.exp(logits2d - max_along_axis)
        denominator = np.sum(exponential, axis=heatmap_axes, keepdims=True)
        res = exponential / denominator

        # Decode heatmap
        result = []
        for ax in heatmap_axes:
            other_heatmap_axes = tuple(other_ax for other_ax in heatmap_axes if other_ax != ax)
            summed_over_other_heatmap_axes = np.sum(res, axis=other_heatmap_axes, keepdims=True)
            coords = np.linspace(0.0, 1.0, res.shape[ax])
            decoded = np.tensordot(summed_over_other_heatmap_axes, coords, axes=[[ax], [0]])
            result.append(np.squeeze(np.expand_dims(decoded, ax), axis=heatmap_axes))
        pred2d = np.stack(result, axis=-1) * 255

        # Get absolute position (if desired)
        is_predicted_to_be_in_fov = is_within_fov(pred2d)

        # If less than 1/3 of the joints is visible, then the resulting pose will be weird
        # n < 30
        # if is_predicted_to_be_in_fov.sum() < is_predicted_to_be_in_fov.size*self.necessary_percentage_visible_joints:
        #     return None

        # Move the skeleton into estimated absolute position if necessary
        pred3d = reconstruct_absolute(pred2d, pred3d, new_K[None, ...], is_predicted_to_be_in_fov, weak_perspective=False)

        # # TODO EXP START show pred2d on bbone
        # bbone_aux = copy.deepcopy(bbone_in[0])
        # pred2d = pred2d[0]
        # is_predicted_to_be_in_fov = is_predicted_to_be_in_fov[0]
        # for p, is_fov in zip(pred2d, is_predicted_to_be_in_fov):
        #     bbone_aux = cv2.circle(bbone_aux, (int(p[0]), int(p[1])), 2, (0, 255, 0) if is_fov else (0, 0, 255), 2)
        # cv2.imshow("2d on bbone", bbone_aux.astype(np.uint8))
        # cv2.waitKey(1)
        # # TODO EXP END
        # # TODO EXP START show reconstructed 3d on bbone
        # bbone_aux = copy.deepcopy(bbone_in[0])
        # pred3d_projected = pred3d @ new_K
        # for p in pred3d_projected[0]:
        #     bbone_aux = cv2.circle(bbone_aux, (int(p[0]+(bbone_aux.shape[1]/2)), int(p[1]+(bbone_aux.shape[0]/2))), 2, (0, 255, 0), 2)
        # cv2.imshow("3d on bbone", bbone_aux.astype(np.uint8))
        # cv2.waitKey(1)
        # # TODO EXP END
        # # TODO EXP START
        # pred3d_projected = pred3d @ homo_inv
        # pred3d_projected = pred3d_projected @ self.K
        # for p in pred3d_projected[0]:
        #     frame = cv2.circle(frame, (int(p[0]+(frame.shape[1]/2)), int(p[1]+(frame.shape[0]/2))), 2, (0, 255, 0), 2)
        # cv2.imshow("3d on origin", frame.astype(np.uint8))
        # cv2.waitKey(1)
        # # TODO EXP END

        # Go back in original space (without augmentation and homography)
        pred3d = pred3d @ homo_inv

        # Get correct skeleton
        pred3d = (pred3d.swapaxes(1, 2) @ self.expand_joints).swapaxes(1, 2)
        if self.skeleton is not None:
            pred3d = pred3d[:, self.skeleton_types[self.skeleton]['indices']]
            edges = self.skeleton_types[self.skeleton]['edges']
        else:
            edges = None

        pred3d = pred3d[0]  # Remove batch dimension

        human_distance = np.sqrt(
            np.sum(np.square(np.array([0, 0, 0]) - np.array(pred3d[0])))) * 2.5
        human_position = pred3d[0, :]
        pred3d = pred3d - pred3d[0, :]

        return {"pose": pred3d,
                "edges": edges,
                "human_distance": human_distance,
                "human_position": human_position}


if __name__ == "__main__":
    from configs.action_rec_config import HPE
    from multiprocessing.managers import BaseManager
    import pycuda.autoinit
    from utils.concurrency.pypy_node import connect

    # Connect to realsense
    BaseManager.register('get_queue')
    manager = BaseManager(address=('172.27.192.1', 5000), authkey=b'abracadabra')
    connect(manager)
    send_out = manager.get_queue('windows_out')

    vis = MPLPosePrinter()

    h = HumanPoseEstimator(**HPE.Args.to_dict())

    for _ in tqdm(range(10000)):
        img = send_out.get()["rgb"]
        # cv2.imwrite('test1.jpg', img)
        # img = cv2.imread('test1.jpg')
        r = h.estimate(img)

        if r is not None:

            p = r["pose"]
            e = r["edges"]
            b = r["bbox"]

            if p is not None:
                print(np.sqrt(np.sum(np.square(np.array([0, 0, 0]) - np.array(p[0])))))
                p = p - p[0]
                vis.clear()
                vis.print_pose(p*5, e)
                vis.sleep(0.001)

            if b is not None:
                x1_, x2_, y1_, y2_ = b
                xm = int((x1_ + x2_) / 2)
                ym = int((y1_ + y2_) / 2)
                l = max(xm - x1_, ym - y1_)
                img = img[(ym - l if ym - l > 0 else 0):(ym + l), (xm - l if xm - l > 0 else 0):(xm + l)]
                img = cv2.resize(img, (224, 224))

        cv2.imshow("", img)
        cv2.waitKey(1)
