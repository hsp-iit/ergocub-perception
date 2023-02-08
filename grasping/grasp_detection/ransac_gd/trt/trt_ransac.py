import torch
from dgl.geometry import farthest_point_sampler

from grasping.utils.inference import TRTRunner
import numpy as np
import copy
from loguru import logger

# This class starts the trt engine and run it 6 times to extract
# all the cube faces
class TrTRansac:
    runner = None

    def __init__(self, engine_path):
        self.ransac = TRTRunner(engine_path)

    def __call__(self, points, eps, iterations):
        res_planes = np.zeros([6, 4])
        res_points = []
        i = 0

        # We copy points because we don't want to modify the input. aux_points:  has one less surface each iteration,
        # and it's used to generate the subsets on which we fit a plane inp_points: aux_point + a number of random
        # points such that the input to the engine has always the same size
        aux_points = copy.deepcopy(points)

        while len(res_points) != 6:
            if aux_points.shape[0] < 10:
                return None
            idx = np.random.randint(0, aux_points.shape[0], size=[iterations * 3])
            subsets = aux_points[idx].reshape(iterations, 3, 3)

            diff = 200 - aux_points.shape[0]
            if diff < 0:
                inp_points = copy.deepcopy(aux_points)
                idx = farthest_point_sampler(torch.tensor(inp_points[None]), 200)[0].numpy()
                # idx = np.random.choice(inp_points.shape[0], 5981, replace=False)
                inp_points = inp_points[idx]
            elif diff > 0:
                inp_points = np.concatenate([aux_points, np.random.random([diff, 3])])
            else:
                inp_points = copy.deepcopy(aux_points)

            scores, planes = self.ransac(inp_points, subsets, eps)
            planes = planes.reshape(iterations, 4)

            naninf = np.any(np.isinf(planes) | np.isnan(planes), axis=1) | np.all(planes == 0, axis=1)
            planes[naninf] = np.array([0, 0, 0, 0])
            scores[naninf] = 0

            parallel = np.any(np.round((planes @ res_planes.T)) >= 1, axis=1)
            scores[parallel] = 0

            trt_plane = planes[np.argmax(scores)]

            plane_points_idx = (np.abs(
                np.concatenate([aux_points, np.ones([aux_points.shape[0], 1])], axis=1) @ trt_plane) < eps)

            if np.sum(scores) != 0:
                new_plane = copy.deepcopy(trt_plane)[None]
                new_plane[..., :3] = new_plane[..., :3] / np.linalg.norm(new_plane[..., :3], axis=1, keepdims=True)

                res_planes[i] = new_plane
                i += 1

                new_points = copy.deepcopy(aux_points[plane_points_idx])
                res_points.append(new_points)

            aux_points = aux_points[~plane_points_idx]

        # Technically they are already normalized but since the engine can approximate values
        # to run faster, we normalize them again.
        return [res_planes, res_points]
