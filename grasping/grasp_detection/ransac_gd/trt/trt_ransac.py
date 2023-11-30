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

    def __call__(self, points, eps, iterations, num_planes=1):
        res_planes = np.zeros([num_planes, 4])
        res_points = []
        i = 0

        # We copy points because we don't want to modify the input. aux_points:  has one less surface each iteration,
        # and it's used to generate the subsets on which we fit a plane inp_points: aux_point + a number of random
        # points such that the input to the engine has always the same size
        aux_points = copy.deepcopy(points)

        while len(res_points) != num_planes:
            if aux_points.shape[0] < 100:
                return None

            diff = 200 - aux_points.shape[0]
            if diff < 0:
                inp_points = copy.deepcopy(aux_points)
                idx = farthest_point_sampler(torch.tensor(inp_points[None]), 200)[0].numpy()
                inp_points = inp_points[idx]
                
                idx = np.random.randint(0, inp_points.shape[0], size=[iterations * 3])
                subsets = inp_points[idx].reshape(iterations, 3, 3)
                
            elif diff > 0:
                idx = np.random.randint(0, aux_points.shape[0], size=[iterations * 3])
                subsets = aux_points[idx].reshape(iterations, 3, 3)
                
                inp_points = np.concatenate([aux_points, np.random.random([diff, 3])])
            else:
                idx = np.random.randint(0, aux_points.shape[0], size=[iterations * 3])
                subsets = aux_points[idx].reshape(iterations, 3, 3)
                inp_points = copy.deepcopy(aux_points)

            scores, planes = self.ransac(inp_points, subsets, eps)
            planes = planes.reshape(iterations, 4)

            naninf = np.any(np.isinf(planes) | np.isnan(planes), axis=1) | np.all(planes == 0, axis=1)
            planes[naninf] = np.array([0, 0, 0, 0])
            scores[naninf] = 0

            parallel = np.any(np.round((planes @ res_planes.T)) >= 1, axis=1)
            scores[parallel] = 0


            # Check that we have actually found planes
            if np.sum(scores) != 0:
                trt_plane = planes[np.argmax(scores)]

                # Capture the points close to the plane
                plane_points_idx = (np.abs(
                    np.concatenate([aux_points, np.ones([aux_points.shape[0], 1])], axis=1) @ trt_plane) < eps)
                            
                new_plane = copy.deepcopy(trt_plane)[None]
                new_plane[..., :3] = new_plane[..., :3] / np.linalg.norm(new_plane[..., :3], axis=1, keepdims=True)

                res_planes[i] = new_plane
                i += 1

                new_points = copy.deepcopy(aux_points[plane_points_idx])
                res_points.append(new_points)
        try:
            aux_points = aux_points[~plane_points_idx]
        except IndexError as e:
            return [res_planes, res_points]

        # Technically they are already normalized but since the engine can approximate values
        # to run faster, we normalize them again.
        return [res_planes, res_points]
    
class RansacTracker:
    def __init__(self, update_thr=0.3, distance_thr=0.01, debug=False) -> None:
        """
        Stores a box instance and the corresponding point cloud up until the box
        "fitness" score on the current point cloud is close to the one of the current
        box (which should have been computed on the current point cloud).
        If the box moves the previous box score should be lower than the current one.
        If it is still the scores should be more or less the same. If the current box
        score is much lower than the previous box score, it is probably an outlier.
        This is why we keep the sign of 'diff'.
        
        update_thr: if the difference between the current and the previous box 
                    doesn't exceed this value, the return value is the previous box.
                    Otherwise, the previous box is updated and the return value is the 
                    current box.
        distance_thr: used to compute the score of a box with respect to the point cloud.
                      the points whose distance to the box is less than this value are 
                      considered "captured" by the box and contribute to its score.
        """
        self.update_thr = update_thr
        self.distance_thr = distance_thr
        self.debug = debug
        
        self.prev_box = None
        self.prev_points = None

    def __call__(self, box, points):
        if self.prev_box is not None:
            if self.debug:
                print(f'curr: {compute_score(box[0], points, self.distance_thr)}', end=' ')
                print(f'prev: {compute_score(self.prev_box[0], points, self.distance_thr)}')
            diff = compute_score(box[0], points, self.distance_thr) - \
                   compute_score(self.prev_box[0], points, self.distance_thr)
           
            if diff < self.update_thr:
                return self.prev_box, self.prev_points

        self.prev_box = box
        self.prev_points = points
        return box, points

def compute_score(planes, pc, threshold):
    tot = 0 
    aux_pc = copy.deepcopy(pc)
    for plane in planes:
        idx = (np.abs(np.concatenate([aux_pc, np.ones([aux_pc.shape[0], 1])], axis=1) @ plane) < threshold) # 0.01
        tot += np.sum(idx)
        aux_pc = aux_pc[~idx]
    return tot / pc.shape[0]
