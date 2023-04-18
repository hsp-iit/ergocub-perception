from logging import INFO

import numpy as np

from grasping.denoising import DbscanDenoiser
from grasping.grasp_detection import RansacGraspDetectorTRT
from grasping.segmentation.fcn.fcn_segmentator_trt import FcnSegmentatorTRT
from grasping.shape_completion.confidence_pcr.decoder import ConfidencePCRDecoder
from grasping.shape_completion.confidence_pcr.encoder import ConfidencePCRDecoderTRT
from utils.concurrency.generic_node import GenericNode
from utils.concurrency.ipc_queue import IPCQueue
from utils.concurrency.py_queue import PyQueue
from utils.concurrency.utils.signals import Signals
from utils.confort import BaseConfig


class Logging(BaseConfig):
    class Logger:
        class Params:
            level = INFO  # Minimum logging level or list of logging levels
            recurring = False

    debug = True
    # options: rgb depth mask 'fps center hands partial scene reconstruction transform
    keys = {'distance': -1, 'hands': np.full([4, 4, 2], -1.)}


class Network(BaseConfig):
    node = GenericNode

    class Args:
        in_queues = {
            'segmentation': PyQueue(ip="localhost", port=50000, queue_name='seg_to_sc', blocking=True),
        }

        out_queues = {
            'to_rpc': PyQueue(ip="localhost", port=50000, queue_name='gd_to_rpc', write_format={'hands': np.full([4, 4, 2], -1.)}),
            
            'visualizer': PyQueue(ip="localhost", port=50000, queue_name='visualizer',
                                  write_format={k: Signals.NOT_OBSERVED for k in ['hands', 'fps_od', 'obj_distance']}),
            '3d_visualizer': PyQueue(ip="localhost", port=50000, queue_name='3d_visualizer',
                                     write_format={k: Signals.NOT_OBSERVED for k in
                                                   ['reconstruction', 'transform', 'scene',
                                                    'hands', 'vertices', 'rgb', 'depth']}),
        }

class Denoiser(BaseConfig):
    model = DbscanDenoiser

    class Args:
        # DBSCAN parameters
        eps = 0.05
        min_samples = 30


class ShapeCompletion(BaseConfig):
    class Encoder:
        model = ConfidencePCRDecoderTRT

        class Args:
            engine_path = 'grasping/shape_completion/confidence_pcr/trt/assets/pcr_docker.engine'

    class Decoder:
        model = ConfidencePCRDecoder

        class Args:
            no_points = 10_000
            steps = 20
            thr = 0.5


class GraspDetection(BaseConfig):
    model = RansacGraspDetectorTRT
        
class RANSAC(BaseConfig):
    class Args:
        engine_path = './grasping/grasp_detection/ransac_gd/trt/assets/ransac200_10000_docker.engine'
        tolerance = 0.01
        iterations = 10000

    class Tracker:
        # 0.2 more sensitive to small movement but tracks better
        # 0.3 worse tracking but less sensitive to small movements (might be better for closing the loop)
        update_thr=0.1
        distance_thr = 0.01