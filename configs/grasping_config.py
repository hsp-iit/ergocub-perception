from logging import INFO

import numpy as np

from grasping.denoising import DbscanDenoiser
from grasping.grasp_detection import RansacGraspDetectorTRT
from grasping.segmentation.fcn.fcn_segmentator_trt import FcnSegmentatorTRT
from grasping.shape_completion.confidence_pcr.decoder import ConfidencePCRDecoder
from grasping.shape_completion.confidence_pcr.encoder import ConfidencePCRDecoderTRT
from utils.concurrency.yarppy_node import YarpPyNode
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

    node = YarpPyNode

    class Args:
        ip = "localhost"
        port = 50000

        in_config = {'realsense': ['rgb', 'depth']}

        out_config = {'visualizer': {k: None for k in ['hands', 'mask', 'fps']},
                      '3d_visualizer': {k: None for k in ['reconstruction', 'partial', 'transform', 'scene', 'hands',
                                                          'planes', 'lines', 'vertices']},
                      'object_detection_rpc': {'distance': -1,
                                               'hands': np.full([4, 4, 2], -1.)}}
        # make the output queue blocking (can be used to put a breakpoint in the sink and debug the process output)
        blocking = False

class Segmentation(BaseConfig):
    model = FcnSegmentatorTRT

    class Args:
        engine_path = './grasping/segmentation/fcn/trt/assets/seg_fp16_docker.engine'


class Denoiser(BaseConfig):
    model = DbscanDenoiser

    class Args:
        # DBSCAN parameters
        eps = 0.05
        min_samples = 10


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

    class Args:
        engine_path = './grasping/grasp_detection/ransac_gd/trt/assets/ransac200_10000_docker.engine'
        # RANSAC parameters
        tolerance = 0.001
        iterations = 10000
