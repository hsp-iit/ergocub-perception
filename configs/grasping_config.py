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
from utils.concurrency.yarp_queue import YarpQueue
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
    node = GenericNode

    class Args:

        in_queues = {
            # in_port_name, out_port_name, data_type, out_name
            'segmentation': PyQueue(ip="localhost", port=50000, queue_name='seg_to_sc', blocking=True),
            # 'camera_pose': YarpQueue(remote_port_name='/ergocub-rs-pose/pose:o',
            #                          local_port_name='/VisualPerception/ShapeCompletion/camera_pose:i',
            #                          data_type='list', read_format='camera_pose'),
        }

        out_queues = {
            'visualizer': PyQueue(ip="localhost", port=50000, queue_name='visualizer',
                                  write_format={k: None for k in ['hands', 'fps_od', 'distance']}),
            '3d_visualizer': PyQueue(ip="localhost", port=50000, queue_name='3d_visualizer',
                                     write_format={k: None for k in ['reconstruction', 'partial', 'transform', 'scene',
                                                                     'hands', 'planes', 'lines', 'vertices']}),
            'rpc': IPCQueue(ipc_key=1234, write_format={'distance': -1, 'hands': np.full([4, 4, 2], -1.)})
        }

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
