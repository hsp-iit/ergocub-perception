import os

base_dir = "action_rec"


# LEGACY
class FocusModelConfig:
    def __init__(self):
        self.name = 'resnet18'


class FaceDetectorConfig:
    def __init__(self):
        self.mode = 'mediapipe'
        self.mediapipe_max_num_faces = 1
        self.mediapipe_static_image_mode = False


class GazeEstimatorConfig:
    def __init__(self):
        self.camera_params = os.path.join(base_dir, "focus", "gaze_estimation", "assets", "camera_params.yaml")
        self.normalized_camera_params = os.path.join(base_dir, "focus", "gaze_estimation", "assets", 'eth-xgaze.yaml')
        self.normalized_camera_distance = 0.6
        self.checkpoint = os.path.join(base_dir, "focus", "gaze_estimation", "weights", "raw", 'eth-xgaze_resnet18.pth')
        self.image_size = [224, 224]


class FocusConfig:
    def __init__(self):
        # GAZE ESTIMATION
        self.face_detector = FaceDetectorConfig()
        self.gaze_estimator = GazeEstimatorConfig()
        self.model = FocusModelConfig()
        self.mode = 'ETH-XGaze'
        self.device = 'cuda'
        self.area_thr = 0.03  # head bounding box must be over this value to be close
        self.close_thr = -0.95  # When close, z value over this thr is considered focus
        self.dist_thr = 0.3  # when distant, roll under this thr is considered focus
        self.foc_rot_thr = 0.7  # when close, roll above this thr is considered not focus
        self.patience = 3  # result is based on the majority of previous observations
        self.sample_params_path = os.path.join(base_dir, "assets", "sample_params.yaml")
