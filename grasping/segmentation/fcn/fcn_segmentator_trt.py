from grasping.utils.inference import TRTRunner


class FcnSegmentatorTRT:
    """Performs basic semantic segmentation to the input image returning the segmentation mask.
        Preprocessing is performed by the engine. Find more details inside
         grasping/segmentation/fcn/trt/export_onnx.py"""
    def __init__(self, engine_path):
        self.engine = TRTRunner(engine_path)

    def __call__(self, x):
        res = self.engine(x)
        res = res[0].reshape(192, 256, 1)

        return res
