import numpy as np
from action_rec.create_engine import create_engine

BATCH_SIZE = 1


if __name__ == "__main__":

    # Image Transformation
    i = {"frame": np.ones(shape=(480, 640, 3), dtype=np.int32),
         "H": np.ones(shape=(BATCH_SIZE, 3, 3), dtype=np.float32)}
    create_engine(  # p,
        'action_rec/hpe/weights/onnxs/image_transformation{}.onnx'.format(BATCH_SIZE),
        'action_rec/hpe/weights/engines/docker/image_transformation{}.engine'.format(BATCH_SIZE),
        i)