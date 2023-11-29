import numpy as np
from action_rec.create_engine import create_engine

BATCH_SIZE = 1


if __name__ == "__main__":
    # YOLO
    i = {"input": np.ones(shape=(1, 3, 256, 256), dtype=np.float32)}
    create_engine(  # p,
        'action_rec/hd/weights/onnxs/yolo.onnx',
        'action_rec/hd/weights/engines/docker/yolo.engine',
        i)
