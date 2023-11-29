import numpy as np
from action_rec.create_engine import create_engine

BATCH_SIZE = 1


if __name__ == "__main__":

    # BackBone
    i = {"images": np.ones(shape=(BATCH_SIZE, 256, 256, 3), dtype=np.float32)}
    create_engine(  # p,
        'action_rec/hpe/weights/onnxs/bbone{}.onnx'.format(BATCH_SIZE),
        'action_rec/hpe/weights/engines/docker/bbone{}.engine'.format(BATCH_SIZE),
        i)
