import numpy as np
from action_rec.create_engine import create_engine

BATCH_SIZE = 1


if __name__ == "__main__":

    # Heads
    i = {"input": np.ones(shape=(81920*BATCH_SIZE,), dtype=np.float32)}
    create_engine(  # p,
        'action_rec/hpe/weights/onnxs/heads{}.onnx'.format(BATCH_SIZE),
        'action_rec/hpe/weights/engines/docker/heads{}.engine'.format(BATCH_SIZE),
        i)
