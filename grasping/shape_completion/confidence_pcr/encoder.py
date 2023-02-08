from grasping.utils.inference import TRTRunner
import torch


class ConfidencePCRDecoderTRT:
    def __init__(self, engine_path):
        self.engine = TRTRunner(engine_path)

    def __call__(self, x):
        res = self.engine(x)
        weights = res[1:]
        weights = [torch.tensor(w).cuda().unsqueeze(0) for w in weights]
        weights = [[weights[i], weights[i + 1], weights[i + 2]] for i in range(0, 12, 3)]

        return weights
