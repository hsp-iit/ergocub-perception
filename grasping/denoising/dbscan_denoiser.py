import subprocess

import numpy as np
from loguru import logger

try:
    from cuml.cluster.dbscan import DBSCAN
except ImportError:
    logger.warning('Using slow DBSCAN implementation')
    from sklearn.cluster import DBSCAN


class DbscanDenoiser:
    def __init__(self, eps, min_samples):
        self.eps = eps
        self.min_samples = min_samples

        # warmup
        self(np.random.randn(4096, 3))

    def __call__(self, input_pc):

        denoised_pc = None
        if input_pc is not None:
            clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(input_pc)  # 0.1 10 are perfect but slow
            close = clustering.labels_[input_pc.argmax(axis=0)[2]]
            denoised_pc = input_pc[clustering.labels_ == close]

        return denoised_pc
