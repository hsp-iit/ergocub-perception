import copy
import sys
from collections import deque
import pickle

import cv2
import numpy as np
import zlib

from pathlib import Path
from loguru import logger

sys.path.insert(0, Path(__file__).parent.parent.as_posix())

from configs.recorder_config import Logging, Network
from utils.logging import setup_logger

setup_logger(**Logging.Logger.Args.to_dict())


class Recorder(Network.node):
    def __init__(self):
        super().__init__(**Network.Args.to_dict())
        self.mode = 'record'
        self.length = 300
        self.reservoir = {'rgb': deque(maxlen=self.length), 'depth': deque(maxlen=self.length)}
        self.frame = 0
        self.stop = False
        self.mode_play = False
        self.wait = 1

        self.h = 0
        self.m = 0

    def loop(self, data):

        cv2.imshow('recorder control', np.zeros((50, 100)))

        if not self.mode_play:
            data = self.read_all()

            # TODO without deepcopy the image is always the same. Why?
            rgb, depth = data.get('rgb'), data.get('depth')

            if rgb is not None and depth is not None:
                self.reservoir['rgb'].append(copy.deepcopy(rgb))
                self.reservoir['depth'].append(copy.deepcopy(depth))
                # cv2.imshow('recorder control', rgb)

            # cv2.imshow('recorder control', np.zeros((50, 100)))

            k = cv2.waitKey(33)

        elif self.mode_play:
            if self.frame >= len(self.reservoir['rgb']):
                self.frame = 0
            if self.frame < 0:
                self.frame = len(self.reservoir['rgb']) - 1

            output = {k: v[self.frame] for k, v in self.reservoir.items()}
            self.write_all(output)

            # cv2.imshow('recorder control', self.reservoir['rgb'][self.frame])
            if not self.stop:
                k = cv2.waitKey(22)
                self.frame += 1
            else:
                k = cv2.waitKey(0)

        if k == ord('a'):
            self.frame -= 1
        elif k == ord('d'):
            self.frame += 1
        if k == ord('q'):
            self.frame = 0
        elif k == ord('e'):
            self.frame = len(self.reservoir['rgb']) - 1
        elif k == ord('s'):
            self.stop = not self.stop
            logger.info("stop" if self.stop else "start")
        elif k == ord('m'):
            self.mode_play = not self.mode_play
            self.wait = 33 if self.mode_play else 1
            logger.info(f'Switched mode to {"play" if self.mode_play else "record"}')
        elif k == ord('p'):
            with Path('./input_dump.pkl').open('wb') as f:
                pickle.dump(self.reservoir, f)


if __name__ == '__main__':
    recorder = Recorder()
    recorder.run()
