import sys
from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation
import copy
from utils.concurrency.utils.signals import Signals

sys.path.insert(0, Path(__file__).parent.parent.as_posix())
from utils.logging import setup_logger

from grasping.utils.misc import draw_mask, project_pc, project_hands
from configs.sink_config import Logging, Network

setup_logger(level=Logging.level)


class Sink(Network.node):
    def __init__(self):
        self.img = np.zeros([480, 640, 3], dtype=np.uint8)
        self.mask = Signals.NOT_OBSERVED
        self.center = Signals.NOT_OBSERVED
        self.hands = Signals.NOT_OBSERVED
        self.obj_distance = Signals.NOT_OBSERVED

        self.fps_hd = Signals.NOT_OBSERVED
        self.fps_hpe = Signals.NOT_OBSERVED
        self.fps_ar = Signals.NOT_OBSERVED
        self.fps_focus = Signals.NOT_OBSERVED

        self.human_distance = Signals.NOT_OBSERVED
        self.focus = Signals.NOT_OBSERVED
        self.pose = Signals.NOT_OBSERVED
        self.bbox = Signals.NOT_OBSERVED
        self.face_bbox = Signals.NOT_OBSERVED
        self.edges = Signals.NOT_OBSERVED
        self.is_true = Signals.NOT_OBSERVED
        self.action = Signals.NOT_OBSERVED
        cv2.namedWindow('Ergocub-Visual-Perception', cv2.WINDOW_NORMAL)
        super().__init__(**Network.Args.to_dict())

    def startup(self):
        pass

    def loop(self, data: dict) -> dict:

        rgb = data.get('rgb', None)
        if rgb is not None:
            self.img = data['rgb']
        img = copy.deepcopy(self.img)

        # GRASPING #####################################################################################################
        mask = data.get('mask', Signals.MISSING_VALUE)

        if mask is not Signals.MISSING_VALUE:
            self.mask = mask
        if self.mask not in Signals:
            img = draw_mask(img, self.mask)

        center = data.get('center', Signals.MISSING_VALUE)
        if center is not Signals.MISSING_VALUE:
            self.center = center
        if self.center not in Signals:
            img = cv2.circle(img, project_pc(self.center)[0], 5, (0, 255, 0)).astype(np.uint8)

        hands = data.get('hands', Signals.MISSING_VALUE)
        if hands is not Signals.MISSING_VALUE:
            self.hands = hands
        if self.hands not in Signals:
            img = project_hands(img, self.hands[..., 0], self.hands[..., 1])

        obj_distance = data.get('obj_distance', Signals.MISSING_VALUE)
        if obj_distance is not Signals.MISSING_VALUE:
            self.obj_distance = obj_distance
        if self.obj_distance not in Signals:
            img = cv2.putText(img, f'OBJ DIST: {self.obj_distance / 1000.:.2f}', (450, 470), cv2.FONT_ITALIC, 0.7,
                              (255, 0, 0), 1,
                              cv2.LINE_AA)

        # HUMAN ########################################################################################################
        fps_hd = data.get('fps_hd', Signals.MISSING_VALUE)
        if fps_hd is not Signals.MISSING_VALUE:
            self.fps_hd = fps_hd
        if self.fps_hd not in Signals:
            img = cv2.putText(img, f'FPS HD: {int(self.fps_hd)}', (10, 20), cv2.FONT_ITALIC, 0.7, (0, 0, 255), 2,
                              cv2.LINE_AA)

        fps_hpe = data.get('fps_hpe', Signals.MISSING_VALUE)
        if fps_hpe is not Signals.MISSING_VALUE:
            self.fps_hpe = fps_hpe
        if self.fps_hpe not in Signals:
            img = cv2.putText(img, f'FPS HPE: {int(self.fps_hpe)}', (10, 40), cv2.FONT_ITALIC, 0.7, (0, 0, 255), 2,
                              cv2.LINE_AA)

        fps_ar = data.get('fps_ar', Signals.MISSING_VALUE)
        if fps_ar is not Signals.MISSING_VALUE:
            self.fps_ar = fps_ar
        if self.fps_ar not in Signals:
            img = cv2.putText(img, f'FPS AR: {int(self.fps_ar)}', (10, 60), cv2.FONT_ITALIC, 0.7, (0, 0, 255), 2,
                              cv2.LINE_AA)

        fps_focus = data.get('fps_focus', Signals.MISSING_VALUE)
        if fps_focus is not Signals.MISSING_VALUE:
            self.fps_focus = fps_focus
        if self.fps_focus not in Signals:
            img = cv2.putText(img, f'FPS FOCUS: {int(self.fps_focus)}', (10, 80), cv2.FONT_ITALIC, 0.7, (0, 0, 255), 2,
                              cv2.LINE_AA)

        human_distance = data.get('human_distance', Signals.MISSING_VALUE)
        if human_distance is not Signals.MISSING_VALUE:
            self.human_distance = human_distance
        if self.human_distance not in Signals:
            img = cv2.putText(img, f'DIST: {self.human_distance:.2f}', (240, 20), cv2.FONT_ITALIC, 0.7, (0, 0, 255), 2,
                              cv2.LINE_AA)

        focus = data.get('focus', Signals.MISSING_VALUE)
        if focus is not Signals.MISSING_VALUE:
            self.focus = focus
        if self.focus not in Signals:
            img = cv2.putText(img, "FOCUS" if self.focus else "NOT FOCUS", (460, 20), cv2.FONT_ITALIC, 0.7,
                              (0, 255, 0) if self.focus else (255, 0, 0), 2, cv2.LINE_AA)

        pose = data.get('pose', Signals.MISSING_VALUE)
        if pose is not Signals.MISSING_VALUE:
            self.pose = pose
            self.edges = data["edges"]
        if self.pose not in Signals:
            size = 150
            img = cv2.rectangle(img, (0, 480-size), (size, 480), (255, 255, 255), cv2.FILLED)
            for edge in self.edges:
                p0 = [int((p*size)+size) for p in self.pose[edge[0]][:2]]
                p1 = [int((p*size)+size) for p in self.pose[edge[1]][:2]]
                p0[1] += int(480 - size*1.5)
                p1[1] += int(480 - size*1.5)
                p0[0] += int(-size/2)
                p1[0] += int(-size/2)
                img = cv2.line(img, p0, p1, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

        bbox = data.get('bbox', Signals.MISSING_VALUE)
        if bbox is not Signals.MISSING_VALUE:
            self.bbox = bbox
        if self.bbox not in Signals:
            x1, y1, x2, y2 = self.bbox
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)

        face_bbox = data.get('face_bbox', Signals.MISSING_VALUE)
        if face_bbox is not Signals.MISSING_VALUE:
            self.face_bbox = face_bbox
        if self.face_bbox not in Signals:
            x1, y1, x2, y2 = self.face_bbox
            color = (255, 0, 0) if not self.focus else (0, 255, 0)
            img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)

        action = data.get('action', Signals.MISSING_VALUE)
        if action is not Signals.MISSING_VALUE:
            self.action = action
        if self.action not in Signals:
            if self.obj_distance is Signals.NOT_OBSERVED or self.obj_distance/1000 > 1.5:  # No box in 1 meter
                if self.action != 'none':
                    textsize = cv2.getTextSize(self.action, cv2.FONT_ITALIC, 1, 2)[0]
                    textX = int((img.shape[1] - textsize[0]) / 2)
                    text_color = (0, 255, 0)
                    img = cv2.putText(img, self.action, (textX, 450), cv2.FONT_ITALIC, 1, text_color, 2, cv2.LINE_AA)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cv2.imshow('Ergocub-Visual-Perception', img)
        cv2.setWindowProperty('Ergocub-Visual-Perception', cv2.WND_PROP_TOPMOST, 1)
        cv2.waitKey(1)
        return {}


if __name__ == '__main__':
    source = Sink()
    source.run()
