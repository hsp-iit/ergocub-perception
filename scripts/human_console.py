import os
import cv2

os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
from vispy import app, scene, visuals
from vispy.scene.visuals import Text, Image
import numpy as np
import math
from pathlib import Path
from loguru import logger
import sys

sys.path.insert(0, Path(__file__).parent.parent.as_posix())
from utils.logging import setup_logger
from configs.human_console_config import Logging, Network

setup_logger(level=Logging.level)

"""
install with conda install -c conda-forge vispy
check backend with python -c "import vispy; print(vispy.sys_info())"
if not pyqt5, install it with pip install pyqt5
"""


def get_color(value):
    if 0 <= value < 0.33:
        return "red"
    if 0.33 < value < 0.66:
        return "orange"
    if 0.66 < value <= 1:
        return "green"
    raise Exception("Wrong argument:", value)


@logger.catch(reraise=True)
class VISPYVisualizer(Network.node):

    def printer(self, x):
        if x.text == '\b':
            if len(self.input_text) > 1:
                self.input_text = self.input_text[:-1]
            self.log_text.text = ''
        elif x.text == '\r':
            self._send_all({"human_console_commands": {"msg": self.input_text[1:]}}, blocking=False)
            # self.output_queue.put(self.input_text[1:])  # Do not send '>'
            self.input_text = '>'
            self.log_text.text = ''
        # elif x.text == '\\':
        #     self.show = not self.show
        elif x.text == '`':
            self.os = not self.os
        else:
            self.input_text += x.text
        self.input_string.text = self.input_text

    def __init__(self):
        super().__init__(**Network.Args.to_dict())

    def startup(self):
        self.input_text = '>'

        self.canvas = scene.SceneCanvas(keys='interactive')
        self.canvas.size = 1200, 600
        self.canvas.events.key_press.connect(self.printer)
        self.canvas.show()

        self.os = True

        # This is the top-level widget that will hold three ViewBoxes, which will
        # be automatically resized whenever the grid is resized.
        grid = self.canvas.central_widget.add_grid()

        # Plot
        b1 = grid.add_view(row=0, col=0)
        b1.border_color = (0.5, 0.5, 0.5, 1)
        b1.camera = scene.TurntableCamera(45, elevation=30, azimuth=0, distance=2)
        self.lines = []
        Plot3D = scene.visuals.create_visual_node(visuals.LinePlotVisual)
        for _ in range(30):
            self.lines.append(Plot3D(
                [],
                width=3.0,
                color="purple",
                edge_color="w",
                symbol="o",
                face_color=(0.2, 0.2, 1, 0.8),
                marker_size=1,
            ))
            b1.add(self.lines[_])
        coords = scene.visuals.GridLines(parent=b1.scene)

        # Info
        self.b2 = grid.add_view(row=0, col=1)
        self.b2.camera = scene.PanZoomCamera(rect=(0, 0, 1, 1))
        self.b2.camera.interactive = False
        self.b2.border_color = (0.5, 0.5, 0.5, 1)
        self.distance_text = Text('', color='white', rotation=0, anchor_x="center", anchor_y="bottom",
                                  font_size=12, pos=(0.25, 0.9))
        self.b2.add(self.distance_text)
        self.focus_text = Text('', color='green', rotation=0, anchor_x="center", anchor_y="bottom",
                               font_size=12, pos=(0.5, 0.9))
        self.b2.add(self.focus_text)
        self.fps_text = Text('', color='white', rotation=0, anchor_x="center", anchor_y="bottom",
                             font_size=12, pos=(0.75, 0.9))
        self.b2.add(self.fps_text)
        # Actions (LABEL OF INFO)
        self.fsscore = Text('fs score', color='white', rotation=0, anchor_x="center", anchor_y="bottom",
                            font_size=12, pos=(5 / 8, 0.75))
        self.b2.add(self.fsscore)
        self.osscore = Text('os score', color='white', rotation=0, anchor_x="center", anchor_y="bottom",
                            font_size=12, pos=(7 / 8, 0.75))
        self.b2.add(self.osscore)
        self.fsscore = Text('rf', color='white', rotation=0, anchor_x="center", anchor_y="bottom",
                            font_size=12, pos=(7 / 16, 0.75))
        self.b2.add(self.fsscore)
        self.os_score = scene.visuals.Rectangle(center=(2, 2), color="white", border_color="white", height=0.1)
        self.b2.add(self.os_score)
        # Actions
        self.focuses = {}
        self.actions_text = {}
        self.values = {}

        # Image
        b3 = grid.add_view(row=1, col=0)
        b3.camera = scene.PanZoomCamera(rect=(0, 0, 640, 480))
        b3.camera.interactive = False
        b3.border_color = (0.5, 0.5, 0.5, 1)
        self.image = Image()
        b3.add(self.image)

        # Commands
        b4 = grid.add_view(row=1, col=1)
        b4.camera = scene.PanZoomCamera(rect=(0, 0, 1, 1))
        b4.camera.interactive = False
        b4.border_color = (0.5, 0.5, 0.5, 1)
        self.desc_add = Text('ADD ACTION: add action_name ss_id [-focus]', color='white', rotation=0,
                             anchor_x="left",
                             anchor_y="bottom",
                             font_size=10, pos=(0.1, 0.9))
        self.desc_save = Text('SAVE: save', color='white', rotation=0, anchor_x="left",
                              anchor_y="bottom",
                              font_size=10, pos=(0.1, 0.85))
        self.desc_load = Text('LOAD: load', color='white', rotation=0, anchor_x="left",
                              anchor_y="bottom",
                              font_size=10, pos=(0.1, 0.8))
        self.desc_debug = Text('DEBUG: debug', color='white', rotation=0, anchor_x="left",
                               anchor_y="bottom",
                               font_size=10, pos=(0.1, 0.75))
        self.desc_remove = Text('REMOVE ACTION: remove action_name', color='white', rotation=0, anchor_x="left",
                                anchor_y="bottom",
                                font_size=10, pos=(0.1, 0.7))
        self.edit_focus = Text('EDIT FOCUS: edit_focus action_name value', color='white', rotation=0, anchor_x="left",
                                anchor_y="bottom",
                                font_size=10, pos=(0.1, 0.65))
        self.edit_os = Text('EDIT OS: edit_os action_name value', color='white', rotation=0, anchor_x="left",
                                anchor_y="bottom",
                                font_size=10, pos=(0.1, 0.6))
        self.input_string = Text(self.input_text, color='purple', rotation=0, anchor_x="left", anchor_y="bottom",
                                 font_size=12, pos=(0.1, 0.3))
        self.log_text = Text('', color='orange', rotation=0, anchor_x="left", anchor_y="bottom",
                             font_size=12, pos=(0.1, 0.2))
        b4.add(self.desc_add)
        b4.add(self.desc_save)
        b4.add(self.desc_load)
        b4.add(self.desc_debug)
        b4.add(self.desc_remove)
        b4.add(self.edit_focus)
        b4.add(self.edit_os)
        b4.add(self.input_string)
        b4.add(self.log_text)

        # Variables
        self.fps = None
        self.rgb = None
        self.bbox = None
        self.face_bbox = None
        self.focus = None
        self.dist = None
        self.pose = None
        self.edges = None
        self.actions = None
        self.is_true = None
        self.requires_focus = None
        self.log = None
        self.requires_os = None

    def loop(self, elements):
        if not elements:
            return

        # LOG
        if "log" in elements.keys():
            self.log = elements["log"]
            if self.log is not None and self.log != ' ':
                self.log_text.text = self.log

        # FPS
        if "fps_ar" in elements.keys():
            self.fps = elements["fps_ar"]
            self.fps_text.text = "FPS: {:.2f}".format(self.fps if self.fps else 0)

        # FOCUS
        if "focus" in elements.keys():
            self.focus = elements["focus"]
            if self.focus:
                self.focus_text.text = "FOCUS"
                self.focus_text.color = "green"
            else:
                self.focus_text.text = "NOT FOC."
                self.focus_text.color = "red"

        # IMAGE, FACE_BBOX, HUMAN_BBOX
        if "bbox" in elements.keys():
            self.bbox = elements["bbox"]
        if "face_bbox" in elements.keys():
            self.face_bbox = elements["face_bbox"]
        if "rgb" in elements.keys():
            self.rgb = elements["rgb"]
            if self.rgb is not None:
                if self.bbox is not None:
                    x1, x2, y1, y2 = self.bbox
                    self.rgb = cv2.rectangle(self.rgb, (x1, y1), (x2, y2), (0, 0, 255), 3)
                if self.face_bbox is not None:
                    x1, y1, x2, y2 = self.face_bbox
                    color = (255, 0, 0) if not self.focus else (0, 255, 0)
                    self.rgb = cv2.rectangle(self.rgb, (x1, y1), (x2, y2), color, 3)
                self.image.set_data(cv2.flip(self.rgb, 0))

        # DIST
        if "human_distance" in elements.keys():
            self.dist = elements["human_distance"] if elements["human_distance"] != -1 else None
            self.distance_text.text = "DIST: {:.2f}m".format(self.dist) if self.dist is not None else "DIST:"

        # POSE
        if "pose" in elements.keys():
            self.pose = elements["pose"]
            if "edges" in elements.keys():
                self.edges = elements["edges"]  # THIS SHOULD NEVER BE NONE
            if self.pose is not None:
                pose = self.pose @ np.matrix([[1, 0, 0],
                                              [0, math.cos(90), -math.sin(90)],
                                              [0, math.sin(90), math.cos(90)]])
                for i, edge in enumerate(self.edges):
                    self.lines[i].set_data((pose[[edge[0], edge[1]]]),
                                           color="purple",
                                           edge_color="white")
            else:
                for i in range(len(self.lines)):
                    self.lines[i].set_data(color="grey",
                                           edge_color="white")

        # ACTIONS
        if "actions" in elements.keys():
            self.actions = elements["actions"]
            if "is_true" in elements.keys():
                self.is_true = elements["is_true"]
            if "requires_focus" in elements.keys():
                self.requires_focus = elements["requires_focus"]
            if "requires_os" in elements.keys():
                self.requires_os = elements["requires_os"]

            # Actions
            if self.actions is not None:

                m = max(self.actions.values()) if len(self.actions) > 0 else 0  # Just max
                for i, action in enumerate(self.actions.keys()):
                    if action is None:
                        continue
                    # The following line prevent bugs when creating rectangle with no width
                    if self.actions[action] == 0:
                        score = 0.001
                    else:
                        score = self.actions[action]
                    if action in self.actions_text.keys():  # Action was already in SS
                        text = action
                        self.actions_text[action].text = text
                        self.values[action].width = score * 0.25
                        self.actions_text[action].pos = (3 / 16, 0.6 - (0.1 * i))
                        self.values[action].center = (4 / 8 + ((score * 0.25) / 2), 0.6 - (0.1 * i))
                        self.values[action].color = get_color(score)
                        self.values[action].border_color = get_color(score)
                        if action in self.focuses.keys():
                            self.focuses[action].color = 'red' if not self.focus else 'green'
                            self.focuses[action].border_color = 'red' if not self.focus else 'green'
                    else:  # Action must be added in SS
                        # Action label
                        self.actions_text[action] = Text('', rotation=0, anchor_x="center", anchor_y="center",
                                                         font_size=12,
                                                         pos=(3 / 16, 0.6 - (0.1 * i)), color="white")
                        self.b2.add(self.actions_text[action])
                        self.values[action] = scene.visuals.Rectangle(
                            center=(4 / 8 + ((score * 0.25) / 2), 0.6 - (0.1 * i)),
                            color=get_color(score), border_color=get_color(score), height=0.1,
                            width=score * 0.25)
                        self.b2.add(self.values[action])
                        # Eye for focus
                        if self.requires_focus[i]:
                            self.focuses[action] = scene.visuals.Rectangle(center=(7 / 16, 0.6 - (0.1 * i)),
                                                                           color='red' if not self.focus else 'green',
                                                                           border_color='red' if not self.focus else 'green',
                                                                           height=0.1, width=0.05)
                            self.b2.add(self.focuses[action])
                    # Os score
                    self.actions_text[action].color = "white"
                    if score == m:  # If action is None, we exit at the beginning
                        if self.requires_os[i]:
                            self.is_true = self.is_true + 0.001 if self.is_true < 0.1 else self.is_true
                            self.os_score.color = get_color(self.is_true)
                            self.os_score.border_color = get_color(self.is_true)
                        else:
                            self.is_true = 1
                            self.os_score.color = 'white'
                            self.os_score.border_color = 'white'

                        self.os_score.width = self.is_true * 0.25
                        self.os_score.center = [(6 / 8) + ((self.is_true * 0.25) / 2), 0.6 - (0.1 * i)]

                        if self.is_true > 0.66:
                            if self.requires_focus[i]:
                                self.actions_text[action].color = "green" if self.focus else "orange"
                            else:
                                self.actions_text[action].color = "green"
                # Remove erased action (if any)
                to_remove = []
                for key in self.actions_text.keys():
                    if key not in self.actions.keys():
                        to_remove.append(key)
                for key in to_remove:
                    self.actions_text[key].parent = None
                    self.values[key].parent = None
                    self.actions_text.pop(key)
                    self.values.pop(key)
                    if key in self.focuses.keys():
                        self.focuses[key].parent = None
                        self.focuses.pop(key)
                if len(self.actions_text) == 0:
                    self.os_score.center = (2, 2)  # MOVE OUTSIDE
        app.process_events()
        return {}


if __name__ == "__main__":
    human_console = VISPYVisualizer()
    human_console.run()
