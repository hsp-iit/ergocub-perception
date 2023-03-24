import cv2
# os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
from vispy import app, scene, visuals
app.use_app('glfw')  # Set backend
from vispy.scene.visuals import Text, Image
import numpy as np
from loguru import logger
import time
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
        if x.text == '-':
            if len(self.input_text) > 1:
                self.input_text = self.input_text[:-1]
            self.log_text.text = ''
        elif x.text == '+':
            command = self.input_text[1:].strip().split()
            if command[0] == "add":
                self.add_action(command[1])
            elif command[0] == "remove":
                if len(command) == 2:  # NO ID, REMOVE ALL ACTION
                    self.write("console_to_ar", {"command": ("remove_action", command[1])})
                elif len(command) == 3:  # ALSO ID, REMOVE ONE EXAMPLE
                    self.write("console_to_ar", {"command": ("remove_example", command[1], int(command[2]))})
            elif command[0] == "debug":
                self.write("console_to_ar", {"command": ("debug",)})
            elif command[0] == "load":
                self.write("console_to_ar", {"command": ("load",)})
            elif command[0] == "save":
                self.write("console_to_ar", {"command": ("save",)})
            else:
                self.log_text.text = "Unknown command"
            self.input_text = '>'
        elif x.text == '`':
            self.os = not self.os
        else:
            self.input_text += x.text
        self.input_string.text = self.input_text

    def __init__(self):
        super().__init__(**Network.Args.to_dict())
        self.window_size = 16  # TODO TAKE THIS FROM CONFIGURATION
        self.input_type = "skeleton"  # TODO TAKE THIS FROM CONFIGURATION

    def startup(self):
        self.input_text = '>'

        self.canvas = scene.SceneCanvas(keys='interactive')
        self.canvas.size = 600, 600
        self.canvas.events.key_press.connect(self.printer)
        self.canvas.show()

        self.os = True

        # This is the top-level widget that will hold three ViewBoxes, which will
        # be automatically resized whenever the grid is resized.
        grid = self.canvas.central_widget.add_grid()

        # Info
        self.b2 = grid.add_view(row=0, col=0)
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

        # Commands
        b4 = grid.add_view(row=1, col=0)
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

        # DIST
        if "human_distance" in elements.keys():
            self.dist = elements["human_distance"] if elements["human_distance"] != -1 else None
            self.distance_text.text = "DIST: {:.2f}m".format(self.dist) if self.dist is not None else "DIST:"

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
            if self.actions is not None and len(self.actions) > 0:
                act_vert_off = 0.5/len(self.actions)
                m = max(self.actions.values()) if len(self.actions) > 0 else 0  # Just max
                for i, action in enumerate(self.actions.keys()):
                    if action is None:
                        continue
                    # The following line prevent bugs when creating rectangle with no width
                    if self.actions[action] < 1e-4:
                        score = 0.001
                    else:
                        score = self.actions[action]
                    if action in self.actions_text.keys():  # Action was already in SS
                        text = action
                        self.actions_text[action].text = text
                        self.values[action].width = score * 0.25
                        self.actions_text[action].pos = (3 / 16, 0.6 - (act_vert_off * i))
                        self.values[action].center = (4 / 8 + ((score * 0.25) / 2), 0.6 - (act_vert_off * i))
                        self.values[action].color = get_color(score)
                        self.values[action].border_color = get_color(score)
                        if action in self.focuses.keys():
                            self.focuses[action].color = 'red' if not self.focus else 'green'
                            self.focuses[action].border_color = 'red' if not self.focus else 'green'
                    else:  # Action must be added in SS
                        # Action label
                        self.actions_text[action] = Text('', rotation=0, anchor_x="center", anchor_y="center",
                                                         font_size=12,
                                                         pos=(3 / 16, 0.6 - (act_vert_off * i)), color="white")
                        self.b2.add(self.actions_text[action])
                        self.values[action] = scene.visuals.Rectangle(
                            center=(4 / 8 + ((score * 0.25) / 2), 0.6 - (act_vert_off * i)),
                            color=get_color(score), border_color=get_color(score), height=act_vert_off,
                            width=score * 0.25)
                        self.b2.add(self.values[action])
                    # Os score
                    self.actions_text[action].color = "white"
                    if score == m:  # If action is None, we exit at the beginning
                        self.is_true = self.is_true + 0.001 if self.is_true < 0.1 else self.is_true
                        self.os_score.color = get_color(self.is_true)
                        self.os_score.border_color = get_color(self.is_true)
                        self.os_score.height = act_vert_off
                        self.os_score.width = self.is_true * 0.25
                        self.os_score.center = [(6 / 8) + ((self.is_true * 0.25) / 2), 0.6 - (act_vert_off * i)]

                        if self.is_true > 0.66:
                            self.actions_text[action].color = "green" if self.focus else "orange"
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

    def add_action(self, flag):
        action_name = flag
        # requires_focus = len(flag) == 3 and flag[2] == "-focus"
        now = time.time()
        self.log_text.text = "WAIT..."
        while (time.time() - now) < 3:
            elements = self.read("human_console_visualizer")
            elements.update(self.read("rgb"))
            self.loop(elements)

        self.log_text.text = "GO!"
        data = [[] for _ in range(self.window_size)]
        i = 0
        # off_time = (self.acquisition_time / self.window_size)
        while i < self.window_size:
            # start = time.time()
            res = self.read("human_console_visualizer")
            res.update(self.read("rgb"))
            self.loop(res)
            self.log_text.text = "{:.2f}%".format((i / (self.window_size - 1)) * 100)
            # Check if the sample is good w.r.t. input type
            good = self.input_type in ["skeleton", "hybrid"] and "pose" in res.keys() and res["pose"] is not None
            good = good or self.input_type == "rgb"
            if good:
                if self.input_type in ["skeleton", "hybrid"]:
                    data[i].append(res["pose"].reshape(-1))  # CAREFUL with the reshape
                if self.input_type in ["rgb", "hybrid"]:
                    data[i].append(res["img_preprocessed"])
                i += 1
            # while (time.time() - start) < off_time:  # Busy wait
            #     continue

        inp = {"flag": action_name,
               "data": {}}

        if self.input_type == "rgb":  # Unique case with images in first position
            inp["data"]["rgb"] = np.stack([x[0] for x in data])
        if self.input_type in ["skeleton", "hybrid"]:
            inp["data"]["sk"] = np.stack([x[0] for x in data])
        if self.input_type == "hybrid":
            inp["data"]["rgb"] = np.stack([x[1] for x in data])

        self.write("console_to_ar", {"command": ("train", inp)})


if __name__ == "__main__":
    human_console = VISPYVisualizer()
    human_console.run()
