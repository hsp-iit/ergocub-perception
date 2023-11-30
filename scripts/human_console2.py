from utils.concurrency.utils.signals import Signals
from loguru import logger
from utils.logging import setup_logger
from configs.human_console_config import Logging, Network
import PySimpleGUI as sg
import numpy as np
import time
import cv2


sg.theme('DarkTeal9')


setup_logger(level=Logging.level)


class SSException(Exception):
    pass


class HumanConsole(Network.node):

    def __init__(self, spawn_location=None):
        super().__init__(**Network.Args.to_dict())
        self.acquisition_time = 2
        self.values = None
        self.last_os_thr = 0.5
        self.last_fs_thr = 0.5
        while True:
            print("Trying to get list of actions...")
            data = self.read("human_console_visualizer", blocking=True)
            values = data.get("actions", Signals.NOT_OBSERVED)
            if values not in Signals:
                print("Found!")
                self.values = list(values.keys())
                break
        self.actions = Signals.NOT_OBSERVED
        self.is_true = Signals.NOT_OBSERVED
        self.log = ""
        self.input_type = "skeleton"
        self.window_size = 16
        self.lay_actions = [[
                         sg.ProgressBar(1, orientation='h', size=(20, 20), key=f"FS-{key}"),
                         sg.ProgressBar(1, orientation='h', size=(20, 20), key=f"OS-{key}"),
                         sg.Text(key, key=f"ACTION-{key}")] 
                         for key in self.values]
        self.lay_thrs = [[sg.Slider(range=(0, 100), size=(20, 20), orientation='h', key='FS-THR'), sg.Slider(range=(0, 100), size=(20, 20), orientation='h', key='OS-THR')]]
        self.lay_commands = [[sg.Button("Remove", key=f"DELETE", size=(6, 1)),
                              sg.Combo(self.values, size=(20,1), enable_events=False, key=f'DELETEACTION', readonly=True),
                              sg.Combo(["all", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], size=(3,1), enable_events=False, key=f'DELETEID', readonly=True)],
                             [sg.Button("Add", key=f"ADD", size=(6, 1)),
                              sg.Combo(self.values, size=(20,1), enable_events=False, key=f'ADDACTION')],
                             [sg.Button("Debug", key=f"DEBUG", size=(6, 1))],
                             [sg.Text("", key="log")]]
        self.lay_io = [[sg.FileBrowse("Load", file_types=(("Support Set", "*.pkl"),), initial_folder="./action_rec/ar/saved"), sg.In(size=(25,1), key='LOAD', enable_events=True), ],
                       [sg.FileSaveAs("Save", file_types=(("Support Set", "*.pkl"),), initial_folder="./action_rec/ar/saved"), sg.In(size=(25,1), key='SAVE', enable_events=True), ]]
        self.lay_support = [[sg.Image(r'SUPPORT_SET.gif', key="SUPPORT_SET", expand_x=True, expand_y=True)]]

        self.lay_left = [[sg.Text("Scores"), sg.HorizontalSeparator()],
                         [sg.Text('Few Shot', size=(20, 1)), sg.Text('Open Set', size=(20, 1))],
                         [sg.Column(self.lay_actions)],
                         [sg.Text("Thresholds"), sg.HorizontalSeparator()],
                         [sg.Column(self.lay_thrs)],
                         [sg.Text("SS Modifiers"), sg.HorizontalSeparator()],
                         [sg.Column(self.lay_commands)],
                         [sg.Text("SS I/O"), sg.HorizontalSeparator()],
                         [sg.Column(self.lay_io)]]
        self.lay_right = [[sg.Column(self.lay_support, scrollable=True,  vertical_scroll_only=True, expand_x=True, expand_y=True)]]
        self.lay_final = [[sg.Column(self.lay_left, expand_x=True, expand_y=True),
                          sg.VerticalSeparator(),
                          sg.Column(self.lay_right, expand_x=True, expand_y=True)]]
        if spawn_location is not None:
            self.window = sg.Window('Few-Shot Console', self.lay_final, location=spawn_location, resizable=True, finalize=True)
        else:
            self.window = sg.Window('Few-Shot Console', self.lay_final, resizable=True, finalize=True)
    def loop(self, data):
        # EXIT IF NECESSARY
        event, val = self.window.read(timeout=10)
        if event == sg.WIN_CLOSED:
            exit()

        # ACTIONS
        actions = data.get('actions', Signals.MISSING_VALUE)
        is_true = data.get('is_true', Signals.MISSING_VALUE)
        if actions is not Signals.MISSING_VALUE:
            self.actions = actions
            self.is_true = is_true
        if self.actions not in Signals:
            # RESTART IF SS HAS CHANGED
            if self.values != list(self.actions.keys()):
                raise SSException("Support set has changed!")
            # UPDATE SCORES
            if len(self.values) > 0:
                best_action = max(zip(self.actions.values(), self.actions.keys()))[1]
                for key in self.actions:
                    self.window[f"FS-{key}"].update(self.actions[key])
                    if key == best_action:
                        self.window[f"OS-{key}"].update(self.is_true[0])
                        if self.actions[best_action] > val['FS-THR']/100 and self.is_true[0] > val['OS-THR']/100:
                            self.window[f"ACTION-{best_action}"].update(text_color="red")
                        else:
                            self.window[f"ACTION-{best_action}"].update(text_color="white")
                    else:
                        self.window[f"OS-{key}"].update(0.)
                        self.window[f"ACTION-{key}"].update(text_color="white")


        # LOG
        log = data.get('log', Signals.MISSING_VALUE)
        if log is not Signals.MISSING_VALUE:
            self.log = log
        if self.log not in Signals:
            if self.log is not None and self.log != ' ':
                self.window["log"].update(self.log)

            if 'SUPPORT_SET' in self.log:
                raise SSException("Loading new support set gif")

        # REMOVE ACTION
        if "DELETE" in event:
            action = val["DELETEACTION"]
            id_to_remove = val["DELETEID"]
            if len(id_to_remove) == 0:
                self.window["log"].update("Please select all or the id of the action to remove")
            if len(action) == 0:
                self.window["log"].update("Please select the action")
            else:
                if id_to_remove == "all":
                    self.write("console_to_ar", {"command": ("remove_action", action)})
                else:
                    self.write("console_to_ar", {"command": ("remove_example", action, int(id_to_remove))})

        # ADD ACTION
        if "ADD" in event:
            action = val["ADDACTION"]
            if len(action) == 0:
                self.window["log"].update("Please select an existing action or write a new one")
            else:
                [self.window[key].update(disabled=True) for key in ["LOAD", "DELETE", "DEBUG", "DELETEACTION", "ADDACTION", "DELETEID", "OS-THR", "FS-THR", "ADD"]]
                self.add_action(action)
                [self.window[key].update(disabled=False) for key in ["SAVE", "DELETE", "DEBUG", "DELETEACTION", "ADDACTION", "DELETEID", "OS-THR", "FS-THR", "ADD"]]

        # DEBUG
        if "DEBUG" in event:
            self.log = "Processing gif image..."
            self.window["log"].update(self.log)
            self.write("console_to_ar", {"command": ("debug",)})

        # LOAD
        if "LOAD" in event:
            self.log = "Loading support set..."
            self.window["log"].update(self.log)           
            self.write("console_to_ar", {"command": ("load", val["LOAD"])})

        # SAVE
        if "SAVE" in event:
            self.write("console_to_ar", {"command": ("save", val["SAVE"])})

        if event == "__TIMEOUT__":
            if val['FS-THR'] != self.last_fs_thr:
                self.last_fs_thr = val['FS-THR']
                self.write("console_to_ar", {"command": ("fs-thr", val['FS-THR']/100)})
            if val['OS-THR'] != self.last_os_thr:
                self.last_os_thr = val['OS-THR']
                self.write("console_to_ar", {"command": ("os-thr", val['OS-THR']/100)}) 
        
        # UPDATE SUPPORT SET
        self.window["SUPPORT_SET"].UpdateAnimation("SUPPORT_SET.gif", time_between_frames=100)


    def add_action(self, action_name):
        now = time.time()
        self.window["log"].update("WAIT...")
        while (time.time() - now) < 3:
            data = self.read("human_console_visualizer")
            self.loop(data)

        self.window["log"].update("GO!")
        data = [[] for _ in range(self.window_size)]
        i = 0
        off_time = (self.acquisition_time / self.window_size)
        while i < self.window_size:
            start = time.time()
            res = self.read("human_console_visualizer")
            self.loop(res)
            self.window["log"].update("{:.2f}%".format((i / (self.window_size - 1)) * 100))
            # Check if the sample is good w.r.t. input type
            good = self.input_type in ["skeleton", "hybrid"] and "pose"in res.keys() and res["pose"] not in Signals
            good = good or self.input_type == "rgb"
            if good:
                if self.input_type in ["skeleton", "hybrid"]:
                    data[i].append(res["pose"].reshape(-1))  # CAREFUL with the reshape
                if self.input_type in ["rgb", "hybrid"]:
                    data[i].append(res["img_preprocessed"])
                i += 1
            while (time.time() - start) < off_time:  # Busy wait
                continue

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
    loc = None
    while True:
        try:
            h = HumanConsole(spawn_location=loc)
            h.run()
        except SSException as e:
            loc = h.window.current_location(more_accurate=True)
            h.window.close()
            print(e)
            continue
