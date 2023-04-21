from utils.concurrency.utils.signals import Signals
from loguru import logger
from utils.logging import setup_logger
from configs.human_console_config import Logging, Network
import PySimpleGUI as sg
import numpy as np
import time

setup_logger(level=Logging.level)


class SSException(Exception):
    pass


@logger.catch(reraise=True)
class HumanConsole(Network.node):

    def __init__(self):
        super().__init__(**Network.Args.to_dict())
        self.acquisition_time = 2
        self.values = None
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
        self.layout1 = [[sg.Button("Delete", key=f"DELETE-{key}"),
                         sg.Button("Add", key=f"AUG-{key}"),
                         sg.ProgressBar(1, orientation='h', size=(20, 20), key=f"FS-{key}"),
                         sg.ProgressBar(1, orientation='h', size=(20, 20), key=f"OS-{key}"),
                         sg.Text(key, key="action")]
                        for key in self.values]
        self.layout2 = [[sg.Text("log", key="log")]]
        self.layout3 = [[sg.Input('', enable_events=True, key='TO_LEARN', font=('Arial Bold', 20), expand_x=True, justification='left'), sg.Button("Add action", key="ADD")]]
        self.layout4 = [[sg.Button("Debug", key=f"DEBUG")]]
        self.layout5 = [[sg.Text('Load'), sg.In(size=(25,1), enable_events=True, key='LOAD'), sg.FileBrowse(file_types=(("Support Set", "*.pkl"),))],
                        [sg.Text('Save'), sg.In(size=(25,1), enable_events=True, key='SAVE'), sg.FileSaveAs(file_types=(("Support Set", "*.pkl"),))]]
        self.layout6 = [[sg.Column(self.layout1)],
                        [sg.Column(self.layout2)],
                        [sg.Column(self.layout3)],
                        [sg.Column(self.layout4)],
                        [sg.Column(self.layout5)]]
        self.window = sg.Window('Few-Shot Console', self.layout6)

    def loop(self, data):
        # EXIT IF NECESSARY
        event, val = self.window.read(timeout=10)
        # print(event, val)
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
                    else:
                        self.window[f"OS-{key}"].update(0.)                   

        # LOG
        log = data.get('log', Signals.MISSING_VALUE)
        if log is not Signals.MISSING_VALUE:
            self.log = log
        if self.log not in Signals:
            if self.log is not None and self.log != ' ':
                self.window["log"].update(self.log)

        # REMOVE ACTION
        if "DELETE" in event:
            action = event.split('-')[1]
            self.write("console_to_ar", {"command": ("remove_action", action)})

        # ADD ACTION
        if "ADD" in event:
            action = val["TO_LEARN"]
            self.add_action(action)

        # AUG ACTION
        if "AUG" in event:
            action = event.split('-')[1]
            self.add_action(action)

        # DEBUG
        if "DEBUG" in event:
            self.log = "Processing gif image..."
            self.window["log"].update(self.log)
            self.write("console_to_ar", {"command": ("debug",)})

        # LOAD
        if "LOAD" in event:
            self.write("console_to_ar", {"command": ("load", val["LOAD"])})

        # SAVE
        if "SAVE" in event:
            self.write("console_to_ar", {"command": ("save", val["SAVE"])})

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
    while True:
        try:
            h = HumanConsole()
            h.run()
        except SSException as e:
            h.window.close()
            print(e)
            continue
