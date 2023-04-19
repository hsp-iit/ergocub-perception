from utils.concurrency.utils.signals import Signals
from loguru import logger
from utils.logging import setup_logger
from configs.human_console_config import Logging, Network
import PySimpleGUI as sg
import numpy as np
import time

setup_logger(level=Logging.level)


@logger.catch(reraise=True)
class HumanConsole(Network.node):

    def __init__(self):
        super().__init__(**Network.Args.to_dict())
        self.values = None
        while True:
            data = self.read("human_console_visualizer", blocking=True)
            values = data.get("actions", Signals.NOT_OBSERVED)
            if values not in Signals:
                self.values = list(values.keys())
            break
        self.actions = Signals.NOT_OBSERVED
        self.log = ""
        self.input_type = "skeleton"
        self.window_size = 16
        self.layout1 = [[sg.Button("Delete", key=f"DELETE-{key}"),
                         sg.Button("Add", key=f"AUG-{key}"),
                         sg.Text(key, key="action"),
                         sg.ProgressBar(1, orientation='h', size=(20, 20), key=key)]
                        for key in self.values]
        self.layout2 = [[sg.Text("log", key="log")]]
        self.layout3 = [[sg.Input('', enable_events=True, key='TO_LEARN', font=('Arial Bold', 20), expand_x=True, justification='left'), sg.Button("Add action", key="ADD")]]
        self.layout4 = [[sg.Button("Debug", key=f"DEBUG")]]
        self.layout5 = [[sg.Column(self.layout1)],
                        [sg.Column(self.layout2)],
                        [sg.Column(self.layout3)],
                        [sg.Column(self.layout4)]]
        self.window = sg.Window('Horizontal Bar', self.layout5)

    def loop(self, data):
        # EXIT IF NECESSARY
        event, val = self.window.read(timeout=10)
        print(event, val)
        # print(data.keys())
        if event == sg.WIN_CLOSED:
            exit()

        # ACTIONS
        actions = data.get('actions', Signals.MISSING_VALUE)
        if actions is not Signals.MISSING_VALUE:
            self.actions = actions
        if self.actions not in Signals:
            if self.values != list(self.actions.keys()):
                exit()
            for key in self.actions:
                self.window[key].update(self.actions[key])

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
            self.write("console_to_ar", {"command": ("debug",)})

    def add_action(self, action_name):
        now = time.time()
        self.window["log"].update("WAIT...")
        while (time.time() - now) < 3:
            data = self.read("human_console_visualizer")
            self.loop(data)

        self.window["log"].update("GO!")
        data = [[] for _ in range(self.window_size)]
        i = 0
        # off_time = (self.acquisition_time / self.window_size)
        while i < self.window_size:
            # start = time.time()
            res = self.read("human_console_visualizer")
            self.loop(res)
            self.window["log"].update("{:.2f}%".format((i / (self.window_size - 1)) * 100))
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
    # Example usage
    # v = {'stand': 0.5,
    #           'hello': 0.3,
    #           'handshake': 0.8,
    #           'lift': 0.3,
    #           'get': 0.8,
    #           'stop': 0.3,
    #           'dab': 0.8,
    #           'rock_paper_scissor': 0.3,
    #           'cross_arms': 0.8,
    #           'pointing_something': 0.8,
    #           'cross_arms2': 0.3,
    #           't_pose': 0.8,
    #           }
    h = HumanConsole()
    h.run()
    h.close()
