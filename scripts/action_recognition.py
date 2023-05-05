import numpy as np
import tensorrt  # Avoid Myelin error (DO NOT REMOVE)
from configs.action_rec_config import Network, AR, Logging
import pycuda.autoinit  # Create context on GPU (DO NOT REMOVE)

from utils.concurrency.utils.signals import Signals
from utils.logging import setup_logger


setup_logger(**Logging.Logger.Params.to_dict())


class ActionRecognition(Network.node):
    def __init__(self, input_type, consistency_window_length, consistency_minimum_detection, os_score_thr, fs_score_thr):
        super().__init__(**Network.Args.to_dict())
        self.input_type = input_type
        self.fps_s = []
        self.last_poses = []
        self.last_n_actions = []
        self.consistency_window_length = consistency_window_length
        self.consistency_minimum_detection = consistency_minimum_detection
        self.os_score_thr = os_score_thr
        self.fs_score_thr = fs_score_thr
        self.ar = None

    def startup(self):
        self.ar = AR.model(**AR.Args.to_dict())
        # self.ar.load()

    def loop(self, data):
        elements = {}

        # Human Console Commands, command[0] is command, else are args
        command = data["command"] if "command" in data.keys() else None
        if command is not None:
            if command[0] == "train":
                elements["log"] = self.ar.train(command[1])
            elif command[0] == "remove_action":
                elements["log"] = self.ar.remove_action(command[1])
            elif command[0] == "remove_example":
                elements["log"] = self.ar.remove_example(command[1], command[2])
            elif command[0] == "debug":
                elements["log"] = self.ar.save_ss_image()
            elif command[0] == "save":
                elements["log"] = self.ar.save(command[1])
            elif command[0] == "load":
                self.ar.load(command[1])
            elif command[0] == "fs-thr":
                self.fs_score_thr = command[1]
            elif command[0] == "os-thr":
                self.os_score_thr = command[1]

        ar_input = {}
        pose = data["pose"]
        if pose in Signals:
            return elements

        ar_input["sk"] = pose.reshape(-1)

        # Make inference
        results = self.ar.inference(ar_input)
        actions, is_true = results
        elements["actions"] = actions
        elements["is_true"] = is_true

        # Filter action with os and consistency window
        elements["action"] = "none"
        if len(elements["actions"]) > 0:
            best_action = max(elements["actions"], key=elements["actions"].get)
            # Reject low fs score
            if elements["actions"][best_action] < self.fs_score_thr:
                best_action = "none"
            # Reject low os score
            if is_true < self.os_score_thr:
                best_action = "none"
            # Consistency window
            if len(self.last_n_actions) > self.consistency_window_length:
                self.last_n_actions = self.last_n_actions[1:]
            self.last_n_actions.append(best_action)

            # BEFORE it was considering an action only all the n detected action was that action
            # if all([elem == self.last_n_actions[-1] for elem in self.last_n_actions]):
                # elements["action"] = best_action
            # NOW it takes the action higher frequency in last n frames
            # max_f = 0
            # for i in self.last_n_actions:
            #     freq = self.last_n_actions.count(i)
            #     if freq > max_f:
            #         max_f = freq
            #         best_index = i
            # elements["action"] = best_index
            # LATEST it detect an action if it appears at least K times over N times
            for action in list(set(self.last_n_actions)):
                freq = self.last_n_actions.count(action)
                if freq > self.consistency_minimum_detection:
                    elements["action"] = action

            
        elements["fps_ar"] = self.fps()
        return elements


if __name__ == "__main__":
    m = ActionRecognition(input_type=AR.Main.input_type,
                          consistency_window_length=AR.Main.consistency_window_length,
                          consistency_minimum_detection=AR.Main.consistency_minimum_detection,
                          os_score_thr=AR.Main.os_score_thr,
                          fs_score_thr=AR.Main.fs_score_thr)
    m.run()
