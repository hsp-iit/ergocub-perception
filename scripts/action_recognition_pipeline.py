import sys
from multiprocessing.managers import BaseManager
from pathlib import Path
import tensorrt  # TODO NEEDED IN ERGOCUB, NOT NEEDED IN ISBFSAR

sys.path.insert(0, Path(__file__).parent.parent.as_posix())
from configs.action_rec_config import Network, HPE, FOCUS, AR, MAIN
import os
import numpy as np
import time
import cv2
from multiprocessing import Process, Queue

docker = os.environ.get('AM_I_IN_A_DOCKER_CONTAINER', False)


class ISBFSAR(Network.node):
    def __init__(self, input_type, cam_width, cam_height, window_size, skeleton_scale, acquisition_time, fps):
        super().__init__(**Network.Args.to_dict())
        self.input_type = input_type
        self.cam_width = cam_width
        self.cam_height = cam_height
        self.window_size = window_size
        self.fps_s = []
        self.last_poses = []
        self.last_n_filtered_actions = []
        self.skeleton_scale = skeleton_scale
        self.acquisition_time = acquisition_time
        self.fps = fps
        self.last_time = None
        self.edges = None
        self.focus_in = None
        self.focus_out = None
        self.focus_proc = None
        self.hpe_in = None
        self.hpe_out = None
        self.hpe_proc = None
        self.ar = None
        self.last_data = None
        self.commands_queue = None
        self.last_log = None

    def startup(self):
        # Load modules
        self.focus_in = Queue(1)
        self.focus_out = Queue(1)
        self.focus_proc = Process(target=run_module, args=(FOCUS, self.focus_in, self.focus_out))
        self.focus_proc.start()

        self.hpe_in = Queue(1)
        self.hpe_out = Queue(1)
        self.hpe_proc = Process(target=run_module, args=(HPE, self.hpe_in, self.hpe_out))
        self.hpe_proc.start()

        self.ar = AR.model(**AR.Args.to_dict())
        self.ar.load()

        # To receive human commands
        BaseManager.register('get_queue')
        manager = BaseManager(address=('localhost', 50000), authkey=b'abracadabra')
        manager.connect()
        self.commands_queue = manager.get_queue("human_console_commands")

    def get_frame(self, img=None, log=None, cap_fps=True):
        """
        get frame, do inference, return all possible info
        keys: img, bbox, img_preprocessed, human_distance, pose, edges, actions, is_true, requires_focus, focus, face_bbox,
        fps
        """
        # Add default keys:values
        elements = {}
        # elements.update(copy.deepcopy(Logging.keys))  # TODO is this needed?

        # If img is not given (not a video), try to get img
        if img is None:
            img = self._recv()["rgbImage"]
        elements["rgb"] = img

        # Msg
        if log is not None:
            self.last_log = log
        elements["log"] = self.last_log

        # Cap fps
        if self.last_time is not None:
            if (time.time() - self.last_time) < 1 / self.fps and cap_fps:
                time.sleep(0.01)
            self.fps_s.append(1. / (time.time() - self.last_time))
            fps_s = self.fps_s[-10:]
            fps = sum(fps_s) / len(fps_s)
            elements["fps_ar"] = fps
        self.last_time = time.time()

        ar_input = {}

        # Start independent modules
        self.focus_in.put(img)
        self.hpe_in.put(img)

        # RGB CASE
        hpe_res = self.hpe_out.get()
        if self.input_type == "hybrid" or self.input_type == "rgb":
            elements["bbox"] = None
            elements["img_preprocessed"] = None
            if hpe_res is not None:
                x1, y1, x2, y2 = hpe_res['bbox']
                elements["bbox"] = x1, x2, y1, y2
                xm = int((x1 + x2) / 2)
                ym = int((y1 + y2) / 2)
                l = max(xm - x1, ym - y1)
                img_ = img[(ym - l if ym - l > 0 else 0):(ym + l), (xm - l if xm - l > 0 else 0):(xm + l)]
                img_ = cv2.resize(img_, (224, 224))
                img_ = img_ / 255.
                img_ = img_ * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img_ = img_.swapaxes(-1, -3).swapaxes(-1, -2)
                ar_input["rgb"] = img_
                elements["img_preprocessed"] = img_

        # SKELETON CASE
        if self.input_type == "hybrid" or self.input_type == "skeleton":
            # elements["human_distance"] = None
            elements["pose"] = None
            elements["edges"] = None
            elements["bbox"] = None
            if hpe_res is not None:
                pose, edges, bbox = hpe_res['pose'], hpe_res['edges'], hpe_res['bbox']
                if self.edges is None:
                    self.edges = edges
                if pose is not None:
                    elements["human_distance"] = np.sqrt(
                        np.sum(np.square(np.array([0, 0, 0]) - np.array(pose[0])))) * 2.5
                    pose = pose - pose[0, :]
                    elements["pose"] = pose
                    ar_input["sk"] = pose.reshape(-1)
                elements["edges"] = edges
                if bbox is not None:
                    elements["bbox"] = bbox

        # Make inference
        results = self.ar.inference(ar_input)
        actions, is_true, requires_focus, requires_os = results
        elements["actions"] = actions
        elements["is_true"] = is_true
        elements["requires_focus"] = requires_focus
        elements["requires_os"] = requires_os

        # FOCUS #######################################################
        focus_ret = self.focus_out.get()
        elements["focus"] = False
        elements["face_bbox"] = None
        if focus_ret is not None:
            focus, face = focus_ret
            elements["focus"] = focus
            elements["face_bbox"] = face.bbox.reshape(-1)

        # set action (for BT) and filtered action (for direct activation)
        elements["action"] = -1
        elements["filtered_action"] = -1
        if len(elements["actions"]) > 0:
            best_action = max(elements["actions"], key=elements["actions"].get)
            best_index = list(elements["actions"].keys()).index(best_action)
            elements["action"] = best_index
            filtered_action = best_index
            if elements["requires_os"][best_index]:
                if is_true < 0.66:
                    filtered_action = -1
            if elements["requires_focus"][best_index]:
                if not elements["focus"]:
                    filtered_action = -1
            if len(self.last_n_filtered_actions) > 16:
                self.last_n_filtered_actions = self.last_n_filtered_actions[1:]
            self.last_n_filtered_actions.append(best_index)
            if not all([elem == self.last_n_filtered_actions[-1] for elem in self.last_n_filtered_actions]):
                filtered_action = -1
            elements["filtered_action"] = filtered_action

        return elements

    def loop(self, data):
        log = None
        if "rgbImage" in data.keys():  # Save last data with image
            self.last_data = data
        else:  # It arrives just a message, but we need all
            data.update(self.last_data)

        if not self.commands_queue.empty():
            msg = self.commands_queue.get()["msg"]
            msg = msg.strip()
            msg = msg.split()

            # select appropriate command
            if msg[0] == 'close' or msg[0] == 'exit' or msg[0] == 'quit' or msg[0] == 'q':
                exit()

            elif msg[0] == "add" and len(msg) > 1:
                log = self.learn_command(msg[1:])
                data = self._recv()

            elif msg[0] == "remove" and len(msg) > 1:
                log = self.forget_command(msg[1])

            elif msg[0] == "save":
                log = self.ar.save()

            elif msg[0] == "load":
                log = self.ar.load()

            elif msg[0] == "debug":
                log = self.debug()

            elif msg[0] == "edit_focus":
                log = self.ar.edit_focus(msg[1], msg[2])

            elif msg[0] == "edit_os":
                log = self.ar.edit_os(msg[1], msg[2])

            else:
                log = "Not a valid command!"
        d = self.get_frame(img=data["rgbImage"], log=log)
        return d

    def forget_command(self, flag):
        if self.ar.remove(flag):
            return "Action {} removed".format(flag)
        else:
            return "Action {} is not in the support set".format(flag)

    def debug(self):
        ss = self.ar.support_set_data

        labels = self.ar.support_set_labels
        if len(ss) == 0:
            return "Support set is empty"
        if self.input_type in ["hybrid", "rgb"]:
            ss_rgb = ss["rgb"].detach().cpu().numpy()
            ss_rgb = ss_rgb.swapaxes(-2, -3).swapaxes(-1, -2)
            ss_rgb = (ss_rgb - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            ss_rgb = (ss_rgb * 255).astype(np.uint8)
            way, shot, seq_len, height, width, _ = ss_rgb.shape
            # Flat image
            # ss_rgb = ss_rgb.swapaxes(0, 2)
            # ss_rgb = ss_rgb.reshape(seq_len, shot, way*height, width, 3)
            sequences = []
            for w in range(way):
                for s in range(shot):
                    support_class = ss_rgb[w][s].swapaxes(0, 1).reshape(height, seq_len * width, 3)
                    support_class = cv2.putText(support_class, f"{labels[w]}, {s}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                                2,
                                                (255, 255, 255), 3, 2)
                    sequences.append(support_class)
            ss_rgb = np.concatenate(sequences, axis=0)
            cv2.imwrite("SUPPORT_SET.png", ss_rgb)
        if self.input_type in ["hybrid", "skeleton"]:
            # ss = np.stack([ss[c]["poses"].detach().cpu().numpy() for c in ss.keys()])
            classes = []
            ss_sk = ss["sk"].detach().cpu().numpy()
            way, shot, _, _ = ss_sk.shape
            for ss_c in ss_sk:  # FOR EACH CLASS, 5, 16, 90
                ss_c = ss_c.reshape(ss_c.shape[:-1] + (30, 3))  # 5, 16, 30 , 3
                size = 250
                zoom = 2
                visual = np.zeros((size * ss_c.shape[0], size * ss_c.shape[1]))
                ss_c = (ss_c + 1) * (size / 2)  # Send each pose from [-1, +1] to [0, size]
                ss_c *= zoom
                ss_c = ss_c[..., :2]
                ss_c[..., 1] += np.arange(ss_c.shape[0])[..., None, None].repeat(ss_c.shape[1], axis=1) * size
                ss_c[..., 0] += np.arange(ss_c.shape[1])[None, ..., None].repeat(ss_c.shape[0], axis=0) * size
                ss_c[..., 1] -= size / 2
                ss_c[..., 0] -= size / 2
                ss_c = ss_c.reshape(-1, 30, 2).astype(int)
                for pose in ss_c:
                    for point in pose:
                        visual = cv2.circle(visual, point, 1, (255, 0, 0))
                    for edge in self.edges:
                        visual = cv2.line(visual, pose[edge[0]], pose[edge[1]], (255, 0, 0))
                classes.append(visual)
            visual = np.concatenate(classes, axis=0)
            for i, label in enumerate(labels):
                visual = cv2.putText(visual, label, (10, 10 + i * size * shot), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                     (255, 255, 255), 1, 2)
            cv2.imwrite("SUPPORT_SET.png", visual)
        return "Support saved to SUPPORT_SET.png"

    def learn_command(self, flag):
        action_name = flag[0]
        try:
            ss_id = int(flag[1])
        except Exception:
            return "Format not valid"
        requires_focus = len(flag) == 3 and flag[2] == "-focus"
        now = time.time()
        while (time.time() - now) < 3:
            self._send_all(self.get_frame(log="WAIT...", cap_fps=False), False)

        self._send_all(self.get_frame(log="GO!", cap_fps=False), False)
        # playsound('assets' + os.sep + 'start.wav')
        data = [[] for _ in range(self.window_size)]
        i = 0
        off_time = (self.acquisition_time / self.window_size)
        while i < self.window_size:
            start = time.time()
            res = self.get_frame(log="{:.2f}%".format((i / (self.window_size - 1)) * 100), cap_fps=False)
            self._send_all(res, False)
            # Check if the sample is good w.r.t. input type
            good = self.input_type in ["skeleton", "hybrid"] and "pose" in res.keys() and res["pose"] is not None
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
               "data": {},
               "requires_focus": requires_focus}

        if self.input_type == "rgb":  # Unique case with images in first position
            inp["data"]["rgb"] = np.stack([x[0] for x in data])
        if self.input_type in ["skeleton", "hybrid"]:
            inp["data"]["sk"] = np.stack([x[0] for x in data])
        if self.input_type == "hybrid":
            inp["data"]["rgb"] = np.stack([x[1] for x in data])
        ret = self.ar.train(inp, ss_id)
        if ret:
            return "Action " + action_name + " learned successfully!"
        else:
            return "Cannot add action"


def run_module(module, input_queue, output_queue):
    import pycuda.autoinit
    x = module.model(**module.Args.to_dict())
    while True:
        inp = input_queue.get()
        y = x.estimate(inp)
        output_queue.put(y)


if __name__ == "__main__":
    m = ISBFSAR(**MAIN.Args.to_dict())
    m.run()
