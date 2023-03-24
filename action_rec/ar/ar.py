from collections import OrderedDict
from .utils.model import TRXOS
import torch
import copy
import pickle as pkl
import os
from .utils.configuration import TRXTrainConfig


class ActionRecognizer:
    def __init__(self, input_type=None, device=None, add_hook=False, final_ckpt_path=None, seq_len=None, way=None,
                 n_joints=None, support_set_path=None, shot=None):
        """                               { "rgb": []
        self.support_set["action_name"] = { "sk": []
                                          { "features": []
        """
        self.input_type = input_type
        self.device = device

        self.ar = TRXOS(TRXTrainConfig(input_type=input_type), add_hook=add_hook)
        # Fix dataparallel
        state_dict = torch.load(final_ckpt_path, map_location=torch.device(0))['model_state_dict']
        state_dict = OrderedDict({param.replace('.module', ''): data for param, data in state_dict.items()})
        self.ar.load_state_dict(state_dict)
        self.ar.cuda()
        self.ar.eval()

        self.support_set = {}
        self.support_set_features = None

        self.previous_frames = []
        self.seq_len = seq_len
        self.way = way
        self.shot = shot
        self.n_joints = n_joints if input_type == "skeleton" else 0
        self.support_set_path = support_set_path

    def inference(self, data):
        """
        It receives an iterable of data that contains poses, images or both
        """
        if data is None or len(data) == 0:
            return {}, 0

        if len(self.support_set) == 0:  # no class to predict
            return {}, 0

        # Process new frame
        data = {k: torch.FloatTensor(v).cuda() for k, v in data.items()}
        self.previous_frames.append(copy.deepcopy(data))
        if len(self.previous_frames) < self.seq_len:  # few samples
            return {}, 0
        elif len(self.previous_frames) == self.seq_len + 1:
            self.previous_frames = self.previous_frames[1:]  # add as last frame

        # Prepare query with previous frames
        for t in list(data.keys()):
            data[t] = torch.stack([elem[t] for elem in self.previous_frames]).unsqueeze(0)

        # Get SS
        class_features = []
        ss_labels = []
        count = 0
        for c in self.support_set.keys():
            class_features.append(torch.stack([elem for elem in self.support_set[c]["features"]]))
            ss_labels = ss_labels + [count]*len(self.support_set[c]["features"])
            count += 1
        ss_f = torch.concatenate(class_features)
        ss_labels = torch.LongTensor(ss_labels).cuda()

        with torch.no_grad():
            outputs = self.ar(None, ss_labels, data, ss_features=ss_f)  # Data is query

        # Softmax
        true_logits = outputs['logits']
        few_shot_result = torch.softmax(true_logits.squeeze(0), dim=0).detach().cpu().numpy()
        open_set_result = outputs['is_true'].squeeze(0).detach().cpu().numpy()

        # Return output
        results = {}
        for i, l in enumerate(list(self.support_set.keys())):
            results[l] = few_shot_result[i]

        return results, open_set_result

    def remove_action(self, flag):
        self.support_set.pop(flag)
        return "Action {} removed".format(flag)

    def remove_example(self, flag, ss_id):
        for key in self.support_set[flag].keys():
            self.support_set[flag][key].pop(ss_id)
        return "Example {} of action {} removed".format(flag, ss_id)

    def train(self, inp):
        if inp["flag"] not in self.support_set.keys():
            self.support_set[inp["flag"]] = {"sk": [], "rgb": [], "features": []}

        self.support_set[inp["flag"]]["sk"].append(inp['data']['sk'])
        self.support_set[inp["flag"]]["features"].append(self.ar.features_extractor["sk"](torch.FloatTensor(inp['data']['sk']).cuda()))
        return "Action {} learned successfully".format(inp["flag"])

    def save(self):
        save_loc = os.path.join(self.support_set_path, self.input_type)
        if not os.path.exists(save_loc):
            os.mkdir(save_loc)
        with open(os.path.join(save_loc, "support_set.pkl"), 'wb') as outfile:
            pkl.dump(self.support_set, outfile)
        return "Classes saved successfully in " + save_loc

    def load(self):
        load_loc = os.path.join(self.support_set_path, self.input_type)

        with open(os.path.join(load_loc, "support_set.pkl"), 'rb') as pkl_file:
            self.support_set = pkl.load(pkl_file)
        for action in self.support_set.keys():
            self.support_set[action]["features"] = []
            for demonstration in self.support_set[action]["sk"]:
                f = self.ar.features_extractor["sk"](torch.FloatTensor(demonstration).cuda())
                self.support_set[action]["features"].append(f)
        return f"Loaded {len(self.support_set)} classes from {load_loc}"

    def save_ss_image(self):
        import numpy as np
        import cv2
        import pickle
        import imageio
        #
        # ss = self.support_set_data
        #
        # labels = self.support_set_labels
        # if len(ss) == 0:
        #     return "Support set is empty"
        # if self.input_type in ["hybrid", "rgb"]:
        #     ss_rgb = ss["rgb"].detach().cpu().numpy()
        #     ss_rgb = ss_rgb.swapaxes(-2, -3).swapaxes(-1, -2)
        #     ss_rgb = (ss_rgb - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        #     ss_rgb = (ss_rgb * 255).astype(np.uint8)
        #     way, shot, seq_len, height, width, _ = ss_rgb.shape
        #     # Flat image
        #     # ss_rgb = ss_rgb.swapaxes(0, 2)
        #     # ss_rgb = ss_rgb.reshape(seq_len, shot, way*height, width, 3)
        #     sequences = []
        #     for w in range(way):
        #         for s in range(shot):
        #             support_class = ss_rgb[w][s].swapaxes(0, 1).reshape(height, seq_len * width, 3)
        #             support_class = cv2.putText(support_class, f"{labels[w]}, {s}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
        #                                         2,
        #                                         (255, 255, 255), 3, 2)
        #             sequences.append(support_class)
        #     ss_rgb = np.concatenate(sequences, axis=0)
        #     cv2.imwrite("SUPPORT_SET.png", ss_rgb)
        if self.input_type in ["hybrid", "skeleton"]:  # TODO MAKE IT BETTER
            with open(os.path.join("action_rec", "hpe", "assets", "skeleton_types.pkl"), "rb") as input_file:
                edges = pickle.load(input_file)['smpl+head_30']['edges']
            # Get maximum number of supports
            max_num_support = 0
            for k in self.support_set.keys():
                if len(self.support_set[k]["sk"]) > max_num_support:
                    max_num_support = len(self.support_set[k]["sk"])
            # Write gif
            size = 250
            support_gifs = []
            for class_name in self.support_set.keys():
                sks = np.stack(self.support_set[class_name]["sk"])
                sks = sks.reshape(sks.shape[:-1] + (30, 3))  # 5, 16, 30 , 3
                sks *= 2
                sks = ((sks + 1)/2)*size  # Send each pose from [-1, +1] to [0, size]
                sks = sks[..., :2]  # 5, 16, 30 , 2
                class_gif = []
                for i in range(sks.shape[1]):  # Repeat 16 times
                    class_visual = np.zeros((size*2, size*max_num_support))
                    class_visual = cv2.putText(class_visual, class_name, (30, int(size/2)), cv2.FONT_HERSHEY_SIMPLEX, 3,
                                               (255, 255, 255), 2, 2)
                    for j in range(sks.shape[0]):
                        class_visual = cv2.putText(class_visual, f"{j}", (int((size*j) + size/2), int(size*(4/5))),
                                                   cv2.FONT_HERSHEY_SIMPLEX, 1,
                                                   (255, 255, 255), 2, 2)
                        for point in sks[j][i]:
                            class_visual = cv2.circle(class_visual, (int(point[0] + j*size),  # x
                                                                     int(point[1] + size)), 1, (255, 0, 0))  # y
                        for edge in edges:
                            class_visual = cv2.line(class_visual,
                                                    (int(sks[j][i][edge[0]][0] + j*size),  # x1
                                                     int(sks[j][i][edge[0]][1] + size)),  # y1
                                                    (int(sks[j][i][edge[1]][0] + j*size),  # x2
                                                     int(sks[j][i][edge[1]][1] + size)),  # y2
                                                    (255, 0, 0))
                    class_gif.append(class_visual)
                support_gifs.append(class_gif)

            n_classes = len(support_gifs)
            support_gifs = [np.stack(elem) for elem in support_gifs]
            support_gifs = np.stack(support_gifs)
            support_gifs = np.swapaxes(support_gifs, 0, 1)
            support_gifs = np.reshape(support_gifs, (16, size * n_classes * 2, size * max_num_support))
            support_gifs = [elem.astype(np.uint8) for elem in support_gifs]
            imageio.mimsave('SUPPORT_SET.gif', support_gifs, fps=12)

        return "Support set image save to SUPPORT_SET.gif"
