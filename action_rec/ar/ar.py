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
        self.input_type = input_type
        self.device = device

        self.ar = TRXOS(TRXTrainConfig(input_type=input_type), add_hook=add_hook)
        # Fix dataparallel
        state_dict = torch.load(final_ckpt_path, map_location=torch.device(0))['model_state_dict']
        state_dict = OrderedDict({param.replace('.module', ''): data for param, data in state_dict.items()})
        self.ar.load_state_dict(state_dict)
        self.ar.cuda()
        self.ar.eval()

        # Now
        self.support_set_data = {}
        if input_type in ["skeleton", "hybrid"]:
            self.support_set_data["sk"] = torch.zeros(way, shot, seq_len, n_joints*3).cuda()
        if input_type in ["rgb", "hybrid"]:
            self.support_set_data["rgb"] = torch.zeros(way, shot, seq_len, 3, 224, 224).cuda()
        self.support_set_mask = torch.zeros(way, shot).cuda()
        self.support_set_labels = [None] * way
        self.requires_focus = [None] * way
        self.support_set_features = None
        self.requires_os = [True] * way

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
            return {}, 0, {}, {}

        if len(self.support_set_labels) == 0:  # no class to predict
            return {}, 0, {}, {}

        # Process new frame
        data = {k: torch.FloatTensor(v).cuda() for k, v in data.items()}
        self.previous_frames.append(copy.deepcopy(data))
        if len(self.previous_frames) < self.seq_len:  # few samples
            return {}, 0, {}, {}
        elif len(self.previous_frames) == self.seq_len + 1:
            self.previous_frames = self.previous_frames[1:]  # add as last frame

        # Prepare query with previous frames
        for t in list(data.keys()):
            data[t] = torch.stack([elem[t] for elem in self.previous_frames]).unsqueeze(0)

        # Get SS
        ss = {}
        ss_f = None
        if self.support_set_features is None:
            if self.input_type in ["skeleton", "hybrid"]:
                ss['sk'] = self.support_set_data["sk"].unsqueeze(0)
            if self.input_type in ["rgb", "hybrid"]:
                ss['rgb'] = self.support_set_data["rgb"].unsqueeze(0)
        else:
            ss_f = self.support_set_features
        labels = self.support_set_mask.unsqueeze(0)
        with torch.no_grad():
            outputs = self.ar(ss, labels, data, ss_features=ss_f)  # RGB, POSES

        # Save support features
        if self.support_set_features is None:
            self.support_set_features = outputs['support_features']

        # Softmax
        true_logits = outputs['logits'][:, torch.any(self.support_set_mask, dim=1)]
        few_shot_result = torch.softmax(true_logits.squeeze(0), dim=0).detach().cpu().numpy()
        open_set_result = outputs['is_true'].squeeze(0).detach().cpu().numpy()

        # Return output
        results = {}
        true_labels = list(filter(lambda x: x is not None, self.support_set_labels))
        for k in range(len(true_labels)):
            results[true_labels[k]] = (few_shot_result[k])

        return results, open_set_result, self.requires_focus, self.requires_os

    def remove(self, flag):
        # Compute index to remove
        if flag in self.support_set_labels:
            class_id = self.support_set_labels.index(flag)
            self.support_set_labels[class_id] = None
            self.support_set_mask[class_id] = 0
            self.requires_focus[class_id] = None
            self.requires_os[class_id] = True
            if self.input_type in ["skeleton", "hybrid"]:
                self.support_set_data["sk"][class_id] = 0
            if self.input_type in ["rgb", "hybrid"]:
                self.support_set_data["rgb"][class_id] = 0
            self.support_set_features = None
            return True
        else:
            return False

    def edit_focus(self, flag, value):
        if flag not in self.support_set_labels:
            return flag + " is not in the support set"
        index = self.support_set_labels.index(flag)
        self.requires_focus[index] = bool(int(value))
        return flag + " now has focus value " + str(bool(int(value)))

    def edit_os(self, flag, value):
        if flag not in self.support_set_labels:
            return flag + " is not in the support set"
        index = self.support_set_labels.index(flag)
        self.requires_os[index] = bool(int(value))
        return flag + " now has os value " + str(bool(int(value)))

    def train(self, inp, ss_id):
        if inp['flag'] not in self.support_set_labels:
            if None in self.support_set_labels:
                first_none_pos = self.support_set_labels.index(None)
            else:
                return False
            self.support_set_labels[first_none_pos] = inp['flag']
        class_id = self.support_set_labels.index(inp['flag'])
        if self.input_type in ["skeleton", "hybrid"]:
            self.support_set_data["sk"][class_id][ss_id] = torch.FloatTensor(inp['data']['sk']).cuda()
        if self.input_type in ["rgb", "hybrid"]:
            self.support_set_data["rgb"][class_id][ss_id] = torch.FloatTensor(inp['data']['rgb']).cuda()
        self.requires_focus[class_id] = inp['requires_focus']
        self.requires_os[class_id] = True
        self.support_set_mask[class_id][ss_id] = 1
        self.support_set_features = None
        return True

    def save(self):
        save_loc = os.path.join(self.support_set_path, self.input_type)
        if not os.path.exists(save_loc):
            os.mkdir(save_loc)

        with open(os.path.join(save_loc, "support_set_data.pkl"), 'wb') as outfile:
            pkl.dump(self.support_set_data, outfile)
        with open(os.path.join(save_loc, "requires_focus.pkl"), 'wb') as outfile:
            pkl.dump(self.requires_focus, outfile)
        with open(os.path.join(save_loc, "support_set_labels.pkl"), 'wb') as outfile:
            pkl.dump(self.support_set_labels, outfile)
        with open(os.path.join(save_loc, "support_set_mask.pkl"), 'wb') as outfile:
            pkl.dump(self.support_set_mask, outfile)
        with open(os.path.join(save_loc, "requires_os.pkl"), 'wb') as outfile:
            pkl.dump(self.requires_os, outfile)

        return "Classes saved successfully in " + save_loc

    def load(self):
        load_loc = os.path.join(self.support_set_path, self.input_type)

        with open(os.path.join(load_loc, "support_set_labels.pkl"), 'rb') as pkl_file:
            self.support_set_labels = pkl.load(pkl_file)
        with open(os.path.join(load_loc, "support_set_data.pkl"), 'rb') as pkl_file:
            self.support_set_data = pkl.load(pkl_file)
        with open(os.path.join(load_loc, "requires_focus.pkl"), 'rb') as pkl_file:
            self.requires_focus = pkl.load(pkl_file)
        with open(os.path.join(load_loc, "support_set_mask.pkl"), 'rb') as pkl_file:
            self.support_set_mask = pkl.load(pkl_file)
        with open(os.path.join(load_loc, "requires_os.pkl"), 'rb') as pkl_file:
            self.requires_os = pkl.load(pkl_file)

        self.support_set_features = None
        return f"Loaded {len(self.support_set_labels)} classes from {load_loc}"
