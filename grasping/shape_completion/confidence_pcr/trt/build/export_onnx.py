import sys
sys.path.append("C:\\Users\\arosasco\\PycharmProjects\\ErgoCub-Visual-Perception\\grasping\\shape_completion\\confidence_pcr")

import torch
from src.configs import server_config
from src.model import PCRNetwork


# # Load model
weights = torch.load('src_aux/checkpoint/completion_ckpt.pt',)
model = PCRNetwork(config=server_config.ModelConfig)
model.load_state_dict(weights['state_dict'])
model.cuda()
model.backbone.cuda()
model.eval()
model.backbone.eval()

x = torch.ones((1, 2024, 3)).cuda()
torch.onnx.export(model.backbone, x, 'test_pcr.onnx', input_names=['input'], output_names=[f'param{i}' for i in range(12)] + ['features'], opset_version=11)