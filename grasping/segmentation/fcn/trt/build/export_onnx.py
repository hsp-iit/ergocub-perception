import torch
from torch import nn
from torchvision import models
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode


class Model(nn.Module):
    """ This model encapsulate data preprocessing along with the model so that it gets exported in the onnx
        The input image is expected to be an RGB (not BGR) numpy array of integer between 0-255 in the
         (h, w, c) format. The size is irrelevant since resizing is performed but, to avoid, distortions,
          the format has to be 4:3 (e.g. 640x480); it will be resized to 256x192."""

    def __init__(self):
        super().__init__()
        model = models.segmentation.fcn_resnet101(pretrained=False)
        model.classifier[4] = nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))  # make the classifier binary
        # strict false as the original fcn_resnet101 has an auxiliary head that we don't use
        model.load_state_dict(torch.load('grasping/segmentation/fcn/checkpoints/segmentation_ckpt.pt'), strict=False)
        model.eval()
        model.cuda()

        self.model = model

        self.tr = T.Compose([lambda img: img.unsqueeze(0).permute(0, 3, 1, 2),
                             lambda img: img / 255,
                             T.Resize((192, 256), InterpolationMode.BILINEAR),
                             T.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])])

    def forward(self, img):
        img = self.tr(img)
        pred = self.model(img)['out']
        mask = torch.argmax(pred, dim=1).permute([1, 2, 0])
        return mask


def main():
    model = Model()
    x = torch.randn((480, 640, 3)).cuda()

    torch.onnx.export(model, x, 'grasping/segmentation/fcn/trt/assets/segmentation.onnx', input_names=['input'],
                      output_names=[f'output'], opset_version=11)


if __name__ == '__main__':
    main()
