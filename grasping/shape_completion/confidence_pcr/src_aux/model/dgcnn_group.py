import torch
from torch import nn

from ..utils.misc import fp_sampling, onnx_cdists


class DGCNN_Grouper(nn.Module):
    def __init__(self):
        super().__init__()
        '''
        K has to be 16
        '''
        self.input_trans = nn.Conv1d(3, 8, 1)

        self.layer1 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=1, bias=False),
                                    nn.GroupNorm(4, 32),
                                    nn.LeakyReLU(negative_slope=0.2)
                                    )

        self.layer2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                    nn.GroupNorm(4, 64),
                                    nn.LeakyReLU(negative_slope=0.2)
                                    )

        self.layer3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, bias=False),
                                    nn.GroupNorm(4, 64),
                                    nn.LeakyReLU(negative_slope=0.2)
                                    )

        self.layer4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=False),
                                    nn.GroupNorm(4, 128),
                                    nn.LeakyReLU(negative_slope=0.2)
                                    )

    @staticmethod
    def fps_downsample(coor, x, num_group: int):
        xyz = coor.transpose(1, 2).contiguous()  # b, n, 3
        fps_idx = fp_sampling(xyz, num_group)

        combined_x = torch.cat([coor, x], dim=1)

        new_combined_x = combined_x.transpose(1, 2)[torch.arange(fps_idx.shape[0]).unsqueeze(-1), fps_idx.long(), :].transpose(1, 2)

        new_coor = new_combined_x[:, :3]
        new_x = new_combined_x[:, 3:]

        return new_coor, new_x

    @staticmethod
    def get_graph_feature(coor_q, x_q, coor_k, x_k):
        # B 3 2048, B 8 2048

        # coor: bs, 3, np, x: bs, c, np

        k = 16
        batch_size = x_k.size(0)
        num_points_k = x_k.size(2)
        num_points_q = x_q.size(2)

        with torch.no_grad():
            # TODO use onnx_cdists just to export to onnx, otherwise use torch.cdist
            dist = onnx_cdists(coor_k.transpose(1, 2), coor_q.transpose(1, 2))
            _, idx = torch.topk(-dist, dim=1, k=16)

            # assert idx.shape[1] == k
            idx_base = torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1) * num_points_k
            idx = idx + idx_base
            idx = idx.view(-1)

        num_dims = x_k.size(1)
        x_k = x_k.transpose(2, 1).contiguous()
        feature = x_k.view(batch_size * num_points_k, -1)[idx, :]
        feature = feature.view(batch_size, k, num_points_q, num_dims).permute(0, 3, 2, 1).contiguous()
        x_q = x_q.view(batch_size, num_dims, num_points_q, 1).expand(-1, -1, -1, k)
        feature = torch.cat((feature - x_q, x_q), dim=1)
        return feature

    def forward(self, x):  # x: B 3 2048
        # x: bs, 3, np

        # bs 3 N(128)   bs C(224)128 N(128)
        coor = x
        f1 = self.input_trans(x)  # out: B 8 2048

        f = self.get_graph_feature(coor, f1, coor, f1)
        f = self.layer1(f)
        f2 = f.max(dim=-1, keepdim=False)[0]

        coor_q, f_q = self.fps_downsample(coor, f2, 512)
        f = self.get_graph_feature(coor_q, f_q, coor, f2)
        f = self.layer2(f)
        f3 = f.max(dim=-1, keepdim=False)[0]
        coor = coor_q

        f = self.get_graph_feature(coor, f3, coor, f3)
        f = self.layer3(f)
        f4 = f.max(dim=-1, keepdim=False)[0]

        coor_q, f_q = self.fps_downsample(coor, f4, 128)
        f = self.get_graph_feature(coor_q, f_q, coor, f4)
        f = self.layer4(f)
        f5 = f.max(dim=-1, keepdim=False)[0]
        coor = coor_q

        return coor, f5
