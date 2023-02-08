import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam

from .src_aux.configs import TrainConfig, ModelConfig
from .src_aux.model.ImplicitFunction import ImplicitFunction


class ConfidencePCRDecoder:
    def __init__(self, thr, steps, no_points):
        self.thr = thr
        self.steps = steps
        self.no_points = no_points

        self.sdf = ImplicitFunction(ModelConfig)

    def __call__(self, fast_weights):

        refined_pred = torch.tensor(torch.randn(1, self.no_points, 3).cpu().detach().numpy() * 1, device=TrainConfig.device,
                                    requires_grad=True)

        loss_function = BCEWithLogitsLoss(reduction='mean')
        optim = Adam([refined_pred], lr=0.1)

        c1, c2, c3, c4 = 1, 0, 0, 0 #1, 0, 0  1, 1e3, 0 # 0, 1e4, 5e2
        new_points = [] # refined_pred.detach().clone()
        for step in range(self.steps):
            results = self.sdf(refined_pred, fast_weights)
            new_points += [refined_pred.detach().clone()[:, (torch.sigmoid(results).squeeze() >= self.thr) * (torch.sigmoid(results).squeeze() <= 1), :]]

            gt = torch.ones_like(results[..., 0], dtype=torch.float32)
            gt[:, :] = 1
            loss1 = c1 * loss_function(results[..., 0], gt)

            loss_value = loss1

            self.sdf.zero_grad()
            optim.zero_grad()
            loss_value.backward(inputs=[refined_pred])
            optim.step()

        ##################################################
        ################# Visualization ##################
        ##################################################
        selected = torch.cat(new_points, dim=1).cpu().squeeze().numpy()
        return selected

