import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget


class MultiInputCamWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.mode = model.mode

    def forward(self, x):
        if self.mode == 'post_only':
            out = self.model(x)
        elif self.mode == 'pre_post':
            pre = x[:, :3, :, :]
            post = x[:, 3:6, :, :]
            out = self.model(pre, post)
        elif self.mode == 'post_premask':
            post = x[:, :3, :, :]
            mask = x[:, 3:4, :, :]
            out = self.model(post, mask)
        elif self.mode == 'pre_post_premask':
            pre = x[:, :3, :, :]
            post = x[:, 3:6, :, :]
            mask = x[:, 6:7, :, :]
            out = self.model(pre, post, mask)
        else:
            out = self.model(x)

        if isinstance(out, tuple):
            return out[0]

        return out


