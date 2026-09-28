import torch
import torch.nn as nn
import torch.nn.functional as F

def laplacian_loss(img, mask=None):
    """计算二阶梯度损失"""
    laplace_kernel = torch.tensor([[0, 1, 0],
                                   [1, -4, 1],
                                   [0, 1, 0]], dtype=torch.float32, device=img.device).unsqueeze(0).unsqueeze(0)
    lap = F.conv2d(img, laplace_kernel, padding=1)
    if mask is not None:
        lap = lap * mask
    return torch.mean(torch.abs(lap))



class GradientLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target, mask=None):

        dx_pred = pred[..., :, 1:] - pred[..., :, :-1]
        dy_pred = pred[..., 1:, :] - pred[..., :-1, :]

        dx_target = target[..., :, 1:] - target[..., :, :-1]
        dy_target = target[..., 1:, :] - target[..., :-1, :]

        diff_x = torch.abs(dx_pred - dx_target)
        diff_y = torch.abs(dy_pred - dy_target)

        if mask is not None:
            mask_x = mask[..., :, 1:] * mask[..., :, :-1]
            mask_y = mask[..., 1:, :] * mask[..., :-1, :]

            diff_x = diff_x * mask_x
            diff_y = diff_y * mask_y

        if self.reduction == "mean":
            return diff_x.mean() + diff_y.mean()

        elif self.reduction == "sum":
            return diff_x.sum() + diff_y.sum()

        return diff_x + diff_y