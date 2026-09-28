import numpy as np
import torch

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)

def calc_regression_metrics(pred, gt):
    """
    pred: (B, K)
    gt:   (B, K)
    """

    pred = pred.detach().cpu()
    gt = gt.detach().cpu()

    abs_err = torch.abs(pred - gt)          # (B, K)
    sq_err = (pred - gt) ** 2

    mae_all = abs_err.mean(dim=0).numpy()   # per-target MAE
    rmse_all = torch.sqrt(sq_err.mean(dim=0)).numpy()

    return {
        "REG_MAE_all": mae_all,
        "REG_RMSE_all": rmse_all
    }

def calc_metrics(pred, gt):
    """
    pred, gt:
        torch.Tensor
        shape = [B,C,H,W]
    """

    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()

    if isinstance(gt, torch.Tensor):
        gt = gt.detach().cpu().numpy()
    batch_size = pred.shape[0]

    psnr_list = []
    ssim_list = []
    mae_list = []
    rmse_list = []

    for i in range(batch_size):

        p = pred[i]
        g = gt[i]

        # 单通道
        if p.shape[0] == 1:
            p = p[0]
            g = g[0]

        psnr = peak_signal_noise_ratio(
            g,
            p,
            data_range=1.0
        )

        ssim = structural_similarity(
            g,
            p,
            data_range=1.0
        )

        mae = np.mean(np.abs(p - g))

        rmse = np.sqrt(
            np.mean((p - g) ** 2)
        )

        psnr_list.append(psnr)
        ssim_list.append(ssim)
        mae_list.append(mae)
        rmse_list.append(rmse)

    return {
        "PSNR": float(np.mean(psnr_list)),
        "SSIM": float(np.mean(ssim_list)),
        "MAE": float(np.mean(mae_list)),
        "RMSE": float(np.mean(rmse_list)),
        "PSNR_all": psnr_list,
        "SSIM_all": ssim_list,
        "MAE_all": mae_list,
        "RMSE_all": rmse_list,
    }