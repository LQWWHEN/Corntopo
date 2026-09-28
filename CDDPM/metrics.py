from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
import numpy as np
import torch

@torch.no_grad()
def calculate_metrics(
        pred,
        target
):

    """
    pred,target:
        Tensor [B,1,H,W]
        range [0,1]
    """

    pred = pred.detach().cpu()

    target = target.detach().cpu()



    mae = torch.mean(
        torch.abs(
            pred-target
        )
    ).item()



    rmse = torch.sqrt(
        torch.mean(
            (pred-target)**2
        )
    ).item()



    psnr = peak_signal_noise_ratio(
        target.numpy(),
        pred.numpy(),
        data_range=1.0
    )


    # SSIM requires 2D image

    pred_np = pred[0,0].numpy()

    target_np = target[0,0].numpy()


    ssim = structural_similarity(

        target_np,

        pred_np,

        data_range=1.0

    )

    return {
        "MAE":mae,
        "RMSE":rmse,
        "PSNR":psnr,
        "SSIM":ssim

    }