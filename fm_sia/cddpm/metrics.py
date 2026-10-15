"""Report full-frame and valid-region image metrics on the same fixed scale.

*_full includes every supplied pixel, including normalized zero-filled targets.
*_valid uses the supplied mask (the caller supplies common pre/post validity).
*_raw_* expresses MAE/RMSE in original units, not a different pixel domain.
Legacy unsuffixed names remain aliases of valid-region metrics.
"""
import numpy as np
import torch
from scipy.ndimage import binary_erosion
from skimage.metrics import structural_similarity


class MetricAccumulator:
    def __init__(self, image_scale=1.0):
        self.image_scale = image_scale
        self.absolute = self.squared = 0.0
        self.pixels = 0
        self.ssim_sum = 0.0
        self.ssim_count = 0
        self.full_absolute = self.full_squared = 0.0
        self.full_pixels = 0
        self.full_ssim_sum = 0.0
        self.full_ssim_count = 0
        self.samples = 0
        self.q_true, self.q_pred, self.s_pred, self.s_true = [], [], [], []

    @torch.no_grad()
    def update(self, pred, target, mask=None, q_pred=None, q_true=None, s_pred=None, s_true=None):
        p, y = pred.detach().cpu().numpy(), target.detach().cpu().numpy()
        m = np.ones_like(y, bool) if mask is None else mask.detach().cpu().numpy().astype(bool)
        if p.ndim != 4 or p.shape != y.shape or m.shape != y.shape:
            raise ValueError('Expected matching prediction, target and mask shapes [B,C,H,W]')
        if not np.isfinite(p).all() or not np.isfinite(y).all():
            raise ValueError('Non-finite image prediction/target')
        for i in range(len(p)):
            full_error = p[i].astype(np.float64)-y[i].astype(np.float64)
            error = p[i][m[i]].astype(np.float64)-y[i][m[i]].astype(np.float64)
            if not error.size:
                raise ValueError('Empty evaluation mask')
            self.absolute += np.abs(error).sum()
            self.squared += np.square(error).sum()
            self.pixels += error.size
            self.full_absolute += np.abs(full_error).sum()
            self.full_squared += np.square(full_error).sum()
            self.full_pixels += full_error.size
            self.samples += 1
            for channel in range(p.shape[1]):
                win = min(7, *p.shape[-2:])
                win -= 1-win % 2
                if win < 3:
                    continue
                # Standard full-image SSIM: all interior window centers, no ROI mask.
                full_ssim, smap = structural_similarity(
                    y[i, channel], p[i, channel], data_range=1.0, win_size=win, full=True)
                self.full_ssim_sum += float(full_ssim)
                self.full_ssim_count += 1
                centers = binary_erosion(m[i, channel], structure=np.ones((win, win)), border_value=0)
                if centers.any():
                    self.ssim_sum += float(smap[centers].mean())
                    self.ssim_count += 1
        if q_pred is not None:
            if q_true is None:
                raise ValueError('Vector predictions require vector targets')
            qp = q_pred.detach().cpu().numpy().astype(np.float64)
            qt = q_true.detach().cpu().numpy().astype(np.float64)
            if qp.shape != (len(p), 2) or qt.shape != qp.shape:
                raise ValueError('Expected matching SIA vectors [B,2]')
            sp = np.linalg.norm(qp, axis=1) if s_pred is None else s_pred.detach().cpu().numpy().astype(np.float64)
            st = np.linalg.norm(qt, axis=1)
            if not np.isfinite(qp).all() or not np.isfinite(qt).all():
                raise ValueError('Non-finite SIA prediction/target')
            self.q_pred.append(qp); self.q_true.append(qt)
        elif s_pred is not None or s_true is not None:
            if s_pred is None or s_true is None:
                raise ValueError('Scalar SIA metrics require both prediction and target')
            sp = s_pred.detach().cpu().numpy().astype(np.float64)
            st = s_true.detach().cpu().numpy().astype(np.float64)
        else:
            return
        if sp.shape != (len(p),) or st.shape != (len(p),) or not np.isfinite(sp).all() or not np.isfinite(st).all() or (sp < 0).any() or (st < 0).any():
            raise ValueError('SIA magnitudes must be finite nonnegative [B] values')
        self.s_pred.append(sp); self.s_true.append(st)

    def compute(self):
        if self.pixels == 0:
            raise ValueError('Empty evaluation dataset')
        mse = self.squared/self.pixels
        result = dict(MAE=float(self.absolute/self.pixels), RMSE=float(np.sqrt(mse)),
                      PSNR=float(-10*np.log10(max(mse, 1e-12))),
                      SSIM=float(self.ssim_sum/self.ssim_count) if self.ssim_count else None,
                      MAE_raw=float(self.absolute/self.pixels*self.image_scale),
                      RMSE_raw=float(np.sqrt(mse)*self.image_scale))
        # Preserve legacy keys/checkpoint selection while adding explicit domains.
        result.update({f'{key}_valid': value for key, value in list(result.items())})
        full_mse = self.full_squared/self.full_pixels
        result.update(
            MAE_full=float(self.full_absolute/self.full_pixels),
            RMSE_full=float(np.sqrt(full_mse)),
            PSNR_full=float(-10*np.log10(max(full_mse, 1e-12))),
            SSIM_full=float(self.full_ssim_sum/self.full_ssim_count) if self.full_ssim_count else None,
            MAE_raw_full=float(self.full_absolute/self.full_pixels*self.image_scale),
            RMSE_raw_full=float(np.sqrt(full_mse)*self.image_scale))
        if self.s_pred:
            ds = np.concatenate(self.s_pred)-np.concatenate(self.s_true)
            result.update(SIA_MAE=float(np.abs(ds).mean()), SIA_RMSE=float(np.sqrt(np.square(ds).mean())),
                          SIA_Bias=float(ds.mean()))
        if self.q_pred:
            err = np.concatenate(self.q_pred)-np.concatenate(self.q_true)
            result.update(Vector_MAE=float(np.linalg.norm(err, axis=1).mean()),
                          Vector_RMSE=float(np.sqrt(np.square(err).sum(1).mean())),
                          DeltaX_MAE=float(np.abs(err[:, 0]).mean()), DeltaY_MAE=float(np.abs(err[:, 1]).mean()))
        return result


@torch.no_grad()
def calculate_metrics(pred, target, mask=None, image_scale=1.0):
    accumulator = MetricAccumulator(image_scale)
    accumulator.update(pred, target, mask)
    return accumulator.compute()

