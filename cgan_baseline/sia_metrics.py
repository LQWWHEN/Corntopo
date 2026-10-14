"""Scalar SIA support around the unchanged FM image/vector metric implementation."""
import numpy as np
import torch
from metrics import MetricAccumulator


class SIAMetricAccumulator(MetricAccumulator):
    def __init__(self, image_scale=1.0):
        super().__init__(image_scale)
        self.scalar_errors = []
        self.sia_reporting_mode = None

    @torch.no_grad()
    def update(self, pred, target, mask=None, q_pred=None, q_true=None, s_pred=None, s_true=None):
        mode = 'vector' if q_pred is not None else 'scalar' if s_pred is not None else 'off'
        if self.sia_reporting_mode is not None and self.sia_reporting_mode != mode:
            raise ValueError('Cannot mix scalar/vector/off predictions in one evaluation')
        self.sia_reporting_mode = mode
        super().update(pred, target, mask, q_pred, q_true, s_pred)
        if mode == 'scalar':
            sp = s_pred.detach().cpu().numpy().astype(np.float64)
            if q_true is not None and q_true.shape == (len(pred), 2):
                # Exactly the FM vector-mode physical ground-truth magnitude.
                st = np.linalg.norm(q_true.detach().cpu().numpy().astype(np.float64), axis=1)
            elif s_true is not None:
                st = s_true.detach().cpu().numpy().astype(np.float64)
            else:
                raise ValueError('Scalar SIA evaluation requires magnitude ground truth')
            if sp.shape != (len(pred),) or st.shape != sp.shape or not np.isfinite(sp).all() or not np.isfinite(st).all() or (st < 0).any():
                raise ValueError('Invalid scalar SIA prediction/target')
            self.scalar_errors.append(sp-st)

    def compute(self):
        result = super().compute()
        if self.scalar_errors:
            error = np.concatenate(self.scalar_errors)
            result.update(SIA_MAE=float(np.abs(error).mean()),
                          SIA_RMSE=float(np.sqrt(np.square(error).mean())),
                          SIA_Bias=float(error.mean()))
        return result
