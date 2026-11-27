"""A patient-specific source in physical, doubled-angle SIA coordinates."""
import numpy as np
import torch

from clinical_features import EYE_SIGN, PRE_VECTOR
from model.fm_sia import JointFlowMatching


def fit_reference(pre, target, sign):
    """Fit two eye intercepts and a shared, rotation-equivariant scalar slope."""
    if len(target) == 0:
        raise ValueError("A clinical reference needs training patients")
    group = (sign > 0).astype(int)
    mean_pre, mean_target = [], []
    for index in (0, 1):
        subset = group == index
        mean_pre.append((pre[subset] if subset.any() else pre).mean(axis=0))
        mean_target.append((target[subset] if subset.any() else target).mean(axis=0))
    mean_pre, mean_target = np.array(mean_pre), np.array(mean_target)
    x = pre - mean_pre[group]
    y = target - mean_target[group]
    variance = np.square(x).sum()
    slope = float((x*y).sum()/variance) if variance > 1e-12 else 0.0
    return dict(pre=mean_pre.tolist(), target=mean_target.tolist(), slope=slope)


def training_arrays(frame, columns):
    return (frame[list(PRE_VECTOR)].to_numpy(np.float64), frame[list(columns)].to_numpy(np.float64),
            frame[EYE_SIGN].to_numpy(np.float64))


def crossfit_references(frame, columns):
    """Every training eye uses coefficients fitted without any eye of that patient."""
    from patient_cv import _labels_and_groups
    labels, by_label, _ = _labels_and_groups(frame)
    patients = np.array([by_label[label] for label in labels])
    pre, target, sign = training_arrays(frame, columns)
    predictions = np.empty_like(target)
    for patient in np.unique(patients):
        held = patients == patient
        fit = fit_reference(pre[~held], target[~held], sign[~held])
        eye = (sign[held] > 0).astype(int)
        predictions[held] = (np.array(fit['target'])[eye] +
                             fit['slope']*(pre[held]-np.array(fit['pre'])[eye]))
    return dict(zip(labels, predictions.astype(np.float32)))


class PreopVectorSourceFlow(JointFlowMatching):
    def __init__(self, *args, clinical_stats, reference, **kwargs):
        super().__init__(*args, **kwargs)
        for name, values in [('reference_mean', clinical_stats['mean'][-3:]),
                             ('reference_std', clinical_stats['std'][-3:]),
                             ('pre_centers', reference['pre']), ('sia_centers', reference['target']),
                             ('slope', reference['slope'])]:
            self.register_buffer(name, torch.as_tensor(values, dtype=torch.float32))

    def conditions_and_center(self, clinical):
        raw = clinical[:, -3:] * self.reference_std + self.reference_mean
        eye = (raw[:, 0] > 0).long()
        center = self.sia_centers[eye] + self.slope*(raw[:, 1:] - self.pre_centers[eye])
        return clinical[:, :-3], center

    def initial_sia_state(self, noise, center=None):
        # Zero standardized displacement is the physical clinical reference.
        return torch.zeros_like(noise)
