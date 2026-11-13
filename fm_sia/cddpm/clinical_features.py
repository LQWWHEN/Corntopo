"""Deterministic preoperative astigmatism and eye representations without outcomes."""
import numpy as np
import pandas as pd

PRE_ASTIG = 'pre_GENERAL DATA_Cornea Front Astig'
PRE_AXIS = 'pre_GENERAL DATA_Cornea Front Axis'
PRE_VECTOR = ('pre_front_X', 'pre_front_Y')
PRE_EYE = 'pre_PATIENT_Eye'
EYE_SIGN = 'pre_eye_sign'


def astigmatism_vector(magnitude, axis_degrees):
    """Full-cylinder double-angle components, using the dataset's flat-axis convention."""
    magnitude, axis = np.broadcast_arrays(np.asarray(magnitude, dtype=np.float64),
                                          np.asarray(axis_degrees, dtype=np.float64))
    if not np.isfinite(magnitude).all() or not np.isfinite(axis).all() or (magnitude < 0).any():
        raise ValueError('Preoperative magnitude and axis must be finite; magnitude must be nonnegative')
    theta = np.deg2rad(2*(axis % 180))
    return np.stack((magnitude*np.cos(theta), magnitude*np.sin(theta)), axis=-1)


def prepare_clinical_features(frame, conditions):
    """Rebuild selected vector/eye columns from raw PRE measurements in a copy.

    Never trust outcome-derived or stale cached vector columns. The input file
    and its hash remain unchanged; no fitting or cohort statistics occur here.
    """
    selected = set(conditions)
    if not selected.intersection((*PRE_VECTOR, EYE_SIGN)):
        return frame
    result = frame.copy()
    if selected.intersection(PRE_VECTOR):
        magnitude = pd.to_numeric(frame[PRE_ASTIG], errors='raise').to_numpy()
        axis = pd.to_numeric(frame[PRE_AXIS], errors='raise').to_numpy()
        q = astigmatism_vector(magnitude, axis)
        result[PRE_VECTOR[0]], result[PRE_VECTOR[1]] = q[:, 0], q[:, 1]
    if EYE_SIGN in selected:
        signs = frame[PRE_EYE].astype(str).str.strip().str.upper().map({'OD': 1., 'OS': -1.})
        if signs.isna().any():
            raise ValueError('Preoperative eye must be OD or OS for laterality conditioning')
        result[EYE_SIGN] = signs
    return result
