import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('tuning_selection', Path(__file__).with_name('tuning_selection.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
REF = dict(PSNR=32., SSIM=.96, SIA_MAE=.3, Axis_ge_0p1D_MAE_deg=31., MAE=.016, RMSE=.024, SIA_RMSE=.4, Vector_RMSE=.7)
LIMIT = {**REF, 'MAE':.01616, 'RMSE':.02424, 'SIA_RMSE':.42, 'Vector_RMSE':.735}

def trial(number, **changes):
    metrics = {k:dict(mean=v) for k,v in {**REF, **changes}.items()}
    return dict(trial=number, audit='PASS', candidate=metrics, **module.screen(metrics, REF, LIMIT))

def test_improvement_does_not_override_image_guard():
    t = trial(1, SIA_MAE=.29, SSIM=.959)
    assert not t['eligible_for_group_selection']

def test_image_improvement_can_qualify_without_sia_gain():
    t = trial(1, PSNR=32.06)
    assert t['eligible_for_group_selection'] and not t['selected_for_frozen_test']

def test_reference_alone_is_not_a_material_gain():
    assert not trial(1)['eligible_for_group_selection']

def test_group_cannot_select_early():
    with pytest.raises(ValueError): module.select_group([trial(1, PSNR=32.06)])
    with pytest.raises(ValueError): module.select_group([trial(1, PSNR=32.06)], True)

def test_group_ties_use_preregistered_order():
    result = module.select_group([trial(3, PSNR=32.06), trial(1, PSNR=32.06), trial(2, PSNR=32.06)], True)
    assert result['ordered_trials'] == [1,2,3]
    assert result['ranks'][1] == [2.,2.,2.,2.]

def test_nonfinite_metrics_rejected():
    with pytest.raises(ValueError): trial(1, PSNR=float('nan'))
