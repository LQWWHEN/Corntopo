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

def test_initial_trial_competes_after_five_tuning_trials():
    candidates = [trial(n, PSNR=32.06) for n in range(1, 6)]
    result = module.select_group(candidates + [trial(0, PSNR=32.08)], True)
    assert result['selected_trial'] == 0
    assert result['ordered_trials'] == list(range(6))
    result = module.select_group(candidates + [trial(0)], True)
    assert result['selected_trial'] == 1
    assert 0 not in result['ranks']

@pytest.mark.parametrize('numbers', [[0, 1, 2], [0, 0, 1, 2, 3], [1, 2, 3, 4, 5, 6]])
def test_initial_trial_is_separate_from_tuning_budget(numbers):
    with pytest.raises(ValueError):
        module.select_group([trial(n) for n in numbers], True)


def test_post050_group_ranks_fallback_candidates_only_when_no_formal_candidate_exists():
    candidates = [trial(0, PSNR=31.9), trial(1, PSNR=31.8, SSIM=.95),
                  trial(2, PSNR=31.8, SSIM=.95, MAE=.02), trial(3, SSIM=.959)]
    result = module.select_post050_group(candidates, REF, LIMIT, group_closed=True)
    assert result['tiers'] == {0:'one_or_two_missed',1:'one_or_two_missed',2:None,3:'one_or_two_missed'}
    assert set(result['ranks']) == {0,1,3}
    candidates[2] = trial(2, PSNR=32.08)
    result = module.select_post050_group(candidates, REF, LIMIT, group_closed=True)
    assert result['selected_trial'] == 2 and result['ordered_trials'] == [2]
