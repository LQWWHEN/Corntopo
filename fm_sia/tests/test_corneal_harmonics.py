import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'cddpm'))
from model.corneal_harmonics import annular_harmonic_loss
from model.fm_common import masked_mse
from train import get_args, build_model, saved_config_value, PIPELINE_VERSION
from predict import load_checkpoint
from clinical_features import EYE_SIGN

torch.set_num_threads(1)


@pytest.mark.parametrize('size', [32, 33, 128])
def test_known_harmonics_and_phase_error(size):
    coordinate = 2*(torch.arange(size, dtype=torch.float64)+.5)/size-1
    y, x = torch.meshgrid(coordinate, coordinate, indexing='ij')
    radius = torch.hypot(x, y)
    wave = (2*torch.atan2(y, x)).cos()*(radius > 0)
    prediction = wave[None, None]
    zero = torch.zeros_like(prediction)
    constant = annular_harmonic_loss(zero+1, zero)
    torch.testing.assert_close(constant, constant.new_tensor(1.), rtol=3e-6, atol=0)
    energy = annular_harmonic_loss(prediction, zero)
    torch.testing.assert_close(energy, wave[radius <= 1].square().mean(), rtol=3e-6, atol=2e-6)
    # Equal spectral magnitudes but opposite phase are different power maps.
    torch.testing.assert_close(annular_harmonic_loss(prediction, -prediction), 4*energy)
    torch.testing.assert_close(annular_harmonic_loss(prediction, prediction), zero.sum())


def test_masked_projection_d4_invariance_and_empty_region_gradients():
    torch.manual_seed(41)
    prediction = torch.randn(2, 1, 33, 33, dtype=torch.float64, requires_grad=True)
    target = torch.randn_like(prediction)
    mask = (torch.rand_like(prediction) > .3).to(prediction)
    reference = annular_harmonic_loss(prediction, target, mask)
    for turns in range(4):
        for flip in [False, True]:
            def transform(value):
                value = torch.rot90(value, turns, (-2, -1))
                return value.flip(-1) if flip else value
            torch.testing.assert_close(reference, annular_harmonic_loss(
                transform(prediction), transform(target), transform(mask)), rtol=2e-9, atol=2e-10)
    contaminated = prediction.detach().clone()
    contaminated[mask == 0] = 99999
    torch.testing.assert_close(reference, annular_harmonic_loss(contaminated, target, mask))
    reference.backward()
    assert torch.isfinite(prediction.grad).all()
    assert torch.count_nonzero(prediction.grad[mask == 0]) == 0
    empty = prediction.detach().requires_grad_(True)
    value = annular_harmonic_loss(empty, target, torch.zeros_like(mask))
    value.backward()
    assert value.item() == 0 and empty.grad.abs().sum() == 0
    sparse = torch.zeros_like(mask); sparse[:, :, 10, 12] = 1
    value = annular_harmonic_loss(empty, target, sparse)
    assert torch.isfinite(value)


def arguments(weight, extra=()):
    return get_args(['--excel_path', 'unused', '--data_root', 'unused',
        '--sia_prediction', 'axial_direct', '--eye_frame', 'right',
        '--conditions', 'pre_age_years', EYE_SIGN, '--fusion_method', 'film',
        '--lambda_sia', '1', '--sample_steps', '2', '--image_backbone', 'fno',
        '--fno_modes', '4', '4', '--fno_hidden_channels', '8', '--fno_layers', '1',
        '--image_harmonic_weight', str(weight), *extra])


def test_harmonic_objective_preserves_parent_capacity_inference_and_saved_reload(tmp_path):
    pytest.importorskip('neuralop')
    torch.set_num_threads(1)
    stats = dict(sia=dict(mean=[.2, .4], std=[.5, .6], canonical_mean=[.3, -.1]),
                 clinical=dict(mean=[50., .1], std=[10., .9]))
    torch.manual_seed(7); parent = build_model(arguments(0), stats).eval()
    torch.manual_seed(7); args = arguments(1); model = build_model(args, stats).eval()
    assert sum(p.numel() for p in parent.parameters()) == sum(p.numel() for p in model.parameters())
    for key, value in parent.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
    pre = torch.rand(2, 1, 32, 32); post = torch.rand_like(pre)
    clinical = torch.tensor([[.2, 1.], [-.1, -1.1/.9]])
    q = torch.tensor([[.4, -.2], [.3, .8]])
    mask = (torch.rand_like(pre) > .2).float(); t = torch.tensor([.2, .7])
    old = parent(pre, post, q, clinical=clinical, mask=mask, t=t)
    new = model(pre, post, q, clinical=clinical, mask=mask, t=t)
    assert 'loss_image_harmonic' not in old
    for key in ['velocity_pred', 'q_clean', 'loss_vector', 'loss_sia', 'loss_image_normalized']:
        torch.testing.assert_close(old[key], new[key], rtol=0, atol=0)
    expected = annular_harmonic_loss(new['velocity_pred'], post-pre, mask)
    torch.testing.assert_close(new['loss_image_harmonic'], expected)
    torch.testing.assert_close(new['loss_image'], masked_mse(new['velocity_pred'], post-pre, mask)+expected)
    torch.testing.assert_close(new['loss'], old['loss']+expected)
    new['loss_image_harmonic'].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.unet.parameters())
    before = model.sample(pre, clinical=clinical)
    old_prediction = parent.sample(pre, clinical=clinical)
    for key in ['pred', 'sia', 'sia_vector']:
        torch.testing.assert_close(before[key], old_prediction[key], rtol=0, atol=0)
    path = tmp_path/'temporary.pth'
    torch.save(dict(pipeline_version=PIPELINE_VERSION, config=vars(args), stats=stats,
                    model=model.state_dict(), ema=None), path)
    restored, _, _, _ = load_checkpoint(path, 'cpu', cpu_threads=1)
    after = restored.sample(pre, clinical=clinical)
    for key in ['pred', 'sia', 'sia_vector']:
        torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
    path.unlink()
    assert saved_config_value({}, 'image_harmonic_weight') == 0.


@pytest.mark.parametrize('weight,extra', [(-1, []), (float('nan'), []),
    (1, ['--loss_balance', 'gradnorm']), (1, ['--image_loss_units', 'diopter']),
    (1, ['--sia_prediction', 'polar_direct', '--eye_frame', 'native'])])
def test_harmonic_scope_and_finite_weight(weight, extra):
    with pytest.raises(ValueError):
        arguments(weight, extra)
