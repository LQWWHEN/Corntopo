import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'cddpm'))
pytest.importorskip('neuralop')
from model.axial_sia import axial_moments
from model.radial_sia import radial_contrast_moments
from train import get_args, build_model, saved_config_value, PIPELINE_VERSION
from predict import load_checkpoint


def arguments(enabled=True, extra=()):
    return get_args(['--excel_path', 'unused', '--data_root', 'unused',
        '--sia_prediction', 'polar_direct', '--conditions', 'pre_age_years',
        '--fusion_method', 'film', '--lambda_sia', '1', '--sample_steps', '2',
        '--image_backbone', 'fno', '--fno_modes', '4', '4',
        '--fno_hidden_channels', '8', '--fno_layers', '1',
        *(['--sia_radial_contrast'] if enabled else []), *extra])


def statistics():
    return dict(sia=dict(mean=[.2, .3], std=[.5, .6], polar_median_magnitude=.5,
                        polar_initial_phase=.4, polar_axis_training_count=2), image=dict(scale=20.))


def test_radial_contrast_distinguishes_equal_global_means_and_transforms():
    torch.set_num_threads(1)
    v = torch.linspace(-31/32, 31/32, 32, dtype=torch.float64)
    yy, xx = torch.meshgrid(-v, v, indexing='ij')
    r2 = xx.square() + yy.square()
    inner, outer = r2 <= .5, (r2 > .5) & (r2 <= 1)
    feature = (inner.double() - outer.double()*inner.sum()/outer.sum())[None, None]
    global_m = axial_moments(feature)
    torch.testing.assert_close(global_m, torch.zeros_like(global_m), rtol=0, atol=1e-12)
    assert radial_contrast_moments(feature)[0, 0] > 1
    torch.testing.assert_close(radial_contrast_moments(torch.ones_like(feature)),
                               torch.zeros_like(global_m), rtol=0, atol=1e-12)
    feature = torch.randn(2, 3, 32, 32, dtype=torch.float64, requires_grad=True)
    actual = radial_contrast_moments(feature).reshape(2, 3, 3)
    flipped = radial_contrast_moments(feature.flip(-1)).reshape(2, 3, 3)
    torch.testing.assert_close(flipped, actual*actual.new_tensor([1, 1, -1])[None, :, None])
    actual.square().sum().backward()
    assert torch.isfinite(feature.grad).all() and feature.grad.abs().sum() > 0
    assert torch.count_nonzero(feature.grad[:, :, r2 > 1]) == 0


def test_parent_initialization_rng_and_original_outputs_are_preserved():
    torch.set_num_threads(1)
    torch.manual_seed(91); parent = build_model(arguments(False), statistics()).eval()
    expected_rng = torch.get_rng_state()
    torch.manual_seed(91); child = build_model(arguments(), statistics()).eval()
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for name, value in parent.state_dict().items():
        torch.testing.assert_close(child.state_dict()[name], value, rtol=0, atol=0)
    assert sum(p.numel() for p in child.parameters())-sum(p.numel() for p in parent.parameters()) == 196608
    pre, clinical = torch.rand(2, 1, 32, 32), torch.zeros(2, 1)
    a, b = parent.sample(pre, clinical=clinical), child.sample(pre, clinical=clinical)
    for name in ['pred', 'sia', 'sia_vector']:
        torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)
    assert saved_config_value({}, 'sia_radial_contrast') is False


def test_radial_readout_gradients_and_preoperative_only_contract():
    torch.set_num_threads(1); model = build_model(arguments(), statistics()).eval()
    with torch.no_grad():
        model.sia_readout[-1].weight.normal_(0, .01)
        model.radial_projection.weight.normal_(0, .01)
    pre = torch.rand(2, 1, 32, 32, requires_grad=True)
    clinical = torch.rand(2, 1, requires_grad=True)
    post = torch.rand_like(pre); gt = torch.tensor([[.6, -.3], [-.3, .4]])
    a = model(pre, post, gt, clinical=clinical, t=torch.tensor([.2, .7]))
    b = model(pre, post+10, gt*3, clinical=clinical, t=torch.tensor([.8, .1]))
    torch.testing.assert_close(a['q_clean'], b['q_clean'], rtol=0, atol=0)
    c = model(pre, post, gt*3, clinical=clinical, t=torch.tensor([.2, .7]))
    torch.testing.assert_close(a['velocity_pred'], c['velocity_pred'], rtol=0, atol=0)
    torch.testing.assert_close(a['loss'], a['loss_image']+a['loss_sia']+a['loss_axis'])
    grads = torch.autograd.grad(a['loss_sia']+a['loss_axis'],
        (model.radial_projection.weight, pre, clinical))
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)
    torch.testing.assert_close(a['q_clean'].norm(dim=-1), a['sia_clean'])


@pytest.mark.parametrize('enabled', [False, True])
def test_saved_weight_roundtrip_and_missing_legacy_option(tmp_path, enabled):
    torch.set_num_threads(1); args = arguments(enabled); model = build_model(args, statistics()).eval()
    with torch.no_grad():
        model.sia_readout[-1].weight.normal_(0, .01)
        if enabled: model.radial_projection.weight.normal_(0, .01)
    config = vars(args).copy()
    if not enabled: config.pop('sia_radial_contrast')
    pre, clinical = torch.rand(1, 1, 32, 32), torch.zeros(1, 1)
    before = model.sample(pre, clinical=clinical)
    path = tmp_path/'temporary.pth'
    try:
        torch.save(dict(pipeline_version=PIPELINE_VERSION, config=config, stats=statistics(),
                        model=model.state_dict(), ema=None), path)
        restored, _, _, _ = load_checkpoint(path, 'cpu', cpu_threads=1)
        after = restored.sample(pre, clinical=clinical)
        for name in ['pred', 'sia', 'sia_vector']:
            torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)
    finally:
        path.unlink(missing_ok=True)


@pytest.mark.parametrize('extra', [
    ['--sia_prediction', 'axial_direct'], ['--image_backbone', 'unet'],
    ['--sia_spatial_conditioning'], ['--sia_velocity_coupling'], ['--image_local_residual'],
    ['--image_gradient_weight', '.25'], ['--sia_rank_weight', '.1'], ['--augmentation', 'd4'],
])
def test_scoped_configuration(extra):
    with pytest.raises(ValueError):
        arguments(extra=extra)
