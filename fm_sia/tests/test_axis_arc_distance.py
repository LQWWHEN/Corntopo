import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'cddpm'))
from model.polar_sia import periodic_axis_loss


def test_shortest_arc_matches_physical_axis_mae_and_gradient():
    # Physical axes 179 vs 1 and 10 vs 80 cross different sides of the circle.
    phase = torch.deg2rad(torch.tensor([358., 20., 40.], dtype=torch.float64)).requires_grad_()
    theta = torch.deg2rad(torch.tensor([2., 160., 0.], dtype=phase.dtype))
    target = torch.stack([theta.cos(), theta.sin()], dim=1)
    target[-1] *= .099
    loss = periodic_axis_loss(phase, target, .1, arc_distance=True)
    assert loss.item() == pytest.approx((2 + 70) / 2 / 45)
    torch.testing.assert_close(loss, periodic_axis_loss(phase + 2*math.pi, target, .1, True))
    loss.backward()
    torch.testing.assert_close(phase.grad, phase.new_tensor([-1/math.pi, -1/math.pi, 0]))


def test_arc_stability_threshold_and_empty_batch_gradient():
    phase = torch.tensor([math.pi, 0.], dtype=torch.float64, requires_grad=True)
    target = phase.new_tensor([[.1, 0], [0, 0]])
    assert periodic_axis_loss(phase, target, .1, True).item() == pytest.approx(2)
    loss = periodic_axis_loss(phase, target * 0, .1, True)
    loss.backward()
    assert loss.item() == 0
    torch.testing.assert_close(phase.grad, torch.zeros_like(phase))


def test_arc_configuration_is_opt_in():
    from train import get_args, saved_config_value
    args = ['--excel_path', 'unused', '--data_root', 'unused', '--image_backbone', 'fno',
            '--sia_prediction', 'polar_direct', '--lambda_sia', '1']
    assert not get_args(args).sia_axis_arc_distance
    assert not saved_config_value({}, 'sia_axis_arc_distance')
    assert get_args(args + ['--sia_axis_arc_distance']).sia_axis_arc_distance
    with pytest.raises(ValueError):
        get_args(args + ['--sia_axis_arc_distance', '--sia_learned_axis_concentration'])


def test_arc_model_changes_only_axis_supervision_and_reloads(tmp_path):
    pytest.importorskip('neuralop')
    from predict import load_checkpoint
    from train import build_model, get_args, PIPELINE_VERSION
    flags = ['--excel_path', 'unused', '--data_root', 'unused', '--image_backbone', 'fno',
             '--sia_prediction', 'polar_direct', '--lambda_sia', '1', '--fno_modes', '4', '4',
             '--fno_hidden_channels', '8', '--fno_layers', '1']
    stats = dict(sia=dict(mean=[0., 0.], std=[1., 1.], polar_median_magnitude=.4,
                          polar_initial_phase=.2), image=dict(scale=20.))
    torch.set_num_threads(1)
    torch.manual_seed(26)
    parent = build_model(get_args(flags), stats)
    torch.manual_seed(26)
    args = get_args(flags + ['--sia_axis_arc_distance'])
    child = build_model(args, stats)
    torch.testing.assert_close(parent.state_dict(), child.state_dict(), rtol=0, atol=0)
    pre = torch.rand(2, 1, 32, 32)
    post = torch.rand_like(pre)
    gt = torch.tensor([[.4, .2], [-.1, .6]])
    kw = dict(t=torch.tensor([.3, .7]))
    a, b = parent(pre, post, gt, **kw), child(pre, post, gt, **kw)
    for key in ['velocity_pred', 'q_clean', 'loss_image', 'loss_sia']:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    assert not torch.isclose(a['loss_axis'], b['loss_axis'])
    b['loss'].backward()
    assert child.sia_readout[-1].bias.grad[1].isfinite()
    path = tmp_path / 'round_trip.pth'
    try:
        torch.save(dict(pipeline_version=PIPELINE_VERSION, config=vars(args), stats=stats,
                        model=child.state_dict(), ema=None), path)
        restored, _, _, _ = load_checkpoint(path, 'cpu', cpu_threads=1)
        torch.testing.assert_close(restored(pre, post, gt, **kw)['loss_axis'], b['loss_axis'])
    finally:
        path.unlink(missing_ok=True)
