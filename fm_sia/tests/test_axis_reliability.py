import math
import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cddpm"))
from model.polar_sia import periodic_axis_loss


def test_reliability_normalization_cutoff_and_cap():
    magnitude = torch.tensor([0., .099, .1, .2, .3, 1.], dtype=torch.float64)
    target = torch.stack((magnitude, torch.zeros_like(magnitude)), dim=-1)
    phase = torch.tensor([.2, .3, .4, .5, .6, .7], dtype=magnitude.dtype, requires_grad=True)
    weights = phase.new_tensor([0., 0., 1., 4., 9., 9.]) / 23
    actual = periodic_axis_loss(phase, target, .1, reliability_scale=.3)
    torch.testing.assert_close(actual, (weights*(1-phase.cos())).sum())
    torch.testing.assert_close(torch.autograd.grad(actual, phase)[0], weights * phase.sin())
    torch.testing.assert_close(actual, periodic_axis_loss(phase+2*math.pi, target, .1, reliability_scale=.3))


def test_equal_weight_and_empty_stable_set():
    phase = torch.tensor([.8, -.4], requires_grad=True)
    target = torch.tensor([[1., 0.], [0., -.6]])
    torch.testing.assert_close(periodic_axis_loss(phase, target, .1, reliability_scale=.3),
                               periodic_axis_loss(phase, target, .1))
    loss = periodic_axis_loss(phase, target*0, .1, reliability_scale=.3)
    assert loss.item() == 0.
    torch.testing.assert_close(torch.autograd.grad(loss, phase)[0], torch.zeros_like(phase))


def test_configuration_is_opt_in():
    from train import get_args, saved_config_value
    flags = ['--excel_path','unused','--data_root','unused','--image_backbone','fno',
             '--sia_prediction','polar_direct','--lambda_sia','1']
    assert get_args(flags).axis_reliability_scale == saved_config_value({},'axis_reliability_scale') == 0.
    assert get_args(flags + ['--axis_reliability_scale','.3']).axis_reliability_scale == .3
    with pytest.raises(ValueError):
        get_args(flags + ['--axis_reliability_scale','.1'])
