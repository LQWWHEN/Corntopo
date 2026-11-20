import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cddpm"))
pytest.importorskip("neuralop")
from predict import load_checkpoint
from train import PIPELINE_VERSION, build_model, forward_batch, get_args, sample_batch


def arguments(backbone="uno", layers=4):
    return get_args([
        "--excel_path", "unused", "--data_root", "unused", "--sia_prediction", "polar_direct",
        "--conditions", "pre_age_years", "--fusion_method", "film", "--lambda_sia", "1",
        "--sample_steps", "2", "--image_backbone", backbone, "--fno_modes", "6", "6",
        "--fno_hidden_channels", "8", "--fno_layers", str(layers),
    ])


def statistics():
    return {"sia": {"mean": [.2, .3], "std": [.5, .6], "polar_median_magnitude": .5,
                    "polar_initial_phase": .4, "polar_axis_training_count": 2}, "image": {"scale": 20.}}


def test_shared_initialization_and_preoperative_sia():
    torch.set_num_threads(1)
    torch.manual_seed(31)
    parent = build_model(arguments("fno"), statistics()).eval()
    parent_rng = torch.get_rng_state()
    torch.manual_seed(31)
    change = build_model(arguments(), statistics()).eval()
    assert torch.equal(parent_rng, torch.get_rng_state())
    for name, value in parent.state_dict().items():
        if not name.startswith("unet.fno."):
            torch.testing.assert_close(value, change.state_dict()[name], atol=0, rtol=0)
    pre = torch.rand(1, 1, 32, 32)
    post = torch.rand_like(pre)
    clinical = torch.ones(1, 1)
    q = torch.tensor([[.3, -.1]])
    with torch.no_grad():
        parent_out = parent(pre, post, q, clinical=clinical, t=torch.tensor([.2]))
        candidate_out = change(pre, post + 4, q + 2, clinical=clinical, t=torch.tensor([.7]))
    torch.testing.assert_close(candidate_out["q_clean"], parent_out["q_clean"], rtol=0, atol=0)


def test_multiresolution_skip_and_full_task_gradients():
    torch.set_num_threads(1)
    args = arguments()
    model = build_model(args, statistics())
    operator = model.unet.fno
    shapes = []
    hooks = [mod.register_forward_hook(lambda module, inputs, out: shapes.append(out.shape[-2:]))
             for mod in operator.fno_blocks]
    pre = torch.rand(2, 1, 32, 32)
    batch = dict(input=pre, target=torch.rand_like(pre), condition=torch.tensor([[.2], [.5]]),
                 mask_pre=torch.ones_like(pre), mask_post=torch.ones_like(pre),
                 sia_vector=torch.tensor([[.3, -.1], [.1, .5]]), label=[1, 2])
    try:
        out = forward_batch(model, batch, "cpu")
    finally:
        for hook in hooks:
            hook.remove()
    assert len(shapes) == 4
    assert shapes[0] == shapes[2] == shapes[3]
    assert all(a == 2 * b for a, b in zip(shapes[0], shapes[1]))
    torch.testing.assert_close(out["loss"], out["loss_image"] + out["loss_sia"] + out["loss_axis"])
    out["loss"].backward()
    for module in (operator.horizontal_skips["0"], operator.fno_blocks[1], model.condition_encoder):
        assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                   for p in module.parameters())


@pytest.mark.parametrize("backbone", ["uno", "half_turn_fno"])
def test_saved_run_and_preoperative_sampling(tmp_path, backbone):
    torch.set_num_threads(1)
    args = arguments(backbone)
    model = build_model(args, statistics()).eval()
    pre = torch.rand(1, 1, 32, 32)
    batch = dict(input=pre, condition=torch.tensor([[.1]]), mask_pre=torch.ones_like(pre), label=[1])
    cached = []
    hook = model.unet.register_forward_pre_hook(lambda module, inputs: cached.append(inputs[2]))
    before = sample_batch(model, batch, args, "cpu")
    hook.remove()
    assert len(cached) == args.sample_steps and cached[0] is cached[1]
    assert before["pred"].shape == pre.shape and before["sia_vector"].shape == (1, 2)
    path = tmp_path / "reload.pth"
    try:
        torch.save(dict(pipeline_version=PIPELINE_VERSION, config=vars(args), stats=statistics(),
                        model=model.state_dict(), ema=None), path)
        restored, _, _, _ = load_checkpoint(path, "cpu", cpu_threads=1)
        after = sample_batch(restored, batch, args, "cpu")
        for key in ("pred", "sia", "sia_vector"):
            assert torch.isfinite(after[key]).all()
            torch.testing.assert_close(before[key], after[key], atol=0, rtol=0)
    finally:
        path.unlink(missing_ok=True)


def test_four_scale_stages_required():
    with pytest.raises(ValueError, match="four operator layers"):
        arguments(layers=3)


def test_rectangular_grid_avoids_bicubic_backward(monkeypatch):
    from torch.nn import functional as F
    from model.u_velocity_operator import UVelocityOperator

    original = F.interpolate
    def supported(*args, **kwargs):
        assert kwargs.get("mode") != "bicubic"
        return original(*args, **kwargs)

    monkeypatch.setattr(F, "interpolate", supported)
    op = UVelocityOperator((4, 4), 8, .05)
    x = torch.randn(1, 65, 29, 33, requires_grad=True)
    y = op(x)
    assert y.shape == (1, 1, 29, 33)
    y.square().mean().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0

def test_half_turn_fno_equivariance_and_parent_parameter_gradients():
    torch.set_num_threads(1)
    torch.manual_seed(38)
    parent = build_model(arguments("fno"), statistics())
    torch.manual_seed(38)
    model = build_model(arguments("half_turn_fno"), statistics())
    assert model.state_dict().keys() == parent.state_dict().keys()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, parent.state_dict()[key], rtol=0, atol=0)
    axes = (-2, -1)
    inputs = torch.randn(2, 65, 19, 23, requires_grad=True)
    ref = inputs.detach().clone().requires_grad_()
    velocity = model.unet.velocity(inputs)
    expected = (parent.unet.fno(ref) + parent.unet.fno(ref.flip(axes)).flip(axes)) / 2
    torch.testing.assert_close(velocity, expected)
    torch.testing.assert_close(model.unet.velocity(inputs.flip(axes)), velocity.flip(axes))
    assert not torch.allclose(velocity, velocity.flip(axes))
    target = torch.randn_like(velocity)
    (velocity * target).mean().backward()
    (expected * target).mean().backward()
    torch.testing.assert_close(inputs.grad, ref.grad)
    for (_, actual), (_, reference) in zip(model.unet.fno.named_parameters(),
                                            parent.unet.fno.named_parameters()):
        if reference.grad is not None:
            torch.testing.assert_close(actual.grad, reference.grad)
    assert inputs.grad.abs().sum() > 0
