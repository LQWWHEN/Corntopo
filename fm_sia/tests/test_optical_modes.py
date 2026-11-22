import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cddpm"))
from model.optical_modes import optical_basis, set_velocity_modes
from predict import load_checkpoint
from test_conditional_uno import arguments, statistics
from train import PIPELINE_VERSION, build_model, forward_batch, sample_batch


def test_projection_replaces_only_resolvable_modes_and_keeps_aperture_mean():
    valid = torch.ones(4, 1, 33, 35, dtype=torch.bool)
    valid[1, :, :15, :20] = False
    valid[2] = False
    valid[3] = False
    valid[3, 0, 16, 13:19] = True
    basis = reader, writer = optical_basis(valid)
    product = torch.einsum("bihw,bjhw->bij", reader, writer)
    torch.testing.assert_close(product[:2], torch.eye(2).expand(2, -1, -1), atol=2e-6, rtol=0)
    assert torch.count_nonzero(reader[2:]) == torch.count_nonzero(writer[2:]) == 0
    velocity = torch.randn(4, 1, 33, 35, requires_grad=True)
    modes = torch.tensor([[0.3, -0.7], [1.0, 0.5], [-0.2, 0.9], [0.7, -0.3]], requires_grad=True)
    out = set_velocity_modes(velocity, modes, basis)
    actual = (reader * out).sum((-2, -1))
    torch.testing.assert_close(actual[:2], modes[:2], atol=2e-6, rtol=0)
    torch.testing.assert_close(out[2:], velocity[2:], atol=0, rtol=0)
    torch.testing.assert_close(out[~valid], velocity[~valid], atol=0, rtol=0)
    yy, xx = torch.meshgrid(
        1 - 2 * (torch.arange(33) + 0.5) / 33, 2 * (torch.arange(35) + 0.5) / 35 - 1, indexing="ij"
    )
    r2 = xx.square() + yy.square()
    central = valid & (r2 > 0) & (r2 <= 0.25)
    shift = ((out - velocity) * central).sum((-2, -1)) / central.sum((-2, -1)).clamp_min(1)
    torch.testing.assert_close(shift, torch.zeros_like(shift), atol=2e-6, rtol=0)
    v_grad, c_grad = torch.autograd.grad(actual.sum(), (velocity, modes))
    assert v_grad.abs().max() < 1e-6
    torch.testing.assert_close(c_grad[:2], torch.ones(2, 2), atol=2e-6, rtol=0)


def test_joint_mode_gradient_physical_units_and_saved_sampling(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    parent = build_model(arguments("fno"), statistics()).eval()
    rng = torch.get_rng_state()
    args = arguments("optical_mode_fno")
    torch.manual_seed(42)
    model = build_model(args, statistics()).eval()
    assert torch.equal(torch.get_rng_state(), rng)
    for name, value in parent.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name], value, atol=0, rtol=0)
    pre = torch.rand(2, 1, 32, 32)
    clinical = torch.tensor([[0.2], [0.8]])
    valid = torch.ones_like(pre)
    valid[1, :, :16, :16] = 0
    batch = dict(
        input=pre,
        target=torch.rand_like(pre),
        condition=clinical,
        mask_pre=valid,
        mask_post=torch.ones_like(pre),
        sia_vector=torch.tensor([[0.6, -0.2], [-0.3, 0.4]]),
        label=[1, 2],
    )
    reader, _ = optical_basis(valid)
    output = forward_batch(model, batch, "cpu")
    torch.testing.assert_close(
        (reader * output["velocity_pred"]).sum((-2, -1)), torch.zeros(2, 2), atol=2e-6, rtol=0
    )
    init = sample_batch(model, batch, args, "cpu")
    base = sample_batch(parent, batch, arguments("fno"), "cpu")
    torch.testing.assert_close(init["sia_vector"], base["sia_vector"], atol=0, rtol=0)
    grad = torch.autograd.grad(output["loss_image"], model.optical_head.weight)[0]
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    with torch.no_grad():
        model.optical_head.weight.normal_(0, 0.04)
        model.optical_to_polar.weight.copy_(torch.eye(2))
    active = forward_batch(model, batch, "cpu")
    torch.testing.assert_close(
        active["loss"], active["loss_image"] + active["loss_sia"] + active["loss_axis"]
    )
    cg, pg = torch.autograd.grad(
        active["loss_sia"] + active["loss_axis"],
        (model.optical_head.weight, model.optical_to_polar.weight),
    )
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in (cg, pg))
    post_mask = torch.ones_like(valid)
    post_mask[:, :, :8] = 0
    changed = forward_batch(
        model,
        dict(batch, target=batch["target"] + 1, sia_vector=-batch["sia_vector"], mask_post=post_mask),
        "cpu",
    )
    torch.testing.assert_close(changed["q_clean"], active["q_clean"], atol=0, rtol=0)
    desired = model.hidden_and_modes(model.condition_encoder(pre), clinical)[1]
    before = sample_batch(model, batch, args, "cpu")
    displacement = (reader * (before["pred"] - pre)).sum((-2, -1)) * statistics()["image"]["scale"]
    torch.testing.assert_close(displacement, desired, atol=5e-6, rtol=1e-5)
    path = tmp_path / "reload.pth"
    try:
        torch.save(
            dict(
                pipeline_version=PIPELINE_VERSION,
                config=vars(args),
                stats=statistics(),
                model=model.state_dict(),
                ema=None,
            ),
            path,
        )
        loaded, _, _, _ = load_checkpoint(path, "cpu", cpu_threads=1)
        after = sample_batch(loaded, batch, args, "cpu")
        for key in ("pred", "sia", "sia_vector"):
            torch.testing.assert_close(after[key], before[key], atol=0, rtol=0)
    finally:
        path.unlink(missing_ok=True)
