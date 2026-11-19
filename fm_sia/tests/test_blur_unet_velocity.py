import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cddpm"))
pytest.importorskip("neuralop")
from model.blur_unet_velocity import BlurPool, BlurUNetVelocity
from predict import load_checkpoint
from train import PIPELINE_VERSION, build_model, sample_batch, forward_batch
from test_conditional_uno import arguments, statistics


def test_downsampling_rejects_checkerboard_and_preserves_interior_mean():
    coords = torch.arange(19)
    checker = (1 - 2 * (coords[:, None] + coords[None, :]).remainder(2)).float()
    inputs = torch.stack((checker, torch.ones_like(checker)))[None]
    out = BlurPool(2)(inputs)
    torch.testing.assert_close(out[0, 0, 1:-1, 1:-1], torch.zeros(8, 8))
    torch.testing.assert_close(out[0, 1, 1:-1, 1:-1], torch.ones(8, 8))


def test_multiscale_odd_grid_and_gradients():
    torch.set_num_threads(1)
    op = BlurUNetVelocity(8)
    x = torch.randn(2, 65, 29, 35, requires_grad=True)
    sizes = []
    hook = op.coarse.register_forward_hook(lambda module, args, result: sizes.append(result.shape[-2:]))
    velocity = op(x)
    hook.remove()
    assert velocity.shape == (2, 1, 29, 35) and sizes == [(8, 9)]
    velocity.square().mean().backward()
    for block in (op.fine, op.middle, op.coarse, op.decode_middle, op.decode_fine):
        grad = block[0].weight.grad
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_parent_condition_and_sia_match_and_saved_sampling(tmp_path):
    torch.set_num_threads(1)
    args = arguments("blur_unet")
    torch.manual_seed(36)
    parent = build_model(arguments("fno"), statistics()).eval()
    rng = torch.get_rng_state()
    torch.manual_seed(36)
    model = build_model(args, statistics()).eval()
    assert torch.equal(rng, torch.get_rng_state())
    state = model.state_dict()
    for name, value in parent.state_dict().items():
        if not name.startswith("unet.fno."):
            torch.testing.assert_close(state[name], value, rtol=0, atol=0)
    pre = torch.rand(2, 1, 32, 32)
    batch = dict(input=pre, condition=torch.tensor([[.2], [.8]]), mask_pre=torch.ones_like(pre), label=[1, 2])
    loss = forward_batch(model, dict(batch, target=torch.rand_like(pre), mask_post=torch.ones_like(pre),
                                     sia_vector=torch.tensor([[.3, -.1], [.1, .5]])), "cpu")
    torch.testing.assert_close(loss["loss"], loss["loss_image"] + loss["loss_axis"] + loss["loss_sia"])
    loss["loss"].backward()
    assert model.unet.fno.output.weight.grad.abs().sum() > 0
    calls = []
    hook = model.unet.clinical_fusion.register_forward_hook(lambda *unused: calls.append(1))
    before = sample_batch(model, batch, args, "cpu")
    hook.remove()
    assert calls == [1]
    ref = sample_batch(parent, batch, arguments("fno"), "cpu")
    torch.testing.assert_close(before["sia_vector"], ref["sia_vector"], rtol=0, atol=0)
    path = tmp_path / "reload.pth"
    try:
        torch.save(dict(pipeline_version=PIPELINE_VERSION, config=vars(args), stats=statistics(),
                        model=model.state_dict(), ema=None), path)
        restored, _, _, _ = load_checkpoint(path, "cpu", cpu_threads=1)
        after = sample_batch(restored, batch, args, "cpu")
        for key in ("pred", "sia_vector", "sia"):
            torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
    finally:
        path.unlink(missing_ok=True)
