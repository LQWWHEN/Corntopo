import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cddpm"))
pytest.importorskip("neuralop")
from model.blur_unet_velocity import BlurPool, BlurUNetVelocity, PyramidReadoutUNet, haar_split
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


def test_haar_detail_retains_three_orientations_and_reconstructs_odd_edges():
    rows, cols = torch.meshgrid(torch.arange(7), torch.arange(9), indexing="ij")
    x = torch.stack([torch.ones_like(rows), (-1) ** rows, (-1) ** cols, (-1) ** (rows + cols)]).float()[None]
    low, detail = haar_split(x)
    assert torch.equal(detail[:, 0], torch.zeros_like(detail[:, 0]))
    torch.testing.assert_close(detail[:, 1:, :6, :8], x[:, 1:, :6, :8])
    torch.testing.assert_close(haar_split(detail)[0], torch.zeros_like(low))
    reconstructed = low.repeat_interleave(2, -2).repeat_interleave(2, -1)[..., :7, :9] + detail
    torch.testing.assert_close(reconstructed, x)


def test_pyramid_readout_cell_means_and_band_gradients():
    torch.manual_seed(51)
    op = PyramidReadoutUNet(4).double()
    coarse = torch.randn(2, 16, 3, 2, dtype=torch.float64, requires_grad=True)
    middle = torch.randn(2, 8, 5, 4, dtype=torch.float64, requires_grad=True)
    fine = torch.randn(2, 4, 9, 7, dtype=torch.float64, requires_grad=True)
    velocity = op.readout(coarse, middle, fine)
    midpoint, fine_detail = haar_split(velocity)
    mean, middle_detail = haar_split(midpoint)
    torch.testing.assert_close(mean, op.coarse_output(coarse) + op.output.bias)
    torch.testing.assert_close(middle_detail, haar_split(op.middle_output(middle))[1])
    torch.testing.assert_close(fine_detail, haar_split(op.output(fine))[1])
    # A coarse-only loss has zero derivative through both local correction heads.
    local_grads = torch.autograd.grad(mean.square().sum(), (fine, middle), retain_graph=True)
    for grad in local_grads:
        torch.testing.assert_close(grad, torch.zeros_like(grad), rtol=0, atol=1.e-12)
    velocity.square().mean().backward()
    for head in (op.coarse_output, op.middle_output, op.output):
        assert torch.isfinite(head.weight.grad).all() and head.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("backbone,parent_backbone", [("blur_unet", "fno"), ("haar_detail_unet", "blur_unet"),
                                                      ("pyramid_readout_unet", "haar_detail_unet")])
def test_parent_condition_and_sia_match_and_saved_sampling(tmp_path, backbone, parent_backbone):
    torch.set_num_threads(1)
    args = arguments(backbone)
    torch.manual_seed(36)
    parent = build_model(arguments(parent_backbone), statistics()).eval()
    rng = torch.get_rng_state()
    torch.manual_seed(36)
    model = build_model(args, statistics()).eval()
    assert torch.equal(rng, torch.get_rng_state())
    state = model.state_dict()
    if backbone == "haar_detail_unet":
        assert model.unet.fno.haar_details
        parent_params = dict(parent.named_parameters())
        assert parent_params.keys() == dict(model.named_parameters()).keys()
        for name, value in model.named_parameters():
            torch.testing.assert_close(value, parent_params[name], rtol=0, atol=0)
    for name, value in parent.state_dict().items():
        if backbone == "pyramid_readout_unet" or not name.startswith("unet.fno."):
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
    ref = sample_batch(parent, batch, arguments(parent_backbone), "cpu")
    torch.testing.assert_close(before["sia_vector"], ref["sia_vector"], rtol=0, atol=0)
    if backbone in ("haar_detail_unet", "pyramid_readout_unet"):
        assert not torch.equal(before["pred"], ref["pred"])
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
