import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cddpm"))
pytest.importorskip("neuralop")
from model.axis_factorized_fno import AxisSpectralConv
from predict import load_checkpoint
from train import PIPELINE_VERSION, build_model, forward_batch, sample_batch
from test_conditional_uno import arguments, statistics


@pytest.mark.parametrize("shape", [(9, 14), (10, 13)])
def test_each_axis_matches_a_low_pass_on_an_oblique_wave(shape):
    h, w = shape
    y, x = torch.meshgrid(torch.arange(h) / h, torch.arange(w) / w, indexing="ij")
    wave = torch.cos(2 * torch.pi * (y + 4 * x))[None, None]
    layer = AxisSpectralConv(1, (4, 4), torch.zeros(1, 1, 1))
    with torch.no_grad():
        layer.weights[0].fill_(1)
        layer.weights[1].zero_()
        torch.testing.assert_close(layer(wave), wave, atol=2e-6, rtol=0)
        layer.weights[0].zero_()
        layer.weights[1].fill_(1)
        torch.testing.assert_close(layer(wave), torch.zeros_like(wave), atol=2e-6, rtol=0)


def test_shared_initialization_joint_gradient_and_frozen_reload(tmp_path):
    torch.set_num_threads(1)
    args = arguments("axis_factorized_fno")
    torch.manual_seed(34)
    parent = build_model(arguments("fno"), statistics()).eval()
    rng = torch.get_rng_state()
    torch.manual_seed(34)
    model = build_model(args, statistics()).eval()
    assert torch.equal(rng, torch.get_rng_state())
    state = model.state_dict()
    for name, value in parent.state_dict().items():
        if ".convs." not in name or name.endswith(".bias"):
            torch.testing.assert_close(state[name], value, rtol=0, atol=0)

    pre = torch.rand(2, 1, 32, 32)
    batch = dict(input=pre, target=torch.rand_like(pre), condition=torch.tensor([[.2], [.5]]),
                 mask_pre=torch.ones_like(pre), mask_post=torch.ones_like(pre),
                 sia_vector=torch.tensor([[.3, -.1], [.1, .5]]), label=[1, 2])
    loss = forward_batch(model, batch, "cpu")
    torch.testing.assert_close(loss["loss"], loss["loss_image"] + loss["loss_sia"] + loss["loss_axis"])
    loss["loss"].backward()
    for conv in model.unet.fno.fno_blocks.convs:
        for weight in conv.weights:
            assert torch.isfinite(weight.grad).all() and weight.grad.abs().sum() > 0
    for key in ("target", "mask_post", "sia_vector"):
        del batch[key]
    result = sample_batch(model, batch, args, "cpu")
    ref = sample_batch(parent, batch, arguments("fno"), "cpu")
    torch.testing.assert_close(result["sia_vector"], ref["sia_vector"], rtol=0, atol=0)
    path = tmp_path / "reload.pth"
    try:
        torch.save(dict(pipeline_version=PIPELINE_VERSION, config=vars(args), stats=statistics(),
                        model=model.state_dict(), ema=None), path)
        restored, _, _, _ = load_checkpoint(path, "cpu", cpu_threads=1)
        again = sample_batch(restored, batch, args, "cpu")
        for key in ("pred", "sia", "sia_vector"):
            torch.testing.assert_close(again[key], result[key], rtol=0, atol=0)
    finally:
        path.unlink(missing_ok=True)
