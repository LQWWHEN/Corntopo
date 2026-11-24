import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cddpm"))
pytest.importorskip("neuralop")
from model.cosine_operator import CosineSpectralConv
from predict import load_checkpoint
from train import PIPELINE_VERSION, build_model, forward_batch, sample_batch
from test_conditional_uno import arguments, statistics


def test_non_square_cosine_modes_and_clipping():
    import math
    for h, w in ((9,14),(10,13)):
        y = torch.arange(h, dtype=torch.float64) + .5
        x = torch.arange(w, dtype=torch.float64) + .5
        wave = (torch.cos(math.pi * 2 * y / h)[:,None] * torch.cos(math.pi * 3 * x / w))[None,None]
        high = (torch.cos(math.pi * 6 * y / h)[:,None] * torch.ones_like(x))[None,None]
        layer = CosineSpectralConv(1, (4,w+2), torch.zeros(1,1,1)).double()
        with torch.no_grad():
            layer.weight.fill_(1)
            layer.weight[:,:,2,3].fill_(2)
        torch.testing.assert_close(layer(1 + wave + high), 1 + 2*wave, atol=1e-11, rtol=0)


def test_shared_initialization_joint_gradient_and_frozen_reload(tmp_path):
    torch.set_num_threads(1)
    args = arguments("cosine_operator")
    torch.manual_seed(49)
    parent = build_model(arguments("fno"), statistics()).eval()
    rng = torch.get_rng_state()
    torch.manual_seed(49)
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
        assert torch.isfinite(conv.weight.grad).all() and conv.weight.grad.abs().sum() > 0
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
