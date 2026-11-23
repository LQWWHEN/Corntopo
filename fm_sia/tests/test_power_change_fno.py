"""Check the evolving power-change channel and the absolute postoperative output."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cddpm"))
from model.fm_common import masked_mse
from predict import load_checkpoint
from test_conditional_uno import arguments, statistics
from train import PIPELINE_VERSION, build_model, sample_batch


def test_same_weights_and_labels_with_relative_flow_state():
    torch.set_num_threads(1)
    nets = []
    for backbone in ("fno", "power_change_fno"):
        torch.manual_seed(46)
        nets.append(build_model(arguments(backbone), statistics()).eval())
    parent, candidate = nets
    torch.testing.assert_close(candidate.state_dict(), parent.state_dict(), atol=0, rtol=0)
    pre = torch.rand(2, 1, 32, 40)
    post = torch.rand_like(pre)
    clinical = torch.tensor([[0.1], [0.6]])
    target = torch.tensor([[0.3, -0.1], [0.2, 0.5]])
    t = torch.tensor([0.0, 0.7])
    mask = torch.ones_like(pre)
    mask[:, :, :3] = 0
    inputs = []
    for model in nets:
        handle = model.unet.register_forward_pre_hook(lambda _, args: inputs.append(args[0].detach()))
        try:
            out = model(pre, post, target, clinical=clinical, t=t, mask=mask)
        finally:
            handle.remove()
        torch.testing.assert_close(out["loss_image"], masked_mse(out["velocity_pred"], post-pre, mask))
        if model is parent:
            parent_q = out["q_clean"].detach()
    tau = t[:, None, None, None]
    torch.testing.assert_close(inputs[0], (1-tau)*pre+tau*post)
    torch.testing.assert_close(inputs[1], tau*(post-pre))
    torch.testing.assert_close(out["q_clean"], parent_q, atol=0, rtol=0)
    out["loss"].backward()
    for p in (candidate.unet.condition_projection[0].weight, candidate.sia_readout[-1].weight):
        assert torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0


def test_absolute_reconstruction_through_sampling_and_reload(tmp_path):
    torch.set_num_threads(1)
    args = arguments("power_change_fno")
    model = build_model(args, statistics()).eval()
    batch = {"input": torch.rand(2, 1, 32, 40), "condition": torch.tensor([[0.1], [0.4]]), "label": [1, 2]}
    calls = []
    hook = model.unet.register_forward_hook(lambda _, args, output: calls.append((args[0], args[2], output)))
    try:
        actual = sample_batch(model, batch, args, "cpu")
    finally:
        hook.remove()
    assert len(calls) == args.sample_steps
    accumulated = torch.zeros_like(batch["input"])
    for state, condition, velocity in calls:
        assert condition is calls[0][1]
        torch.testing.assert_close(state, accumulated, atol=0, rtol=0)
        accumulated = accumulated + velocity / args.sample_steps
    torch.testing.assert_close(actual["pred"], batch["input"] + accumulated, atol=0, rtol=0)
    saved = tmp_path / "selected.pth"
    try:
        torch.save(dict(pipeline_version=PIPELINE_VERSION, config=vars(args), stats=statistics(),
                        model=model.state_dict(), ema=None), saved)
        restored, _, _, _ = load_checkpoint(saved, "cpu", cpu_threads=1)
        recovered = sample_batch(restored, batch, args, "cpu")
        for key in ("pred", "sia", "sia_vector"):
            torch.testing.assert_close(actual[key], recovered[key], atol=0, rtol=0)
    finally:
        saved.unlink(missing_ok=True)
