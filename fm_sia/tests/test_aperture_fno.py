"""Check spatial support at each Fourier block and through the joint training entry points."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cddpm"))
from predict import load_checkpoint
from test_conditional_uno import arguments, statistics
from train import PIPELINE_VERSION, build_model, forward_batch, sample_batch


def test_full_grid_recovers_parent_with_no_padding():
    torch.set_num_threads(1)
    nets = []
    for name in ("fno", "aperture_fno"):
        recipe = arguments(name)
        recipe.fno_domain_padding = 0
        torch.manual_seed(44)
        nets.append(build_model(recipe, statistics()).eval())
    parent, trial = nets
    torch.testing.assert_close(trial.state_dict(), parent.state_dict(), atol=0, rtol=0)
    pre = torch.rand(2, 1, 32, 40)
    clinical = torch.tensor([[.1], [.5]])
    features = parent.condition_encoder(pre)
    time = torch.tensor([.2, .7])
    original = parent.image_velocity(pre, time, features, clinical)
    actual = trial.image_velocity(pre, time, features, clinical, torch.ones_like(pre))
    torch.testing.assert_close(actual, original, atol=0, rtol=0)


def test_aperture_blocks_gradients_and_checkpoint_sampling(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(44)
    args = arguments("aperture_fno")
    model = build_model(args, statistics()).eval()
    pre = torch.rand(2, 1, 32, 40)
    support = torch.zeros_like(pre)
    support[0, :, 3:26, 2:25] = 1
    support[1, :, 7:27, 9:32] = 1
    support[:, :, 14:16, 17:20] = 0
    pre = pre * support
    clinical = torch.tensor([[.1], [.5]])
    b = dict(input=pre, target=torch.rand_like(pre), condition=clinical, mask_pre=support,
             mask_post=torch.ones_like(pre), sia_vector=torch.tensor([[.3, -.1], [.2, .5]]), label=[1, 2])
    blocks = []
    handle = model.unet.fno.fno_blocks.register_forward_pre_hook(lambda mod, inputs: blocks.append(inputs[0]))
    try:
        out = forward_batch(model, b, "cpu")
    finally:
        handle.remove()
    padded = model.unet.fno.domain_padding.pad(support)
    assert len(blocks) == args.fno_layers
    for block in blocks:
        assert block.abs().max() > 0 and not (block * (1 - padded)).count_nonzero()
    assert not (out["velocity_pred"] * (1 - support)).count_nonzero()
    out["loss"].backward()
    for param in (model.unet.condition_projection[-1].weight, model.sia_readout[-1].weight):
        assert torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0
    with torch.no_grad():
        features = model.condition_encoder(pre)
        prepared = model.unet.prepare_condition(features, clinical, support)
        state = torch.rand_like(pre)
        time = torch.tensor([.2, .7])
        a = model.unet(state, time, prepared)
        bstate = model.unet(state + 100 * (1 - support), time, prepared)
        torch.testing.assert_close(a, bstate, atol=0, rtol=0)
    pre_only = {k: b[k] for k in ("input", "condition", "mask_pre", "label")}
    calls = []
    handle = model.unet.register_forward_pre_hook(lambda mod, inputs: calls.append(inputs[2]))
    try:
        before = sample_batch(model, pre_only, args, "cpu")
    finally:
        handle.remove()
    assert len(calls) == args.sample_steps and calls[0] is calls[-1]
    assert not ((before["pred"] - pre) * (1 - support)).count_nonzero()
    assert ((before["pred"] - pre) * support).abs().sum() > 0
    path = tmp_path / "selected.pth"
    try:
        torch.save(dict(pipeline_version=PIPELINE_VERSION, config=vars(args), stats=statistics(),
                        model=model.state_dict(), ema=None), path)
        restored, _, _, _ = load_checkpoint(path, "cpu", cpu_threads=1)
        after = sample_batch(restored, pre_only, args, "cpu")
        for key in ("pred", "sia", "sia_vector"):
            torch.testing.assert_close(before[key], after[key], atol=0, rtol=0)
    finally:
        path.unlink(missing_ok=True)
