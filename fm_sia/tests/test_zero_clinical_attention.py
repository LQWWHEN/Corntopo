import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cddpm"))
pytest.importorskip("neuralop")
from predict import load_checkpoint
from train import PIPELINE_VERSION, build_model, forward_batch, get_args, sample_batch


def arguments(method):
    return get_args([
        "--excel_path", "unused", "--data_root", "unused", "--image_backbone", "fno",
        "--sia_prediction", "polar_direct", "--conditions", "pre_age_years",
        "pre_GENERAL DATA_Cornea Front Astig", "pre_GENERAL DATA_K Max (Front)", "pre_GENERAL DATA_Pachy Min",
        "--fusion_method", method, "--lambda_sia", "1", "--fno_hidden_channels", "8",
        "--fno_modes", "4", "4", "--fno_layers", "1", "--sample_steps", "2",
    ])


STATS = {"sia": {"mean": [.2, .3], "std": [.5, .6], "polar_median_magnitude": .5,
                "polar_initial_phase": .4, "polar_axis_training_count": 2}, "image": {"scale": 20.}}


def test_initial_joint_outputs_and_shared_parameters_match_film():
    torch.set_num_threads(1)
    torch.manual_seed(32)
    parent = build_model(arguments("film"), STATS).eval()
    parent_rng = torch.get_rng_state()
    torch.manual_seed(32)
    model = build_model(arguments("cross_attention_zero"), STATS).eval()
    assert torch.equal(parent_rng, torch.get_rng_state())
    state = model.state_dict()
    for name, value in parent.state_dict().items():
        if not name.startswith("unet.clinical_fusion."):
            torch.testing.assert_close(value, state[name], rtol=0, atol=0)
    pre = torch.rand(2, 1, 32, 32)
    post, q = torch.rand_like(pre), torch.tensor([[.5, .3], [-.2, .3]])
    kwargs = dict(clinical=torch.randn(2, 4), t=torch.tensor([.2, .7]))
    with torch.no_grad():
        first, second = parent(pre, post, q, **kwargs), model(pre, post, q, **kwargs)
    for key in ("velocity_pred", "q_clean", "loss_image", "loss_sia", "loss_axis"):
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)


def test_attention_learns_spatially_varying_clinical_response_and_reloads(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(32)
    args = arguments("cross_attention_zero")
    model = build_model(args, STATS).eval()
    pre = torch.rand(2, 1, 32, 32)
    clinical = torch.randn(2, 4)
    batch = dict(input=pre, target=torch.rand_like(pre), condition=clinical, mask_pre=torch.ones_like(pre),
                 mask_post=torch.ones_like(pre), sia_vector=torch.tensor([[.3, -.1], [.1, .5]]), label=[1, 2])
    optimizer = torch.optim.SGD(model.parameters(), lr=.02)
    loss = forward_batch(model, batch, "cpu")["loss"]
    loss.backward()
    attention = model.unet.clinical_fusion.attention
    assert attention.out_proj.weight.grad.abs().sum() > 0
    optimizer.step()
    optimizer.zero_grad()
    changed = clinical.clone()
    changed[:, 1] += 1
    features = model.condition_encoder(pre)
    difference = model.unet.prepare_condition(features, clinical) - model.unet.prepare_condition(features, changed)
    assert difference.flatten(2).std(-1).max() > 0
    forward_batch(model, batch, "cpu")["loss"].backward()
    assert torch.isfinite(attention.in_proj_weight.grad).all() and attention.in_proj_weight.grad.abs().sum() > 0
    pre_only = {key: batch[key] for key in ("input", "mask_pre", "condition", "label")}
    calls = []
    hook = attention.register_forward_pre_hook(lambda module, inputs: calls.append(tuple(inputs[1].shape)))
    before = sample_batch(model, pre_only, args, "cpu")
    hook.remove()
    assert calls == [(2, 5, 64)]
    path = tmp_path / "reload.pth"
    try:
        torch.save(dict(pipeline_version=PIPELINE_VERSION, config=vars(args), stats=STATS,
                        model=model.state_dict(), ema=None), path)
        restored, _, _, _ = load_checkpoint(path, "cpu", cpu_threads=1)
        after = sample_batch(restored, pre_only, args, "cpu")
        for key in ("pred", "sia", "sia_vector"):
            torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
    finally:
        path.unlink(missing_ok=True)
