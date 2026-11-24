import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cddpm"))
from predict import load_checkpoint
from test_conditional_uno import arguments, statistics
from train import PIPELINE_VERSION, build_model, sample_batch


def make_model(method):
    torch.set_num_threads(1)
    torch.manual_seed(48)
    args = arguments("fno")
    args.fusion_method = method
    return build_model(args, statistics()).eval(), args


def test_clinical_modulation_along_flow_path():
    parent, _ = make_model("film")
    candidate, _ = make_model("early_clinical_film")
    torch.testing.assert_close(parent.state_dict(), candidate.state_dict(), atol=0, rtol=0)
    pre = torch.rand(3, 1, 32, 40)
    clinical = torch.tensor([[0.3], [0.5], [1.2]])
    t = torch.tensor([0., .5, 1.])
    features = candidate.condition_encoder(pre)
    cache = candidate.unet.prepare_condition(features, clinical)
    parent_condition = parent.unet.prepare_condition(features, clinical)
    torch.testing.assert_close(cache.at_time(t), parent_condition, atol=0, rtol=0)
    with torch.no_grad():
        candidate.unet.clinical_fusion.affine.weight.normal_(std=.1)
        parent.unet.load_state_dict(candidate.unet.state_dict())
    cache = candidate.unet.prepare_condition(features, clinical)
    original = parent.unet.prepare_condition(features, clinical)
    assert (cache.gain.abs().sum() > 0) and (cache.shift.abs().sum() > 0)
    expected = torch.stack([2 * original[0] - cache.image[0], original[1], cache.image[2]])
    torch.testing.assert_close(cache.at_time(t), expected)
    out = candidate(pre, torch.rand_like(pre), torch.rand(3, 2), t=t, clinical=clinical)
    out["loss"].backward()
    for p in (candidate.unet.clinical_fusion.affine.weight, candidate.sia_readout[-1].weight):
        assert torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0


def test_sampling_reuses_preoperative_condition_and_reloads(tmp_path):
    model, args = make_model("early_clinical_film")
    with torch.no_grad():
        model.unet.clinical_fusion.affine.bias.normal_(std=.2)
    batch = {"input": torch.rand(2, 1, 32, 40), "condition": torch.tensor([[.2], [.4]]), "label": [1, 2]}
    calls = []
    hook = model.unet.register_forward_pre_hook(lambda _, inputs: calls.append(inputs[2]))
    try:
        expected = sample_batch(model, batch, args, "cpu")
    finally:
        hook.remove()
    assert len(calls) == args.sample_steps and all(cache is calls[0] for cache in calls)
    saved = tmp_path / "selected.pth"
    try:
        torch.save(dict(pipeline_version=PIPELINE_VERSION, config=vars(args), stats=statistics(),
                        model=model.state_dict(), ema=None), saved)
        restored, _, _, _ = load_checkpoint(saved, "cpu", cpu_threads=1)
        actual = sample_batch(restored, batch, args, "cpu")
        for key in ("pred", "sia", "sia_vector"):
            torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
    finally:
        saved.unlink(missing_ok=True)
