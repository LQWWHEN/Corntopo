import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cddpm"))
pytest.importorskip("neuralop")
from predict import load_checkpoint
from train import PIPELINE_VERSION, build_model, forward_batch, get_args, sample_batch


STATS = {"image": {"scale": 20.}, "sia": {"mean": [.2, .3], "std": [.5, .6],
         "polar_median_magnitude": .5, "polar_initial_phase": .4, "polar_axis_training_count": 2}}


def arguments(backbone="fno_pyramid_control"):
    return get_args(["--excel_path","unused","--data_root","unused","--sia_prediction","polar_direct",
                     "--conditions","pre_age_years","--fusion_method","film","--lambda_sia","1",
                     "--image_backbone",backbone,"--fno_modes","4","4","--fno_hidden_channels","8",
                     "--fno_layers","4","--sample_steps","2"])


def batch():
    pre = torch.rand(2, 1, 32, 32)
    return dict(input=pre, target=torch.rand_like(pre), condition=torch.randn(2, 1),
                mask_pre=(torch.rand_like(pre) > .2).float(), mask_post=torch.ones_like(pre),
                sia_vector=torch.tensor([[.4, -.2], [.3, .2]]), label=[1, 2])


def test_zero_controls_preserve_parent_and_learn_at_every_scale():
    torch.set_num_threads(1)
    torch.manual_seed(33)
    parent = build_model(arguments("fno"), STATS).eval()
    rng = torch.get_rng_state()
    torch.manual_seed(33)
    model = build_model(arguments(), STATS).eval()
    assert torch.equal(rng, torch.get_rng_state())
    current = model.state_dict()
    for name, param in parent.state_dict().items():
        torch.testing.assert_close(param, current[name], atol=0, rtol=0)
    b = batch()
    torch.manual_seed(34)
    old = forward_batch(parent, b, "cpu")
    torch.manual_seed(34)
    new = forward_batch(model, b, "cpu")
    for key in ("velocity_pred", "q_clean", "loss_image", "loss_sia", "loss_axis"):
        torch.testing.assert_close(old[key], new[key], atol=0, rtol=0)
    new["loss"].backward()
    for layer in model.unet.control_projections:
        assert torch.isfinite(layer.weight.grad).all() and layer.weight.grad.abs().sum() > 0
    first, second = sample_batch(parent, b, arguments("fno"), "cpu"), sample_batch(model, b, arguments(), "cpu")
    for key in ("pred", "sia", "sia_vector"):
        torch.testing.assert_close(first[key], second[key], atol=0, rtol=0)


def test_fitted_controls_are_aperture_centered_and_preoperative():
    torch.set_num_threads(1)
    model = build_model(arguments(), STATS).eval()
    b = batch()
    with torch.no_grad():
        for layer in model.unet.control_projections:
            layer.weight.normal_(0, .02)
        features = model.condition_encoder(b["input"])
        support = b["mask_pre"]
        empty = model.unet.prepare_condition(features, b["condition"], torch.zeros_like(support))
        valid = model.unet.prepare_condition(features, b["condition"], support)
        for residual, missing in zip(valid.controls, empty.controls):
            raw = model.unet.fno.domain_padding.unpad(residual)
            assert raw.abs().max() > 0 and not torch.count_nonzero(missing)
            assert not torch.count_nonzero(raw * (1 - support))
            torch.testing.assert_close(raw.mean((-2, -1)), torch.zeros_like(raw[...,0,0]), atol=1e-6, rtol=0)
        full, no_support = (model.sample(b["input"], clinical=b["condition"], pre_mask=pm)
                            for pm in (support, torch.zeros_like(support)))
        assert not torch.equal(full["pred"], no_support["pred"])
        torch.testing.assert_close(full["sia_vector"], no_support["sia_vector"], atol=0, rtol=0)


def test_cache_and_selected_weight_reload(tmp_path):
    torch.set_num_threads(1)
    args = arguments()
    model = build_model(args, STATS).eval()
    with torch.no_grad():
        for projection in model.unet.control_projections:
            projection.weight.normal_(0, .02)
    b = batch()
    pre_only = {k:b[k] for k in ("input","condition","mask_pre","label")}
    calls = []
    hook = model.unet.register_forward_pre_hook(lambda module, inputs: calls.append(inputs[2]))
    old = sample_batch(model, pre_only, args, "cpu")
    hook.remove()
    assert len(calls) == 2 and calls[0] is calls[1] and len(calls[0].controls) == 4
    path = tmp_path / "selected.pth"
    try:
        torch.save(dict(pipeline_version=PIPELINE_VERSION,config=vars(args),stats=STATS,model=model.state_dict(),ema=None),path)
        restored, _, _, _ = load_checkpoint(path,"cpu",cpu_threads=1)
        new = sample_batch(restored,pre_only,args,"cpu")
        for key in ("pred","sia","sia_vector"):
            torch.testing.assert_close(old[key],new[key],atol=0,rtol=0)
    finally:
        path.unlink(missing_ok=True)
