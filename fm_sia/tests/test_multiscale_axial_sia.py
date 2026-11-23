"""Verify native-grid angular pooling and its gradient path into the shared image projections."""
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cddpm"))
from model.axial_sia import axial_moments
from predict import load_checkpoint
from test_conditional_uno import arguments, statistics
from train import PIPELINE_VERSION, build_model, get_args


def recipe():
    return get_args(["--excel_path", "unused", "--data_root", "unused", "--conditions", "pre_age_years",
                     "--fusion_method", "film", "--sia_prediction", "polar_direct", "--lambda_sia", "1",
                     "--image_backbone", "fno", "--fno_modes", "6", "6", "--fno_hidden_channels", "8",
                     "--sample_steps", "2", "--sia_feature_fusion", "multiscale_axial"])


def test_native_projection_against_spatial_reference():
    torch.set_num_threads(1)
    model = build_model(recipe(), statistics()).eval()
    clinical = torch.tensor([[0.3], [-0.2]])
    features = [torch.randn(2, channels, height, width)
                for channels, height, width in ((128, 25, 32), (256, 12, 16), (512, 6, 8), (512, 3, 4))]
    with torch.no_grad():
        model.axial_scale_projection.weight.normal_(0, .08)
        reference = []
        for feature, layer in zip(features, model.unet.condition_projection):
            projected = F.conv2d(feature, layer.weight)
            reference.append(axial_moments(projected).reshape(2, 3, 16)[:, 1:])
        delta = model.axial_scale_projection(torch.cat(reference, dim=-1))
        deep = axial_moments(features[-1]).reshape(2, 3, 512)
        values = model.sia_features(features, clinical)
        torch.testing.assert_close(values[:, :512], deep[:, 0], atol=0, rtol=0)
        torch.testing.assert_close(values[:, 512:1536].reshape(2, 2, 512), deep[:, 1:]+delta)
        torch.testing.assert_close(values[:, 1536:], model.unet.clinical_encoder(clinical)[0])


def test_joint_initialization_gradients_and_reload(tmp_path):
    torch.manual_seed(47)
    torch.set_num_threads(1)
    parent = build_model(arguments("fno"), statistics()).eval()
    torch.manual_seed(47)
    args = recipe()
    model = build_model(args, statistics()).eval()
    for n, p in parent.state_dict().items():
        torch.testing.assert_close(model.state_dict()[n], p, rtol=0, atol=0)
    pre = torch.rand(2, 1, 32, 40)
    post, clinical = torch.rand_like(pre), torch.tensor([[.1], [.8]])
    vec, t = torch.tensor([[.48, .15], [-.3, .22]]), torch.tensor([.2, .8])
    out = model(pre, post, vec, clinical=clinical, t=t)
    old = parent(pre, post, vec, clinical=clinical, t=t)
    for key in ("velocity_pred", "q_clean", "loss_image", "loss_axis", "loss_sia"):
        torch.testing.assert_close(out[key], old[key], atol=0, rtol=0)
    with torch.no_grad():
        model.sia_readout[-1].weight.normal_(0, .03)
        model.axial_scale_projection.weight.normal_(0, .03)
    out = model(pre, post, vec, clinical=clinical, t=t)
    weights = [model.axial_scale_projection.weight, *[layer.weight for layer in model.unet.condition_projection]]
    grads = torch.autograd.grad(out["loss_axis"] + out["loss_sia"], weights)
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)
    before = model.sample(pre, clinical=clinical)
    torch.testing.assert_close(before["sia_vector"], out["q_clean"], atol=0, rtol=0)
    path = tmp_path / "selected.pth"
    try:
        torch.save(dict(pipeline_version=PIPELINE_VERSION, config=vars(args), stats=statistics(),
                        model=model.state_dict(), ema=None), path)
        loaded, _, _, _ = load_checkpoint(path, "cpu", cpu_threads=1)
        after = loaded.sample(pre, clinical=clinical)
        for key in ("pred", "sia", "sia_vector"):
            torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
    finally:
        path.unlink(missing_ok=True)
