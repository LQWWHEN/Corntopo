"""Checks for the query direction, aperture support and the maintained joint entry points."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cddpm"))
from model.axial_sia import axial_moments, axial_weights
from model.clinical_spatial_pool import ClinicalSpatialPool
from predict import load_checkpoint
from test_conditional_uno import arguments, statistics
from train import PIPELINE_VERSION, build_model, get_args


def test_query_selects_within_patient_and_keeps_both_direction_moments():
    pool = ClinicalSpatialPool(1, 1, key_width=1)
    feature = torch.arange(25.).reshape(1, 1, 5, 5).expand(2, -1, -1, -1).clone()
    feature[:, :, 0, 0] = 1000
    clinical = torch.tensor([[1.], [-1.]])
    parent = axial_moments(feature)
    torch.testing.assert_close(pool(feature, clinical), parent, atol=0, rtol=0)
    with torch.no_grad():
        pool.key.weight.fill_(.2)
        pool.query.weight.fill_(1)
    moments = pool(feature, clinical)
    assert moments[0, 0] > parent[0, 0] > moments[1, 0]
    torch.testing.assert_close(moments[:, 1:], parent[:, 1:], atol=0, rtol=0)
    changed = feature.clone()
    changed[:, :, axial_weights(feature)[0] == 0] = -2000
    torch.testing.assert_close(pool(changed, clinical), moments, atol=0, rtol=0)
    torch.testing.assert_close(pool(feature[:1], clinical[:1]), moments[:1])


def test_joint_initialization_gradients_preoperative_inference_and_reload(tmp_path):
    torch.set_num_threads(1)
    base = arguments("fno")
    torch.manual_seed(41)
    parent = build_model(base, statistics()).eval()
    rng = torch.get_rng_state()
    recipe = get_args(["--excel_path", "unused", "--data_root", "unused", "--conditions", "pre_age_years",
                       "--fusion_method", "film", "--sia_prediction", "polar_direct", "--lambda_sia", "1",
                       "--image_backbone", "fno", "--fno_modes", "6", "6", "--fno_hidden_channels", "8",
                       "--sample_steps", "2", "--sia_feature_fusion", "clinical_spatial_pool"])
    torch.manual_seed(41)
    model = build_model(recipe, statistics()).eval()
    assert torch.equal(torch.get_rng_state(), rng)
    for name, value in parent.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name], value, atol=0, rtol=0)
    pre = torch.rand(2, 1, 32, 32)
    post = torch.rand_like(pre)
    clinical = torch.tensor([[.2], [.8]])
    target = torch.tensor([[.6, -.3], [-.3, .4]])
    t = torch.tensor([.2, .7])
    full = model(pre, post, target, clinical=clinical, t=t)
    ref = parent(pre, post, target, clinical=clinical, t=t)
    for key in ("velocity_pred", "q_clean", "loss_image", "loss_sia", "loss_axis"):
        torch.testing.assert_close(full[key], ref[key], atol=0, rtol=0)
    # The parent output head starts constant. Activate it to test the new gradient path.
    with torch.no_grad():
        model.sia_readout[-1].weight.normal_(0, .01)
    loss = model(pre, post, target, clinical=clinical, t=t)["loss"]
    grad = torch.autograd.grad(loss, model.spatial_pool.query.weight)[0]
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    with torch.no_grad():
        model.spatial_pool.query.weight.normal_(0, .02)
    active = model(pre, post, target, clinical=clinical, t=t)
    key_grad = torch.autograd.grad(active["loss_sia"] + active["loss_axis"], model.spatial_pool.key.weight)[0]
    assert torch.isfinite(key_grad).all() and key_grad.abs().sum() > 0
    shifted = model(pre, post + .5, -target, clinical=clinical, t=1 - t)
    torch.testing.assert_close(shifted["q_clean"], active["q_clean"], atol=0, rtol=0)
    before = model.sample(pre, clinical=clinical)
    path = tmp_path / "reload.pth"
    try:
        torch.save(dict(pipeline_version=PIPELINE_VERSION, config=vars(recipe), stats=statistics(),
                        model=model.state_dict(), ema=None), path)
        loaded, _, _, _ = load_checkpoint(path, "cpu", cpu_threads=1)
        after = loaded.sample(pre, clinical=clinical)
        for key in ("pred", "sia", "sia_vector"):
            torch.testing.assert_close(before[key], after[key], atol=0, rtol=0)
    finally:
        path.unlink(missing_ok=True)
