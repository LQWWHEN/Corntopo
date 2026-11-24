"""Verify field selection by the strength of paired preoperative angular moments."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cddpm"))
from predict import load_checkpoint
from test_conditional_uno import statistics
from train import PIPELINE_VERSION, build_model, get_args


def recipe(fusion="clinical_field_attention"):
    return get_args(["--excel_path", "unused", "--data_root", "unused", "--conditions", "pre_age_years", "pre_GENERAL DATA_Cornea Front Astig", "pre_GENERAL DATA_K Max (Front)", "pre_GENERAL DATA_Pachy Min",
                     "--fusion_method", "film", "--sia_prediction", "polar_direct", "--lambda_sia", "1",
                     "--image_backbone", "fno", "--fno_modes", "6", "6", "--fno_hidden_channels", "8",
                     "--sample_steps", "2", "--sia_feature_fusion", fusion])


def test_query_is_invariant_to_moment_pair_rotation_and_can_distinguish_fields():
    torch.set_num_threads(1)
    model = build_model(recipe(), statistics()).eval()
    dirs = torch.randn(2, 2, 512)
    tokens = torch.randn(2, 4, 128)
    with torch.no_grad():
        torch.testing.assert_close(model.field_weights(dirs, tokens), torch.full((2,4), .25))
        model.field_query.weight.normal_(std=.1)
        weights = model.field_weights(dirs, tokens)
        rotated = torch.stack((-dirs[:,1], dirs[:,0]), dim=1)
        torch.testing.assert_close(model.field_weights(rotated, tokens), weights)
        torch.testing.assert_close(model.field_weights(torch.zeros_like(dirs), tokens), torch.full((2,4), .25))
        torch.testing.assert_close(weights.sum(dim=1), torch.ones(2))
        assert (weights - .25).abs().max() > 1e-4


def test_joint_initialization_gradients_and_reload(tmp_path):
    torch.manual_seed(50)
    torch.set_num_threads(1)
    parent = build_model(recipe("lab"), statistics()).eval()
    torch.manual_seed(50)
    args = recipe()
    model = build_model(args, statistics()).eval()
    for n, p in parent.state_dict().items():
        torch.testing.assert_close(model.state_dict()[n], p, rtol=0, atol=0)
    pre = torch.rand(2, 1, 32, 40)
    post, clinical = torch.rand_like(pre), torch.tensor([[.1,.4,-.2,.3], [.8,-.4,.2,.5]])
    vec, t = torch.tensor([[.48, .15], [-.3, .22]]), torch.tensor([.2, .8])
    out = model(pre, post, vec, clinical=clinical, t=t)
    old = parent(pre, post, vec, clinical=clinical, t=t)
    for key in ("velocity_pred", "q_clean", "loss_image", "loss_axis", "loss_sia"):
        torch.testing.assert_close(out[key], old[key], atol=0, rtol=0)
    with torch.no_grad():
        model.sia_readout[-1].weight.normal_(0, .03)
        model.field_query.weight.normal_(0, .03)
    out = model(pre, post, vec, clinical=clinical, t=t)
    weights = [model.field_query.weight, model.unet.clinical_encoder.scalar_projection.weight, model.unet.clinical_encoder.feature_embedding]
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
