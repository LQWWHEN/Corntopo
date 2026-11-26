import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cddpm"))
from model.aperture_sia import aperture_moments
from model.axial_sia import axial_moments


def test_full_aperture_matches_parent_and_partial_aperture_rejects_constant_leakage():
    torch.manual_seed(4)
    feature = torch.randn(2, 3, 16, 16, dtype=torch.float64, requires_grad=True)
    mask = torch.ones(2, 1, 32, 32, dtype=feature.dtype)
    torch.testing.assert_close(aperture_moments(feature, mask), axial_moments(feature))
    mask[0, :, :, :15] = 0
    mask[1, :, 7:26, :21] = 0
    constant = aperture_moments(torch.ones_like(feature), mask).reshape(2, 3, 3)
    torch.testing.assert_close(constant[:, 0], torch.ones_like(constant[:, 0]))
    torch.testing.assert_close(constant[:, 1:], torch.zeros_like(constant[:, 1:]))
    actual = aperture_moments(feature, mask).reshape(2, 3, 3)
    mirrored = aperture_moments(feature.flip(-1), mask.flip(-1)).reshape_as(actual)
    torch.testing.assert_close(mirrored, actual * actual.new_tensor([1, 1, -1])[None, :, None])
    actual.square().sum().backward()
    assert torch.isfinite(feature.grad).all() and torch.count_nonzero(feature.grad[0, :, :, :7]) == 0
    assert torch.count_nonzero(feature.grad) > 0
    torch.testing.assert_close(aperture_moments(feature, mask * 0), torch.zeros(2, 9, dtype=feature.dtype))


def test_partial_cells_have_area_weighted_mean():
    mask = torch.zeros(1, 1, 8, 8)
    mask[..., 2:4, 2:4] = 1
    mask[..., 2, 4] = 1
    feature = torch.zeros(1, 1, 4, 4)
    feature[..., 1, 1] = 3
    feature[..., 1, 2] = 8
    assert aperture_moments(feature, mask)[0, 0].item() == pytest.approx(4.)


def test_uncentered_aperture_preserves_mean_and_only_leaks_constants_on_asymmetric_support():
    feature = torch.ones(1, 1, 8, 8, dtype=torch.float64)
    mask = torch.ones_like(feature)
    torch.testing.assert_close(aperture_moments(feature, mask, center_basis=False),
                               aperture_moments(feature, mask))
    mask[..., 3:, :] = 0
    raw = aperture_moments(feature, mask, center_basis=False)
    centered = aperture_moments(feature, mask)
    torch.testing.assert_close(raw[:, 0], centered[:, 0])
    torch.testing.assert_close(centered[:, 1:], torch.zeros_like(centered[:, 1:]))
    assert abs(raw[0, 1]) > .1


def test_normal_entry_points_and_saved_weight_reload(tmp_path):
    from predict import load_checkpoint
    from train import PIPELINE_VERSION, build_model, forward_batch, get_args, sample_batch, saved_config_value
    torch.set_num_threads(1)
    flags = ["--excel_path", "unused", "--data_root", "unused", "--sia_prediction","polar_direct",
             "--conditions", "pre_age_years", "--fusion_method", "film", "--lambda_sia","1",
             "--sample_steps","2","--image_backbone","fno","--fno_modes","4","4",
             "--fno_hidden_channels","8","--fno_layers","1"]
    a = get_args(flags + ["--sia_aperture_moments"])
    stats = dict(sia=dict(mean=[.2,.3], std=[.5,.6], polar_median_magnitude=.5,
                          polar_initial_phase=.4, polar_axis_training_count=2),image=dict(scale=20.))
    assert saved_config_value({}, "sia_aperture_moments") is False
    torch.manual_seed(82)
    parent = build_model(get_args(flags), stats)
    torch.manual_seed(82)
    model = build_model(a, stats).eval()
    assert model.state_dict().keys() == parent.state_dict().keys()
    for k,v in parent.state_dict().items():
        torch.testing.assert_close(model.state_dict()[k], v, rtol=0, atol=0)
    with torch.no_grad():
        model.sia_readout[-1].weight.normal_(0,.1)
    pre = torch.rand(2, 1, 32, 32)
    mask = torch.ones_like(pre)
    mask[:, :, :13, :17] = 0
    batch = dict(input=pre, target=torch.rand_like(pre), condition=torch.zeros(2,1), mask_pre=mask,
                 mask_post=torch.ones_like(pre), sia_vector=torch.tensor([[.6,-.3],[.1,.4]]),label=[1,2])
    before = sample_batch(model, batch, a, "cpu")
    train = forward_batch(model, batch, "cpu")
    torch.testing.assert_close(train["q_clean"], before["sia_vector"])
    altered = forward_batch(model, dict(batch, target=1-pre, mask_post=pre>.5, sia_vector=-batch["sia_vector"]), "cpu")
    torch.testing.assert_close(altered["q_clean"], train["q_clean"], rtol=0, atol=0)
    full = sample_batch(model, dict(batch,mask_pre=torch.ones_like(pre)), a, "cpu")
    assert not torch.allclose(full["sia_vector"], before["sia_vector"])
    torch.testing.assert_close(full["pred"], before["pred"], rtol=0, atol=0)
    train["loss"].backward()
    assert torch.isfinite(model.sia_readout[0].weight.grad).all()
    checkpoint = tmp_path/"temporary.pth"
    try:
        torch.save(dict(pipeline_version=PIPELINE_VERSION,config=vars(a),stats=stats,model=model.state_dict(),ema=None),checkpoint)
        restored, _, _, _ = load_checkpoint(checkpoint, "cpu", cpu_threads=1)
        after = sample_batch(restored, batch, a, "cpu")
        for k in ["pred","sia","sia_vector"]:
            torch.testing.assert_close(after[k], before[k], rtol=0, atol=0)
    finally:
        checkpoint.unlink(missing_ok=True)
