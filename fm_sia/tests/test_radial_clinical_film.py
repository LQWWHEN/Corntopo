import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cddpm"))
pytest.importorskip("neuralop")
from model.unet import ClinicalFusion, radial_contrast
from predict import load_checkpoint
from train import PIPELINE_VERSION, build_model, forward_batch, sample_batch
from test_zero_clinical_attention import STATS, arguments


def test_clinical_contrast_is_radial_and_zero_mean():
    for shape in ((17, 17), (18, 20), (1, 1)):
        image = torch.zeros(1, 1, *shape)
        basis = radial_contrast(image)
        torch.testing.assert_close(basis.mean(), torch.tensor(0.), atol=1e-6, rtol=0)
        torch.testing.assert_close(basis.flip((0, 1)), basis, atol=1e-6, rtol=0)
        if shape == (17, 17):
            torch.testing.assert_close(basis.T, basis)
            assert basis[8, 8] > 0 and basis[8, 14] < 0 and basis[0, 0] == 0

    layer = ClinicalFusion(1, 1, "radial_film")
    with torch.no_grad():
        layer.radial_affine.weight.copy_(torch.tensor([[2.], [1.]]))
    image = torch.ones(2, 1, 17, 17)
    out = layer(image, torch.tensor([[0.], [1.]]), None)
    torch.testing.assert_close(out[0], image[0])
    torch.testing.assert_close((out[1] - image[1])[0], 3 * radial_contrast(image))


def test_matched_parent_training_and_preoperative_reload(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(35)
    parent = build_model(arguments("film"), STATS).eval()
    rng = torch.get_rng_state()
    torch.manual_seed(35)
    args = arguments("radial_film")
    model = build_model(args, STATS).eval()
    assert torch.equal(rng, torch.get_rng_state())
    candidate_state = model.state_dict()
    for name, value in parent.state_dict().items():
        torch.testing.assert_close(candidate_state[name], value, atol=0, rtol=0)
    pre = torch.rand(2, 1, 32, 32)
    batch = dict(input=pre, target=torch.rand_like(pre), condition=torch.randn(2, 4),
                 mask_pre=torch.ones_like(pre), mask_post=torch.ones_like(pre),
                 sia_vector=torch.tensor([[.3, -.1], [.1, .5]]), label=[1, 2])
    for m in (parent, model):
        torch.manual_seed(35)
        losses = forward_batch(m, batch, "cpu")
        if m is parent:
            reference = {k: losses[k].detach() for k in ("loss_image", "loss_sia", "loss_axis")}
        else:
            for k, expected in reference.items():
                torch.testing.assert_close(losses[k], expected, rtol=0, atol=0)
    losses["loss"].backward()
    radial_weight = model.unet.clinical_fusion.radial_affine.weight
    for grad in radial_weight.grad.chunk(2):
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    with torch.no_grad():
        radial_weight.add_(radial_weight.grad, alpha=-.03)
    for key in ("target", "mask_post", "sia_vector"):
        del batch[key]
    calls = []
    hook = model.unet.clinical_fusion.register_forward_hook(lambda *unused: calls.append(1))
    measured = sample_batch(model, batch, args, "cpu")
    hook.remove()
    assert len(calls) == 1
    parent_out = sample_batch(parent, batch, arguments("film"), "cpu")
    torch.testing.assert_close(measured["sia_vector"], parent_out["sia_vector"], atol=0, rtol=0)
    path = tmp_path / "reload.pth"
    try:
        torch.save(dict(pipeline_version=PIPELINE_VERSION, config=vars(args), stats=STATS,
                        model=model.state_dict(), ema=None), path)
        restored, _, _, _ = load_checkpoint(path, "cpu", cpu_threads=1)
        after = sample_batch(restored, batch, args, "cpu")
        for key in ("pred", "sia_vector", "sia"):
            torch.testing.assert_close(after[key], measured[key], atol=0, rtol=0)
    finally:
        path.unlink(missing_ok=True)
