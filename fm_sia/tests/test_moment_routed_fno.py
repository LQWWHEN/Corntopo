"""Verify the image-only gradient intervention and the joint model's forward paths."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cddpm"))
from model.axial_sia import axial_weights
from model.moment_routed_fno import ImageConditionGradient
from predict import load_checkpoint
from test_conditional_uno import arguments, statistics
from train import PIPELINE_VERSION, build_model


def expected_gradient(gradient):
    weights = axial_weights(gradient.double()).flatten(1)
    flat = gradient.double().flatten(-2)
    return (flat - ((flat @ torch.linalg.pinv(weights)) @ weights)).reshape_as(gradient)


def test_routing_removes_only_axial_span_on_rectangular_and_degenerate_grids():
    torch.manual_seed(43)
    for height, width in ((7, 10), (1, 1)):
        feature = torch.randn(2, 3, height, width, dtype=torch.float64, requires_grad=True)
        gradient = torch.randn_like(feature)
        routed = ImageConditionGradient.apply(feature)
        torch.testing.assert_close(routed, feature, rtol=0, atol=0)
        actual = torch.autograd.grad(routed, feature, gradient)[0]
        torch.testing.assert_close(actual, expected_gradient(gradient), rtol=0, atol=1e-12)


def test_joint_gradient_paths_parent_outputs_and_reload(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(43)
    parent = build_model(arguments("fno"), statistics())
    recipe = arguments("moment_routed_fno")
    torch.manual_seed(43)
    model = build_model(recipe, statistics())
    torch.testing.assert_close(model.state_dict(), parent.state_dict(), atol=0, rtol=0)
    # Activate the normally zero-initialized SIA output weights to exercise both task paths.
    with torch.no_grad():
        parent.sia_readout[-1].weight.normal_(0, .01)
        model.load_state_dict(parent.state_dict())
    pre = torch.rand(2, 1, 32, 32)
    post = torch.rand_like(pre)
    target = torch.tensor([[.5, -.2], [-.3, .4]])
    clinical = torch.tensor([[.2], [.9]])
    t = torch.tensor([.25, .75])
    outputs, image_gradients, sia_gradients, projection_gradients = [], [], [], []
    for net in (parent, model):
        feature = []
        handle = net.condition_encoder.enc4.register_forward_hook(lambda _m, _x, value: feature.append(value))
        try:
            out = net(pre, post, target, clinical=clinical, t=t)
        finally:
            handle.remove()
        grad, projection_grad = torch.autograd.grad(
            out["loss_image"], (feature[0], net.unet.condition_projection[-1].weight), retain_graph=True)
        image_gradients.append(grad)
        projection_gradients.append(projection_grad)
        sia_gradients.append(torch.autograd.grad(out["loss_sia"] + out["loss_axis"], feature[0])[0])
        outputs.append(out)
    for key in ("velocity_pred", "q_clean", "loss_image", "loss_sia", "loss_axis"):
        torch.testing.assert_close(outputs[0][key], outputs[1][key], atol=0, rtol=0)
    torch.testing.assert_close(image_gradients[1], expected_gradient(image_gradients[0]).float(), atol=1e-9, rtol=1e-5)
    assert (image_gradients[1] - image_gradients[0]).norm() > 0 and image_gradients[1].norm() > 0
    assert sia_gradients[0].norm() > 0
    torch.testing.assert_close(sia_gradients[1], sia_gradients[0], atol=0, rtol=0)
    torch.testing.assert_close(projection_gradients[1], projection_gradients[0], atol=0, rtol=0)
    path = tmp_path / "routed.pth"
    try:
        torch.save(dict(pipeline_version=PIPELINE_VERSION, config=vars(recipe), stats=statistics(),
                        model=model.state_dict(), ema=None), path)
        loaded, _, _, _ = load_checkpoint(path, "cpu", cpu_threads=1)
        before = parent.eval().sample(pre, clinical=clinical)
        after = loaded.sample(pre, clinical=clinical)
        for key in ("pred", "sia_vector", "sia"):
            torch.testing.assert_close(after[key], before[key], atol=0, rtol=0)
    finally:
        path.unlink(missing_ok=True)
