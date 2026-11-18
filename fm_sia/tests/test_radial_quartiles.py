import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cddpm"))
from model.polar_sia import radial_quartiles, quartile_loss


def test_order_scale_and_pinball():
    m = torch.tensor([.2, 1., 0.])
    offsets = torch.tensor([[0., 0.], [-10., -10.], [50., 50.]])
    q = radial_quartiles(m, offsets)
    assert ((q[:, 0] >= 0) & (q[:, 0] <= m) & (q[:, 1] >= m)).all()
    torch.testing.assert_close(q[0], torch.tensor([.1, .4]))
    torch.testing.assert_close(radial_quartiles(2*m, offsets), 2*q)
    # GT lies below both predictions in row 1 and above both in row 2.
    actual = quartile_loss(torch.tensor([[1., 2.], [1., 2.]]), torch.tensor([0., 3.]))
    torch.testing.assert_close(actual, torch.tensor((.75 + .5 + .5 + .75)/2))


def test_parent_predictions_auxiliary_gradient_and_reload(tmp_path):
    from train import build_model, get_args, PIPELINE_VERSION, saved_config_value
    from predict import load_checkpoint
    torch.set_num_threads(1)
    flags = ['--excel_path','unused','--data_root','unused','--image_backbone','fno',
             '--sia_prediction','polar_direct','--lambda_sia','1','--fno_modes','4','4',
             '--fno_hidden_channels','8','--fno_layers','1']
    stats = dict(sia=dict(mean=[0.,0.],std=[1.,1.],polar_median_magnitude=.4,
                          polar_initial_phase=.2),image=dict(scale=20.))
    torch.manual_seed(28)
    parent = build_model(get_args(flags), stats)
    torch.manual_seed(28)
    args = get_args(flags + ['--sia_quartile_weight','.25'])
    child = build_model(args, stats)
    for name, tensor in parent.state_dict().items():
        torch.testing.assert_close(tensor, child.state_dict()[name], rtol=0, atol=0)
    pre = torch.rand(2,1,32,32)
    post = torch.rand_like(pre)
    gt = torch.tensor([[0.,0.],[.8,.9]])
    kw = dict(t=torch.tensor([.2,.7]))
    a, b = parent(pre,post,gt,**kw), child(pre,post,gt,**kw)
    for name in ['q_clean','velocity_pred','loss_axis']:
        torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)
    torch.testing.assert_close(b['loss'], a['loss']+.25*b['loss_radial_quartiles'])
    b['loss'].backward()
    assert child.quartile_head.bias.grad.abs().min() > 0
    assert saved_config_value({},'sia_quartile_weight') == 0.
    path = tmp_path/'temporary.pth'
    try:
        torch.save(dict(pipeline_version=PIPELINE_VERSION,config=vars(args),stats=stats,
                        model=child.state_dict(),ema=None),path)
        model, _, _, _ = load_checkpoint(path,'cpu',cpu_threads=1)
        torch.testing.assert_close(model(pre,post,gt,**kw)['loss'],b['loss'])
    finally:
        path.unlink(missing_ok=True)
    with pytest.raises(ValueError):
        get_args(flags + ['--sia_quartile_weight','-.1'])
