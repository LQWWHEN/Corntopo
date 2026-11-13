import sys
from pathlib import Path
import numpy as np
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'cddpm'))
from model.axial_sia import axial_moments
from metrics import axis_metrics, axis_details
from train import get_args, build_model, fit_sia_stats
from task_balancing import build_task_balancer
import pandas as pd

torch.set_num_threads(1)


def test_moments_retain_orientation_lost_by_global_average():
    feature = torch.randn(2, 3, 16, 16)
    a = axial_moments(feature).reshape(2, 3, 3)
    b = axial_moments(feature.rot90(1, (-2, -1))).reshape(2, 3, 3)
    torch.testing.assert_close(a[:, 0], b[:, 0])
    torch.testing.assert_close(a[:, 1:], -b[:, 1:])
    assert a[:, 1:].abs().sum() > 0
    c = axial_moments(feature.flip(-1)).reshape(2, 3, 3)
    torch.testing.assert_close(a[:, :2], c[:, :2])
    torch.testing.assert_close(a[:, 2], -c[:, 2])
    constant = axial_moments(torch.ones_like(feature)).reshape(2, 3, 3)
    torch.testing.assert_close(constant[:, 1:], torch.zeros_like(constant[:, 1:]), atol=1e-7, rtol=0)


def vector(angle, magnitude=1.):
    return magnitude*np.array([np.cos(np.deg2rad(2*angle)), np.sin(np.deg2rad(2*angle))])


def test_periodic_axis_zero_handling_and_threshold_coverage():
    qp = np.array([vector(1), [0., 0.], vector(50), vector(30)])
    qt = np.array([vector(179), vector(20), [0., 0.], vector(40, .05)])
    details = axis_details(qp, qt)
    assert details['error'][0] == pytest.approx(2)
    assert np.isnan(details['pred'][1]) and np.isnan(details['true'][2])
    result = axis_metrics(qp, qt)
    assert result['Axis_ge_0p1D_GT_count'] == 2
    assert result['Axis_ge_0p1D_coverage'] == .5
    assert result['Axis_ge_0p1D_undefined_pred_count'] == 1
    assert result['Axis_ge_0p1D_MAE_deg'] == pytest.approx(46)
    assert result['Axis_nonzero_MAE_deg'] == pytest.approx(34)
    assert axis_metrics(np.zeros((2,2)),np.zeros((2,2)))['Axis_nonzero_MAE_deg'] is None


def test_axial_graph_training_stats_and_loss_contract(tmp_path):
    args = get_args(['--excel_path','unused','--data_root','unused','--sia_prediction','axial_direct',
                     '--lambda_sia','1','--conditions','pre_age_years','--fusion_method','film','--sample_steps','2'])
    df = pd.DataFrame(dict(label_index=[1,2,3],delta_front_X=[.2,.4,999],delta_front_Y=[.4,.6,-999]))
    stats = {'sia':fit_sia_stats(args,df,[1,2])}
    assert stats['sia']['mean'] == pytest.approx([.3,.5])
    df.loc[2,['delta_front_X','delta_front_Y']]=0
    assert stats['sia'] == fit_sia_stats(args,df,[1,2])
    torch.manual_seed(10)
    model = build_model(args,stats)
    assert model.unet.sia_interaction is None and not args.save_checkpoints
    pre = torch.rand(1,1,32,32,requires_grad=True)
    post = torch.rand_like(pre,requires_grad=True)
    clinical = torch.tensor([[.2]],requires_grad=True)
    target = torch.tensor([[.6,-.3]])
    initial = model(pre,post,target,clinical=clinical)
    torch.testing.assert_close(initial['q_clean'],torch.tensor([[.3,.5]]))
    with torch.no_grad(): model.sia_readout[-1].weight.normal_(0,.01)
    a = model(pre,post,target,clinical=clinical,t=torch.tensor([.1]))
    b = model(pre,post+10,target,clinical=clinical,t=torch.tensor([.9]),noise=torch.randn(1,2))
    torch.testing.assert_close(a['q_clean'],b['q_clean'])
    gradients = torch.autograd.grad(a['q_clean'].sum(),(pre,clinical,post),retain_graph=True,allow_unused=True)
    assert gradients[2] is None
    assert all(g is not None and torch.isfinite(g).all() and g.abs().sum()>0 for g in gradients[:2])
    total,_ = build_task_balancer(args,model,'cpu').combine(a)
    torch.testing.assert_close(total,a['loss_image']+a['loss_vector']+a['loss_sia'])
    total.backward()
    for parameter in (model.condition_encoder.in_conv.weight,model.sia_readout[-1].weight,model.unet.out.weight):
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum()>0
    model.eval()
    first = model.sample(pre.detach(),clinical=clinical.detach(),noise=torch.zeros(1,2))
    second = model.sample(pre.detach(),clinical=clinical.detach(),noise=torch.ones(1,2)*100)
    torch.testing.assert_close(first['pred'],second['pred'])
    torch.testing.assert_close(first['sia_vector'],second['sia_vector'])
    torch.testing.assert_close(first['sia'],first['sia_vector'].norm(dim=-1))
    from predict import predict_case
    path = tmp_path/'pre.npy'
    np.save(path, pre.detach().numpy()[0,0])
    args.image_size = 32
    result,_ = predict_case(model,args,dict(**stats,image={'minimum':0.,'scale':1.},clinical=None),
                            'cpu',path,[.2])
    assert 0 <= result['pred_axis_deg'] < 180
    assert result['pred_SIA_D'] == pytest.approx(first['sia'].item())
    with pytest.raises(ValueError,match='Axial direct'):
        get_args(['--excel_path','x','--data_root','x','--sia_prediction','axial_direct','--sia_mode','scalar'])
