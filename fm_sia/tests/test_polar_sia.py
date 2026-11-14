import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'cddpm'))
from model.polar_sia import decode_polar,periodic_axis_loss
from train import get_args,fit_sia_stats,build_model,PIPELINE_VERSION
from task_balancing import build_task_balancer
from predict import load_checkpoint

torch.set_num_threads(1)


def arguments(extra=()):
    return get_args(['--excel_path','unused','--data_root','unused','--sia_prediction','polar_direct',
        '--lambda_sia','1','--conditions','pre_age_years','--fusion_method','film','--sample_steps','2',*extra])


def statistics(args):
    df=pd.DataFrame(dict(label_index=[1,2,3],delta_front_X=[1.,0.,999.],delta_front_Y=[0.,1.,-999.]))
    return df,{'sia':fit_sia_stats(args,df,[1,2])}


def test_polar_periodicity_consistency_and_separate_output_gradients():
    raw=torch.tensor([[.3,math.radians(2)],[.3,math.radians(362)],[-20.,0.]],dtype=torch.float64,requires_grad=True)
    q,magnitude,phase=decode_polar(raw)
    torch.testing.assert_close(q.norm(dim=-1),magnitude)
    torch.testing.assert_close(q[0],q[1])
    target=torch.tensor([[math.cos(math.radians(358)),math.sin(math.radians(358))],
                         [0.,0.],[.05,0.]],dtype=torch.float64)
    axis=periodic_axis_loss(phase,target,.1)
    assert axis.item()==pytest.approx(1-math.cos(math.radians(4)))
    grad,=torch.autograd.grad(axis,raw,retain_graph=True)
    assert torch.count_nonzero(grad[:,0])==0 and torch.count_nonzero(grad[1:])==0
    radial=F.l1_loss(magnitude,torch.tensor([2.,2.,2.],dtype=torch.float64))
    grad,=torch.autograd.grad(radial,raw)
    assert torch.count_nonzero(grad[:,1])==0 and torch.isfinite(grad).all()


def test_undefined_and_near_zero_axes_have_finite_zero_loss():
    phase=torch.tensor([0.,100.],requires_grad=True)
    loss=periodic_axis_loss(phase,torch.tensor([[0.,0.],[.05,0.]]),.1)
    assert loss.item()==0
    loss.backward();torch.testing.assert_close(phase.grad,torch.zeros_like(phase))
    assert periodic_axis_loss(phase,torch.zeros(2,2),0).item()==0


def test_polar_statistics_are_training_only_and_handle_zero_targets():
    args=arguments();df,stats=statistics(args)
    assert stats['sia']['polar_median_magnitude']==1
    assert stats['sia']['polar_initial_phase']==pytest.approx(math.pi/4)
    assert stats['sia']['polar_axis_training_count']==2
    df.loc[2,['delta_front_X','delta_front_Y']]=[-99999,99999]
    assert fit_sia_stats(args,df,[1,2])==stats['sia']
    df.loc[:1,['delta_front_X','delta_front_Y']]=0
    empty=fit_sia_stats(args,df,[1,2])
    assert empty['polar_axis_training_count']==0 and empty['polar_initial_phase']==0
    assert empty['polar_median_magnitude']==0


def test_parent_capacity_image_initialization_and_full_task_gradient_contract():
    args=arguments();_,stats=statistics(args)
    torch.manual_seed(12);model=build_model(args,stats)
    parent_args=arguments(['--sia_prediction','axial_direct'])
    torch.manual_seed(12);parent=build_model(parent_args,stats)
    assert sum(p.numel() for p in model.parameters())==sum(p.numel() for p in parent.parameters())
    for key,value in model.state_dict().items():
        if key!='sia_readout.2.bias':torch.testing.assert_close(value,parent.state_dict()[key])
    pre=torch.rand(1,1,32,32,requires_grad=True);post=torch.rand_like(pre,requires_grad=True)
    clinical=torch.tensor([[.2]],requires_grad=True);gt=torch.tensor([[.6,-.3]])
    initial=model(pre,post,gt,clinical=clinical)
    torch.testing.assert_close(initial['sia_clean'],torch.ones(1))
    assert build_task_balancer(args,model,'cpu') is None
    with torch.no_grad():model.sia_readout[-1].weight.normal_(0,.01)
    a=model(pre,post,gt,clinical=clinical,t=torch.tensor([.1]))
    b=model(pre,post+10,gt,clinical=clinical,t=torch.tensor([.9]),noise=torch.ones(1,2)*100)
    torch.testing.assert_close(a['q_clean'],b['q_clean'])
    torch.testing.assert_close(a['q_clean'].norm(dim=-1),a['sia_clean'])
    torch.testing.assert_close(a['loss'],a['loss_image']+a['loss_sia']+a['loss_axis'])
    gradients=torch.autograd.grad(a['q_clean'].sum(),(pre,clinical,post),retain_graph=True,allow_unused=True)
    assert gradients[2] is None
    assert all(g is not None and torch.isfinite(g).all() and g.abs().sum()>0 for g in gradients[:2])
    a['loss'].backward()
    for parameter in (model.condition_encoder.in_conv.weight,model.sia_readout[-1].weight,model.unet.out.weight):
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum()>0
    model.eval()
    output=model.sample(pre.detach(),clinical=clinical.detach())
    torch.testing.assert_close(output['sia_vector'].norm(dim=-1),output['sia'])


def test_polar_fno_checkpoint_roundtrip(tmp_path):
    pytest.importorskip('neuralop')
    args=arguments(['--image_backbone','fno','--fno_modes','4','4','--fno_hidden_channels','8','--fno_layers','1'])
    _,stats=statistics(args);model=build_model(args,stats).eval()
    pre=torch.rand(1,1,32,32);clinical=torch.zeros(1,1)
    before=model.sample(pre,clinical=clinical)
    path=tmp_path/'temporary.pth'
    torch.save(dict(pipeline_version=PIPELINE_VERSION,config=vars(args),stats=stats,
                    model=model.state_dict(),ema=None),path)
    restored,_,_,_=load_checkpoint(path,'cpu')
    after=restored.sample(pre,clinical=clinical)
    for key in ('pred','sia','sia_vector'):torch.testing.assert_close(before[key],after[key])
    path.unlink()


@pytest.mark.parametrize('extra',[
    ['--sia_mode','scalar'],['--loss_balance','gradnorm'],['--lambda_sia','0'],
    ['--eye_frame','right'],['--lambda_axis','0'],['--axis_loss_min_magnitude','-1'],
])
def test_polar_cli_rejects_incompatible_objectives(extra):
    with pytest.raises(ValueError):arguments(extra)
