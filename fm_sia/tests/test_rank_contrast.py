import math
import sys
from pathlib import Path
import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'cddpm'))
from model.rank_contrast import axial_rank_contrast
from train import get_args,build_model,saved_config_value,PIPELINE_VERSION
from predict import load_checkpoint


def arguments(weight=.1,extra=()):
    return get_args(['--excel_path','unused','--data_root','unused','--sia_prediction','polar_direct',
        '--conditions','pre_age_years','--fusion_method','film','--lambda_sia','1','--sample_steps','2',
        '--image_backbone','fno','--fno_modes','4','4','--fno_hidden_channels','8','--fno_layers','1',
        '--sia_rank_weight',str(weight),*extra])


def stats():
    return dict(sia=dict(mean=[.2,.3],std=[.5,.6],polar_median_magnitude=.5,
                        polar_initial_phase=.4,polar_axis_training_count=2),image=dict(scale=20.))


def test_loss_and_gradient_against_explicit_rank_sets():
    torch.set_num_threads(1)
    feature=torch.tensor([[1.,0.],[.9,.1],[0.,1.],[-1.,0.]],dtype=torch.float64,requires_grad=True)
    target=torch.tensor([[0.,0.],[1.,0.],[0.,1.],[2.,0.]],dtype=torch.float64,requires_grad=True)
    unit=F.normalize(feature,dim=-1);terms=[]
    for i in range(4):
        for j in range(4):
            if i==j:continue
            d=(target[i]-target[j]).square().sum().item()
            choices=[k for k in range(4) if k!=i and (target[i]-target[k]).square().sum().item()>=d-1e-6]
            terms.append(torch.logsumexp(torch.stack([unit[i].dot(unit[k])/.2 for k in choices]),0)-unit[i].dot(unit[j])/.2)
    expected=torch.stack(terms).mean();actual=axial_rank_contrast(feature,target)
    torch.testing.assert_close(actual,expected)
    gradients=torch.autograd.grad(actual,(feature,target),retain_graph=True,allow_unused=True)
    assert gradients[1] is None
    torch.testing.assert_close(gradients[0],torch.autograd.grad(expected,feature)[0])
    permutation=torch.tensor([3,1,0,2])
    torch.testing.assert_close(actual,axial_rank_contrast(feature[permutation],target[permutation]))


def test_double_angle_wrap_reflection_and_common_rotation():
    amplitude=torch.tensor([.05,.5,.5,1.,2.],dtype=torch.float64)
    axis=torch.tensor([0.,1.,179.,90.,40.],dtype=torch.float64)
    phase=axis*math.pi/90
    q=amplitude[:,None]*torch.stack([phase.cos(),phase.sin()],1)
    squared=(q[:,None]-q[None]).square().sum(-1)
    assert squared[1,2]<squared[1,3]  # 1 and179 degrees are neighbouring axes.
    torch.testing.assert_close(squared,(amplitude[:,None]-amplitude[None]).square()+
        2*amplitude[:,None]*amplitude[None]*(1-torch.cos(phase[:,None]-phase[None])))
    torch.manual_seed(5);feature=torch.randn(5,8,dtype=torch.float64)
    theta=.7;rotation=q.new_tensor([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]])
    expected=axial_rank_contrast(feature,q)
    torch.testing.assert_close(expected,axial_rank_contrast(feature,q@rotation))
    torch.testing.assert_close(expected,axial_rank_contrast(feature,q*q.new_tensor([1,-1])))


def test_continuous_order_has_lower_loss_than_scrambled_order():
    x=torch.arange(4,dtype=torch.float64)
    feature=torch.stack([torch.cos(.3*x),torch.sin(.3*x)],1)
    target=torch.stack([x,torch.zeros_like(x)],1)
    assert axial_rank_contrast(feature,target)<axial_rank_contrast(feature[[0,2,1,3]],target)


@pytest.mark.parametrize('batch',[1,2,5])
def test_small_batch_zero_features_and_tied_targets_are_finite(batch):
    feature=torch.zeros(batch,8,requires_grad=True);target=torch.zeros(batch,2)
    loss=axial_rank_contrast(feature,target);loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(feature.grad).all()
    if batch<3:assert loss.item()==0 and torch.count_nonzero(feature.grad)==0
    else:assert loss.item()==pytest.approx(math.log(batch-1))


def test_unchanged_parent_capacity_rng_predictions_and_auxiliary_gradient():
    pytest.importorskip('neuralop');torch.set_num_threads(1)
    torch.manual_seed(17);parent=build_model(arguments(0),stats()).eval();rng=torch.get_rng_state()
    torch.manual_seed(17);model=build_model(arguments(),stats()).eval()
    assert torch.equal(rng,torch.get_rng_state())
    for name,value in parent.state_dict().items():torch.testing.assert_close(value,model.state_dict()[name],rtol=0,atol=0)
    assert sum(p.numel() for p in parent.parameters())==sum(p.numel() for p in model.parameters())
    with torch.no_grad():model.sia_readout[-1].weight.normal_(0,.01)
    parent.load_state_dict(model.state_dict())
    pre=torch.rand(4,1,32,32);post=torch.rand_like(pre);clinical=torch.randn(4,1)
    target=torch.tensor([[.1,.2],[1.,.1],[-.3,.5],[.4,-.6]]);t=torch.ones(4)*.2
    a=model(pre,post,target,clinical=clinical,t=t);b=parent(pre,post,target,clinical=clinical,t=t)
    for key in ('q_clean','velocity_pred','loss_image','loss_sia','loss_axis'):
        torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
    torch.testing.assert_close(a['loss'],b['loss']+.1*a['loss_rank_contrast'])
    assert 'loss_rank_contrast' not in b
    gradients=torch.autograd.grad(a['loss_rank_contrast'],(model.condition_encoder.in_conv.weight,model.sia_readout[0].weight))
    assert all(torch.isfinite(g).all() and g.abs().sum()>0 for g in gradients)
    changed=model(pre,post,target[[2,1,0,3]],clinical=clinical,t=t)
    torch.testing.assert_close(a['q_clean'],changed['q_clean'],rtol=0,atol=0)
    assert saved_config_value({},'sia_rank_weight')==0 and saved_config_value({},'sia_rank_temperature')==.2


def test_rank_configuration_saved_reload_preserves_inference(tmp_path):
    pytest.importorskip('neuralop');torch.set_num_threads(1)
    args=arguments();model=build_model(args,stats()).eval()
    pre=torch.rand(1,1,32,32);clinical=torch.zeros(1,1);before=model.sample(pre,clinical=clinical)
    path=tmp_path/'temporary.pth';torch.save(dict(pipeline_version=PIPELINE_VERSION,config=vars(args),
        stats=stats(),model=model.state_dict(),ema=None),path)
    restored,_,_,_=load_checkpoint(path,'cpu',cpu_threads=1)
    assert restored.rank_weight==.1 and restored.rank_temperature==.2
    after=restored.sample(pre,clinical=clinical)
    for key in ('pred','sia','sia_vector'):torch.testing.assert_close(before[key],after[key],rtol=0,atol=0)
    path.unlink()


@pytest.mark.parametrize('coupled',[False,True])
def test_legacy_polar_checkpoint_without_rank_flags(tmp_path,coupled):
    pytest.importorskip('neuralop');torch.set_num_threads(1)
    args=arguments(0,extra=['--sia_velocity_coupling'] if coupled else [])
    model=build_model(args,stats()).eval();pre=torch.rand(1,1,32,32);clinical=torch.zeros(1,1)
    before=model.sample(pre,clinical=clinical)
    config=vars(args).copy();config.pop('sia_rank_weight');config.pop('sia_rank_temperature')
    if not coupled:
        config.pop('sia_velocity_coupling');config.pop('endpoint_image_weight')
    path=tmp_path/'legacy.pth'
    torch.save(dict(pipeline_version=PIPELINE_VERSION,config=config,stats=stats(),model=model.state_dict(),ema=None),path)
    restored,_,_,_=load_checkpoint(path,'cpu',cpu_threads=1)
    after=restored.sample(pre,clinical=clinical)
    for key in ('pred','sia','sia_vector'):torch.testing.assert_close(before[key],after[key],rtol=0,atol=0)
    path.unlink()


@pytest.mark.parametrize('extra',[
    ['--sia_rank_weight','-1'],['--sia_rank_weight','nan'],['--sia_rank_temperature','0'],
    ['--sia_velocity_coupling'],['--sia_prediction','axial_direct'],
])
def test_rank_scope_and_hyperparameters(extra):
    with pytest.raises(ValueError):arguments(extra=extra)
