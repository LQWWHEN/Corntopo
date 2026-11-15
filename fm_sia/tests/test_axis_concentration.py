import math
import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'cddpm'))
from model.axis_concentration import AxisConcentration, concentration_axis_loss
from model.polar_sia import periodic_axis_loss


def test_shifted_likelihood_matches_distribution_and_is_periodic():
    phase=torch.tensor([.2,-2.,5.],dtype=torch.float64,requires_grad=True)
    theta=torch.tensor([-.8,1.,2.],dtype=torch.float64)
    target=torch.stack((theta.cos(),theta.sin()),-1)
    kappa=torch.tensor([.01,1.,80.],dtype=torch.float64,requires_grad=True)
    actual=concentration_axis_loss(phase,target,kappa,.1)
    reference=-torch.distributions.VonMises(phase,kappa).log_prob(theta).mean()
    shift=math.log(2*math.pi)+torch.special.i0e(torch.ones((),dtype=torch.float64)).log()
    torch.testing.assert_close(actual,reference-shift)
    torch.testing.assert_close(actual,concentration_axis_loss(phase+2*math.pi,target,kappa,.1))
    ga=torch.autograd.grad(actual,(phase,kappa),retain_graph=True)
    gb=torch.autograd.grad(reference,(phase,kappa))
    for a,b in zip(ga,gb):torch.testing.assert_close(a,b)


def test_unit_concentration_matches_parent_phase_loss_and_gradient():
    phase=torch.tensor([.4,2.,-.1],requires_grad=True)
    gt=torch.tensor([[.8,.4],[.03,.01],[0.,0.]])
    a=concentration_axis_loss(phase,gt,torch.ones_like(phase),.1)
    b=periodic_axis_loss(phase,gt,.1)
    torch.testing.assert_close(a,b)
    torch.testing.assert_close(torch.autograd.grad(a,phase)[0],torch.autograd.grad(b,phase)[0])


def test_extreme_concentration_and_missing_axes_have_finite_gradients():
    phase=torch.tensor([0.,math.pi,1.],dtype=torch.float64,requires_grad=True)
    kappa=torch.tensor([1e-10,1.,1e6],dtype=torch.float64,requires_grad=True)
    gt=torch.tensor([[1.,0.]]*3,dtype=torch.float64)
    loss=concentration_axis_loss(phase,gt,kappa,.1)
    assert torch.isfinite(loss)
    for g in torch.autograd.grad(loss,(phase,kappa)):assert torch.isfinite(g).all()
    for threshold in [0.,.1]:
        loss=concentration_axis_loss(phase,gt*0,kappa,threshold)
        assert loss==0
        for g in torch.autograd.grad(loss,(phase,kappa)):assert torch.count_nonzero(g)==0


def arguments(enabled=True,extra=()):
    from train import get_args
    return get_args(['--excel_path','unused','--data_root','unused','--sia_prediction','polar_direct',
        '--lambda_sia','1','--conditions','pre_age_years','--fusion_method','film','--sample_steps','2',
        '--image_backbone','fno','--fno_modes','4','4','--fno_hidden_channels','8','--fno_layers','1',
        *(['--sia_learned_axis_concentration'] if enabled else []),*extra])


def test_parent_rng_full_task_and_checkpoint_contract(tmp_path):
    pytest.importorskip('neuralop')
    from train import build_model,PIPELINE_VERSION,saved_config_value
    from predict import load_checkpoint
    torch.set_num_threads(1)
    stats=dict(sia=dict(mean=[.2,.3],std=[.5,.6],polar_median_magnitude=.5,
                       polar_initial_phase=.4,polar_axis_training_count=2),image=dict(scale=20.))
    torch.manual_seed(91);parent=build_model(arguments(False),stats);rng=torch.get_rng_state()
    torch.manual_seed(91);child=build_model(arguments(),stats)
    assert torch.equal(rng,torch.get_rng_state())
    for k,v in parent.state_dict().items():torch.testing.assert_close(v,child.state_dict()[k],rtol=0,atol=0)
    assert sum(p.numel() for p in child.parameters())-sum(p.numel() for p in parent.parameters())==129
    pre=torch.rand(2,1,32,32);post=torch.rand_like(pre);clinical=torch.rand(2,1);t=torch.tensor([.2,.7])
    gt=torch.tensor([[.6,-.3],[0.,0.]])
    a=parent(pre,post,gt,clinical=clinical,t=t);b=child(pre,post,gt,clinical=clinical,t=t)
    for k in ['loss','loss_axis','loss_sia','loss_image','q_clean','velocity_pred']:torch.testing.assert_close(a[k],b[k])
    assert b['axis_kappa_mean']==1
    b['loss'].backward();assert child.axis_concentration_head.weight.grad.abs().sum()>0
    with torch.no_grad():child.sia_readout[-1].weight.normal_(0,.01)
    pre.requires_grad_();post.requires_grad_();clinical.requires_grad_()
    out=child(pre,post,gt,clinical=clinical,t=t)
    grads=torch.autograd.grad(out['q_clean'].sum(),(pre,clinical,post),retain_graph=True,allow_unused=True)
    assert grads[2] is None
    assert all(g is not None and g.abs().sum()>0 for g in grads[:2])
    torch.testing.assert_close(out['q_clean'].norm(dim=-1),out['sia_clean'])
    out['loss'].backward()
    assert all(torch.isfinite(p.grad).all() for p in child.parameters() if p.grad is not None)
    child.eval();before=child.sample(pre,clinical=clinical);path=tmp_path/'temporary.pth'
    try:
        torch.save(dict(pipeline_version=PIPELINE_VERSION,config=vars(arguments()),stats=stats,
                        model=child.state_dict(),ema=None),path)
        restored,_,_,_=load_checkpoint(path,'cpu',cpu_threads=1)
        for k in ['pred','sia','sia_vector']:torch.testing.assert_close(before[k],restored.sample(pre,clinical=clinical)[k],rtol=0,atol=0)
        torch.testing.assert_close(child(pre,post,gt,clinical=clinical,t=t)['loss_axis'],restored(pre,post,gt,clinical=clinical,t=t)['loss_axis'])
    finally:path.unlink(missing_ok=True)
    assert saved_config_value({},'sia_learned_axis_concentration') is False


@pytest.mark.parametrize('extra',[
    ['--sia_split_hidden'],['--sia_axis_gradient_projection'],['--sia_prediction','axial_direct'],
    ['--image_backbone','unet'],['--sia_clinical_moments'],['--sia_radial_contrast'],
    ['--sia_spatial_conditioning'],['--sia_velocity_coupling'],['--sia_rank_weight','.1']])
def test_scope(extra):
    with pytest.raises(ValueError):arguments(extra=extra)
