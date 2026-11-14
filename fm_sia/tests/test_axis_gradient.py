import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'cddpm'))
from model.axis_gradient import project_axis_gradient, axis_gradient_surrogate


def test_projection_conflict_positive_zero_and_per_case_independence():
    r=torch.tensor([[1.,0.],[1.,0.],[0.,0.],[0.,2.]],dtype=torch.float64)
    a=torch.tensor([[-2.,3.],[2.,3.],[4.,5.],[3.,-2.]],dtype=torch.float64)
    p,c=project_axis_gradient(r,a)
    torch.testing.assert_close(p,torch.tensor([[0.,3.],[2.,3.],[4.,5.],[3.,0.]],dtype=r.dtype))
    assert c.tolist()==[True,False,False,True]
    assert ((p*r).sum(-1)>=0).all()
    for i in range(4):torch.testing.assert_close(project_axis_gradient(r[i:i+1],a[i:i+1])[0],p[i:i+1])


def test_surrogate_preserves_value_and_output_parameters_but_changes_hidden_gradient():
    h=torch.tensor([[1.,2.]],requires_grad=True)
    w=torch.tensor([[1.,0.],[-2.,3.]],requires_grad=True)
    y=h@w.T; radial=y[:,0].sum();angular=y[:,1].sum()
    correction,log=axis_gradient_surrogate(h,radial,angular)
    assert correction.item()==0 and log['gradient_axis_conflict_fraction']==1
    gh,gw=torch.autograd.grad(radial+angular+correction,(h,w),retain_graph=True)
    _,expected_w=torch.autograd.grad(radial+angular,(h,w))
    torch.testing.assert_close(gh,torch.tensor([[1.,3.]]))
    torch.testing.assert_close(gw,expected_w)
    assert log['gradient_axis_dot_after']==0


def args(enabled=True,extra=()):
    from train import get_args
    return get_args(['--excel_path','unused','--data_root','unused','--sia_prediction','polar_direct',
        '--lambda_sia','1','--conditions','pre_age_years','--fusion_method','film','--sample_steps','2',
        '--image_backbone','fno','--fno_modes','4','4','--fno_hidden_channels','8','--fno_layers','1',
        *(['--sia_axis_gradient_projection'] if enabled else []),*extra])


def stats():
    return dict(sia=dict(mean=[.2,.3],std=[.5,.6],polar_median_magnitude=.5,
                        polar_initial_phase=.4,polar_axis_training_count=2),image=dict(scale=20.))


def test_parent_state_rng_inference_and_training_only_routing(tmp_path):
    pytest.importorskip('neuralop')
    from train import build_model,PIPELINE_VERSION,saved_config_value
    from predict import load_checkpoint
    torch.set_num_threads(1)
    torch.manual_seed(91);parent=build_model(args(False),stats());rng=torch.get_rng_state()
    torch.manual_seed(91);child=build_model(args(),stats())
    assert torch.equal(rng,torch.get_rng_state())
    for k,v in parent.state_dict().items():torch.testing.assert_close(v,child.state_dict()[k],rtol=0,atol=0)
    with torch.no_grad():
        parent.sia_readout[-1].weight.fill_(.01)
    child.load_state_dict(parent.state_dict())
    pre=torch.rand(2,1,32,32);post=torch.rand_like(pre);clinical=torch.rand(2,1);t=torch.tensor([.2,.7])
    with torch.no_grad():
        _,_,phase=parent.predict_polar(parent.condition_encoder(pre),clinical)
        gt=2*torch.stack(((phase-torch.pi/2).cos(),(phase-torch.pi/2).sin()),-1)
    a=parent(pre,post,gt,clinical=clinical,t=t);b=child(pre,post,gt,clinical=clinical,t=t)
    for key in ['loss','loss_axis','loss_sia','loss_image','q_clean','velocity_pred']:
        torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
    assert b['gradient_axis_conflict_fraction']==1
    a['loss'].backward();b['loss'].backward()
    torch.testing.assert_close(parent.sia_readout[-1].weight.grad,child.sia_readout[-1].weight.grad)
    assert not torch.allclose(parent.sia_readout[0].weight.grad,child.sia_readout[0].weight.grad)
    for pp,cp in zip(parent.unet.fno.parameters(),child.unet.fno.parameters()):
        if pp.grad is None:assert cp.grad is None
        else:torch.testing.assert_close(pp.grad,cp.grad)
    child.eval();parent.eval()
    with torch.no_grad():
        no_grad=child(pre,post,gt,clinical=clinical,t=t)
        assert 'gradient_axis_conflict_fraction' not in no_grad
    before=child.sample(pre,clinical=clinical);reference=parent.sample(pre,clinical=clinical)
    for k in ['pred','sia','sia_vector']:torch.testing.assert_close(before[k],reference[k],rtol=0,atol=0)
    path=tmp_path/'temporary.pth'
    try:
        torch.save(dict(pipeline_version=PIPELINE_VERSION,config=vars(args()),stats=stats(),
                        model=child.state_dict(),ema=None),path)
        restored,_,_,_=load_checkpoint(path,'cpu',cpu_threads=1)
        for k in ['pred','sia','sia_vector']:
            torch.testing.assert_close(before[k],restored.sample(pre,clinical=clinical)[k],rtol=0,atol=0)
    finally:path.unlink(missing_ok=True)
    assert saved_config_value({},'sia_axis_gradient_projection') is False


def test_zero_axis_targets_and_initial_zero_readout_have_finite_gradients():
    pytest.importorskip('neuralop')
    from train import build_model
    torch.set_num_threads(1);m=build_model(args(),stats())
    pre=torch.rand(2,1,32,32);out=m(pre,torch.rand_like(pre),torch.zeros(2,2),clinical=torch.zeros(2,1))
    out['loss'].backward()
    assert out['loss_axis']==0 and out['gradient_axis_conflict_fraction']==0
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


@pytest.mark.parametrize('extra',[
    ['--sia_prediction','axial_direct'],['--image_backbone','unet'],['--sia_clinical_moments'],
    ['--sia_radial_contrast'],['--sia_spatial_conditioning'],['--sia_velocity_coupling'],
    ['--image_local_residual'],['--sia_rank_weight','.1'],['--image_gradient_weight','.25'],
    ['--augmentation','d4']])
def test_scoped_configuration(extra):
    from train import get_args
    with pytest.raises(ValueError):args(extra=extra)
