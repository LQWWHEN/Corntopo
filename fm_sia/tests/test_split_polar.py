import sys
from pathlib import Path
import pytest
import torch
from torch import nn
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'cddpm'))
from model.split_polar import SplitPolarReadout


def test_split_preserves_function_rng_and_separates_task_gradients():
    parent=nn.Sequential(nn.Linear(9,128),nn.SiLU(),nn.Linear(128,2)).double()
    rng=torch.get_rng_state();child=SplitPolarReadout(parent)
    assert torch.equal(rng,torch.get_rng_state())
    x=torch.randn(4,9,dtype=torch.float64,requires_grad=True)
    torch.testing.assert_close(parent(x),child(x))
    assert child.magnitude.hidden.weight.data_ptr()!=child.phase.hidden.weight.data_ptr()
    assert sum(p.numel() for p in child.parameters())-sum(p.numel() for p in parent.parameters())==10*128
    for own,other,index in [(child.magnitude,child.phase,0),(child.phase,child.magnitude,1)]:
        child.zero_grad();x.grad=None
        child(x)[:,index].sum().backward()
        assert own.hidden.weight.grad.abs().sum()>0 and x.grad.abs().sum()>0
        assert all(p.grad is None or p.grad.count_nonzero()==0 for p in other.parameters())


def args(enabled=True,extra=()):
    from train import get_args
    return get_args(['--excel_path','unused','--data_root','unused','--sia_prediction','polar_direct',
        '--lambda_sia','1','--conditions','pre_age_years','--fusion_method','film','--sample_steps','2',
        '--image_backbone','fno','--fno_modes','4','4','--fno_hidden_channels','8','--fno_layers','1',
        *(['--sia_split_hidden'] if enabled else []),*extra])


def test_full_task_parent_initialization_no_postoperative_input_and_reload(tmp_path):
    pytest.importorskip('neuralop')
    from train import build_model,PIPELINE_VERSION,saved_config_value
    from predict import load_checkpoint
    torch.set_num_threads(1)
    stats=dict(sia=dict(mean=[.2,.3],std=[.5,.6],polar_median_magnitude=.5,
                       polar_initial_phase=.4,polar_axis_training_count=2),image=dict(scale=20.))
    torch.manual_seed(91);parent=build_model(args(False),stats);rng=torch.get_rng_state()
    torch.manual_seed(91);child=build_model(args(),stats)
    assert torch.equal(rng,torch.get_rng_state())
    for k,v in parent.state_dict().items():
        if not k.startswith('sia_readout.'):torch.testing.assert_close(v,child.state_dict()[k],rtol=0,atol=0)
    assert sum(p.numel() for p in child.parameters())-sum(p.numel() for p in parent.parameters())==213120
    pre=torch.rand(2,1,32,32);post=torch.rand_like(pre);clinical=torch.rand(2,1);t=torch.tensor([.2,.7])
    gt=torch.tensor([[.6,-.3],[0.,0.]])
    a=parent(pre,post,gt,clinical=clinical,t=t);b=child(pre,post,gt,clinical=clinical,t=t)
    for k in ['loss','loss_axis','loss_sia','loss_image','q_clean','velocity_pred']:
        torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
    with torch.no_grad():
        child.sia_readout.magnitude.weight.normal_(0,.01)
        child.sia_readout.phase.weight.normal_(0,.01)
    pre.requires_grad_();post.requires_grad_();clinical.requires_grad_()
    out=child(pre,post,gt,clinical=clinical,t=t)
    grads=torch.autograd.grad(out['q_clean'].sum(),(pre,clinical,post),retain_graph=True,allow_unused=True)
    assert grads[2] is None
    assert all(g is not None and g.abs().sum()>0 for g in grads[:2])
    torch.testing.assert_close(out['q_clean'].norm(dim=-1),out['sia_clean'])
    out['loss'].backward()
    assert all(torch.isfinite(p.grad).all() for p in child.parameters() if p.grad is not None)
    child.eval();before=child.sample(pre,clinical=clinical)
    path=tmp_path/'temporary.pth'
    try:
        torch.save(dict(pipeline_version=PIPELINE_VERSION,config=vars(args()),stats=stats,
                        model=child.state_dict(),ema=None),path)
        restored,_,_,_=load_checkpoint(path,'cpu',cpu_threads=1)
        for k in ['pred','sia','sia_vector']:
            torch.testing.assert_close(before[k],restored.sample(pre,clinical=clinical)[k],rtol=0,atol=0)
    finally:path.unlink(missing_ok=True)
    assert saved_config_value({},'sia_split_hidden') is False


@pytest.mark.parametrize('extra',[
    ['--sia_axis_gradient_projection'],['--sia_prediction','axial_direct'],['--image_backbone','unet'],
    ['--sia_clinical_moments'],['--sia_radial_contrast'],['--sia_spatial_conditioning'],
    ['--sia_velocity_coupling'],['--image_local_residual'],['--sia_rank_weight','.1'],
    ['--image_gradient_weight','.25'],['--augmentation','d4']])
def test_scope_rejects_unplanned_combinations(extra):
    with pytest.raises(ValueError):args(extra=extra)
