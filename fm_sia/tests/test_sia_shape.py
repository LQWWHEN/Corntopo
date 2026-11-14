import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'cddpm'))
from model.sia_shape import quadrupole_field
from train import get_args, build_model, saved_config_value, PIPELINE_VERSION, forward_batch, sample_batch
from predict import load_checkpoint


def arguments(enabled=True, extra=()):
    return get_args(['--excel_path','unused','--data_root','unused',
        '--sia_prediction','polar_direct','--conditions','pre_age_years','--fusion_method','film',
        '--lambda_sia','1','--sample_steps','3','--image_backbone','fno',
        '--fno_modes','4','4','--fno_hidden_channels','8','--fno_layers','1',
        *(['--sia_spatial_conditioning'] if enabled else []),*extra])


def statistics():
    return dict(sia=dict(mean=[.2,.3],std=[.5,.6],polar_median_magnitude=.5,
                         polar_initial_phase=.4,polar_axis_training_count=2),image=dict(scale=20.))


def test_quadrupole_physical_axes_scaling_and_mask():
    q=torch.tensor([[1.,0.],[0.,1.],[0.,0.]],dtype=torch.float64,requires_grad=True)
    field=quadrupole_field(q,5,5)
    # Pixel centres are0.4 apart. Rows increase physical y, matching origin=lower.
    assert field[0,0,2,3].item()==pytest.approx(.16)
    assert field[0,0,3,2].item()==pytest.approx(-.16)
    assert field[1,0,1,3].item()==pytest.approx(-.32)
    assert field[1,0,3,3].item()==pytest.approx(.32)
    assert torch.count_nonzero(field[2])==0 and torch.count_nonzero(field[:,:,2,2])==0
    torch.testing.assert_close(quadrupole_field(3*q,5,5),3*field)
    mask=torch.ones_like(field);mask[:,:,:,2:]=0
    masked=quadrupole_field(q,5,5,mask)
    torch.testing.assert_close(masked,field*mask)
    assert torch.count_nonzero(masked[:,:,:,2:])==0
    masked.square().sum().backward();assert torch.isfinite(q.grad).all()


def test_quadrupole_double_angle_rotation_and_reflection():
    q=torch.tensor([[.6,.8],[-.3,.2]],dtype=torch.float64)
    field=quadrupole_field(q,32,32)
    torch.testing.assert_close(quadrupole_field(-q,32,32),field.rot90(1,(-2,-1)),atol=1e-15,rtol=1e-14)
    reflected=q*torch.tensor([1.,-1.])
    torch.testing.assert_close(quadrupole_field(reflected,32,32),field.flip(-1),atol=1e-15,rtol=1e-14)
    #180 degrees in axis is360 degrees in phase and leaves the spatial code unchanged.
    phase=torch.atan2(q[:,1],q[:,0])+2*torch.pi
    periodic=q.norm(dim=1)[:,None]*torch.stack((phase.cos(),phase.sin()),dim=1)
    torch.testing.assert_close(quadrupole_field(periodic,32,32),field,atol=1e-15,rtol=1e-14)


def test_zero_adapter_preserves_parent_rng_weights_losses_and_predictions():
    pytest.importorskip('neuralop');torch.set_num_threads(1)
    torch.manual_seed(18);parent=build_model(arguments(False),statistics()).eval();rng=torch.get_rng_state()
    torch.manual_seed(18);model=build_model(arguments(),statistics()).eval()
    assert torch.equal(rng,torch.get_rng_state())
    for k,v in parent.state_dict().items():torch.testing.assert_close(v,model.state_dict()[k],rtol=0,atol=0)
    dofs=lambda m:sum(p.numel()*(2 if p.is_complex() else 1) for p in m.parameters())
    assert dofs(model)-dofs(parent)==64 and torch.count_nonzero(model.sia_shape_gain)==0
    pre=torch.rand(1,1,32,32);post=torch.rand_like(pre);clinical=torch.zeros(1,1);gt=torch.tensor([[.6,-.3]])
    kw=dict(clinical=clinical,t=torch.tensor([.3]))
    a=parent(pre,post,gt,**kw);b=model(pre,post,gt,pre_mask=torch.ones_like(pre),**kw)
    for k in ['loss','loss_image','loss_sia','loss_axis','q_clean','velocity_pred']:
        torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
    a=parent.sample(pre,clinical=clinical);b=model.sample(pre,clinical=clinical)
    for k in ['pred','sia','sia_vector']:torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
    assert saved_config_value({},'sia_spatial_conditioning') is False


def test_condition_never_uses_gt_and_image_loss_reaches_sia_head():
    pytest.importorskip('neuralop');torch.set_num_threads(1)
    model=build_model(arguments(),statistics()).eval()
    with torch.no_grad():
        model.sia_shape_gain.fill_(.1);model.sia_readout[-1].weight.normal_(0,.01)
    pre=torch.rand(1,1,32,32,requires_grad=True);post=torch.rand_like(pre,requires_grad=True)
    clinical=torch.tensor([[.2]],requires_grad=True);gt=torch.tensor([[.6,-.3]])
    pm=(torch.rand_like(pre)>.2).float();kw=dict(clinical=clinical,pre_mask=pm,t=torch.tensor([.2]))
    a=model(pre,post,gt,**kw);different_gt=model(pre,post,gt+10,**kw)
    b=model(pre,post+5,gt+10,clinical=clinical,pre_mask=pm,mask=1-pm,t=torch.tensor([.8]),noise=gt*9)
    for k in ['q_clean','sia_condition']:
        torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
    torch.testing.assert_close(a['velocity_pred'],different_gt['velocity_pred'],rtol=0,atol=0)
    assert torch.count_nonzero(a['sia_condition'][pm==0])==0
    grads=torch.autograd.grad(a['q_clean'].sum(),(pre,clinical,post),retain_graph=True,allow_unused=True)
    assert grads[2] is None and all(v is not None and v.abs().sum()>0 for v in grads[:2])
    gq,gw=torch.autograd.grad(a['loss_image'],(a['q_clean'],model.sia_readout[-1].weight),retain_graph=True)
    assert all(torch.isfinite(v).all() and v.abs().sum()>0 for v in (gq,gw))
    torch.testing.assert_close(a['loss'],a['loss_image']+a['loss_sia']+a['loss_axis'])


def test_maintained_paths_cache_predicted_condition_and_reload(tmp_path):
    pytest.importorskip('neuralop');torch.set_num_threads(1);torch.manual_seed(18)
    args=arguments();model=build_model(args,statistics()).eval()
    with torch.no_grad():model.sia_shape_gain.fill_(.1)
    pre=torch.rand(1,1,32,32);pm=(torch.rand_like(pre)>.2).float()
    batch=dict(input=pre,target=torch.rand_like(pre),condition=torch.zeros(1,1),mask_pre=pm,
               mask_post=torch.ones_like(pre),sia_vector=torch.tensor([[.6,-.3]]),label=[1])
    counts=[];conditions=[]
    h1=model.sia_readout.register_forward_hook(lambda *unused:counts.append(1))
    h2=model.unet.register_forward_pre_hook(lambda m,a:conditions.append(a[2]))
    before=sample_batch(model,batch,args,'cpu');h1.remove();h2.remove()
    assert len(counts)==1 and len(conditions)==args.sample_steps
    assert all(c is conditions[0] for c in conditions)
    train=forward_batch(model,batch,'cpu')
    torch.testing.assert_close(train['sia_condition'],quadrupole_field(train['q_clean'],32,32,pm))
    path=tmp_path/'temporary.pth'
    try:
        torch.save(dict(pipeline_version=PIPELINE_VERSION,config=vars(args),stats=statistics(),model=model.state_dict(),ema=None),path)
        restored,_,_,_=load_checkpoint(path,'cpu',cpu_threads=1)
        after=sample_batch(restored,batch,args,'cpu')
        for k in ['pred','sia','sia_vector']:torch.testing.assert_close(before[k],after[k],rtol=0,atol=0)
    finally:path.unlink(missing_ok=True)


@pytest.mark.parametrize('extra',[
    ['--sia_prediction','axial_direct'],['--image_backbone','unet'],
    ['--sia_velocity_coupling'],['--sia_rank_weight','.1'],['--augmentation','d4'],
])
def test_spatial_conditioning_scope(extra):
    with pytest.raises(ValueError):arguments(extra=extra)
