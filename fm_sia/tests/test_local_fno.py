import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'cddpm'))
pytest.importorskip('neuralop')
from model.local_fno import mean_free_correction
from train import get_args,build_model,saved_config_value,PIPELINE_VERSION,forward_batch,sample_batch
from predict import load_checkpoint


def arguments(enabled=True,extra=()):
    return get_args(['--excel_path','unused','--data_root','unused','--sia_prediction','polar_direct',
        '--conditions','pre_age_years','--fusion_method','film','--lambda_sia','1','--sample_steps','3',
        '--image_backbone','fno','--fno_modes','4','4','--fno_hidden_channels','8','--fno_layers','1',
        *(['--image_local_residual'] if enabled else []),*extra])


def statistics():
    return dict(sia=dict(mean=[.2,.3],std=[.5,.6],polar_median_magnitude=.5,
                        polar_initial_phase=.4,polar_axis_training_count=2),image=dict(scale=20.))


def test_preoperative_projection_hand_calculation_and_gradients():
    raw=torch.tensor([[[[1.,3.,999.],[5.,7.,999.]]]],dtype=torch.float64,requires_grad=True)
    mask=torch.tensor([[[[1.,1.,0.],[1.,1.,0.]]]],dtype=torch.float64)
    result=mean_free_correction(raw,mask)
    torch.testing.assert_close(result,raw.new_tensor([[[[-3.,-1.,0.],[1.,3.,0.]]]]))
    assert result.sum().item()==0
    result.square().sum().backward();assert torch.count_nonzero(raw.grad[mask==0])==0
    torch.testing.assert_close(mean_free_correction(raw+10,mask),result)
    assert torch.count_nonzero(mean_free_correction(raw,torch.zeros_like(mask)))==0
    for transform in [lambda x:x.rot90(1,(-2,-1)),lambda x:x.flip(-1)]:
        torch.testing.assert_close(mean_free_correction(transform(raw),transform(mask)),transform(result))
    assert torch.autograd.gradcheck(lambda x:mean_free_correction(x,mask),(raw,))
    with pytest.raises(ValueError):mean_free_correction(raw,mask[:,:,:,:1])


def test_zero_branch_preserves_parent_rng_capacity_and_predictions():
    torch.set_num_threads(1);torch.manual_seed(20)
    parent=build_model(arguments(False),statistics()).eval();rng=torch.get_rng_state()
    torch.manual_seed(20);model=build_model(arguments(),statistics()).eval()
    assert torch.equal(rng,torch.get_rng_state())
    for k,v in parent.state_dict().items():torch.testing.assert_close(v,model.state_dict()[k],rtol=0,atol=0)
    dofs=lambda m:sum(p.numel()*(2 if p.is_complex() else 1) for p in m.parameters())
    assert dofs(model)-dofs(parent)==32528
    pre=torch.rand(1,1,32,32);post=torch.rand_like(pre);clinical=torch.zeros(1,1);gt=torch.tensor([[.5,.2]])
    pm=(torch.rand_like(pre)>.2).float();kw=dict(clinical=clinical,t=torch.tensor([.4]))
    a=parent(pre,post,gt,**kw);b=model(pre,post,gt,pre_mask=pm,**kw)
    for k in ['loss','loss_image','loss_sia','loss_axis','velocity_pred','q_clean']:
        torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
    a=parent.sample(pre,clinical=clinical);b=model.sample(pre,clinical=clinical,pre_mask=pm)
    for k in ['pred','sia','sia_vector']:torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
    assert saved_config_value({},'image_local_residual') is False


def test_learned_branch_is_masked_mean_free_and_trainable():
    torch.set_num_threads(1);model=build_model(arguments(),statistics()).eval()
    with torch.no_grad():model.unet.local_output[-1].weight.normal_(0,.1)
    pre=torch.rand(2,1,32,32);clinical=torch.zeros(2,1);pm=(torch.rand_like(pre)>.2).float()
    condition=model.condition_encoder(pre);t=torch.tensor([.1,.8])
    global_v,local=model.unet.velocity_components(pre,t,condition,clinical,pm)
    torch.testing.assert_close(local.sum((-2,-1)),torch.zeros(2,1),atol=1e-5,rtol=0)
    assert torch.count_nonzero(local[pm==0])==0 and local.abs().sum()>0
    combined=model.unet(pre,t,condition,clinical,pm)
    torch.testing.assert_close(combined,global_v+local)
    gradients=torch.autograd.grad(local.square().mean(),(model.unet.local_fine[0].weight,model.unet.local_output[-1].weight))
    assert all(torch.isfinite(g).all() and g.abs().sum()>0 for g in gradients)
    gt=torch.tensor([[.5,.2],[-.2,.3]]);post=torch.rand_like(pre)
    a=model(pre,post,gt,clinical=clinical,pre_mask=pm,t=t)
    b=model(pre,post+10,gt+5,clinical=clinical,pre_mask=1-pm,mask=pm,t=1-t)
    torch.testing.assert_close(a['q_clean'],b['q_clean'],rtol=0,atol=0)
    changed_gt=model(pre,post,gt+5,clinical=clinical,pre_mask=pm,t=t)
    torch.testing.assert_close(a['velocity_pred'],changed_gt['velocity_pred'],rtol=0,atol=0)


def test_maintained_paths_cache_condition_and_reload(tmp_path):
    torch.set_num_threads(1);args=arguments();model=build_model(args,statistics()).eval()
    with torch.no_grad():model.unet.local_output[-1].weight.normal_(0,.1)
    pre=torch.rand(1,1,32,32);pm=(torch.rand_like(pre)>.2).float()
    batch=dict(input=pre,target=torch.rand_like(pre),condition=torch.zeros(1,1),mask_pre=pm,
               mask_post=torch.ones_like(pre),sia_vector=torch.tensor([[.6,-.3]]),label=[1])
    seen=[];masks=[]
    hook=model.unet.register_forward_pre_hook(lambda m,a,k:(seen.append(a[2]),masks.append(k['pre_mask'])) and None,with_kwargs=True)
    before=sample_batch(model,batch,args,'cpu');hook.remove()
    assert len(seen)==3 and all(x is seen[0] for x in seen) and all(x is pm for x in masks)
    output=forward_batch(model,batch,'cpu')
    torch.testing.assert_close(output['loss'],output['loss_image']+output['loss_sia']+output['loss_axis'])
    path=tmp_path/'temporary.pth'
    try:
        torch.save(dict(pipeline_version=PIPELINE_VERSION,config=vars(args),stats=statistics(),model=model.state_dict(),ema=None),path)
        restored,_,_,_=load_checkpoint(path,'cpu',cpu_threads=1)
        after=sample_batch(restored,batch,args,'cpu')
        for k in ['pred','sia','sia_vector']:torch.testing.assert_close(before[k],after[k],rtol=0,atol=0)
    finally:path.unlink(missing_ok=True)


def test_legacy_polar_checkpoint_without_local_flag(tmp_path):
    torch.set_num_threads(1);args=arguments(False);model=build_model(args,statistics()).eval()
    config=vars(args).copy();config.pop('image_local_residual')
    pre=torch.rand(1,1,32,32);clinical=torch.zeros(1,1);before=model.sample(pre,clinical=clinical)
    path=tmp_path/'legacy.pth'
    try:
        torch.save(dict(pipeline_version=PIPELINE_VERSION,config=config,stats=statistics(),model=model.state_dict(),ema=None),path)
        restored,_,_,_=load_checkpoint(path,'cpu',cpu_threads=1);after=restored.sample(pre,clinical=clinical)
        for k in ['pred','sia','sia_vector']:torch.testing.assert_close(before[k],after[k],rtol=0,atol=0)
    finally:path.unlink(missing_ok=True)


@pytest.mark.parametrize('extra',[
    ['--sia_prediction','axial_direct'],['--image_backbone','unet'],['--sia_spatial_conditioning'],
    ['--sia_velocity_coupling'],['--image_gradient_weight','.25'],['--sia_rank_weight','.1'],
    ['--augmentation','d4'],['--reconstruction_domain','full'],
])
def test_local_branch_configuration_scope(extra):
    with pytest.raises(ValueError):arguments(extra=extra)
