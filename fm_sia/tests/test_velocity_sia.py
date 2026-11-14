import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'cddpm'))
from model.velocity_sia import velocity_moments
from model.fm_common import masked_mse
from train import get_args, build_model, saved_config_value, PIPELINE_VERSION, forward_batch, sample_batch
from predict import load_checkpoint


def arguments(coupled=True, extra=()):
    return get_args(['--excel_path', 'unused', '--data_root', 'unused',
        '--sia_prediction', 'polar_direct', '--conditions', 'pre_age_years', '--fusion_method', 'film',
        '--lambda_sia', '1', '--sample_steps', '3', '--image_backbone', 'fno',
        '--fno_modes', '4', '4', '--fno_hidden_channels', '8', '--fno_layers', '1',
        *(['--sia_velocity_coupling'] if coupled else []), *extra])


def statistics():
    return dict(sia=dict(mean=[.2, .3], std=[.5, .6], polar_median_magnitude=.5,
                         polar_initial_phase=.4, polar_axis_training_count=2), image=dict(scale=20.))


def test_fixed_velocity_moments_units_and_reflection_parity():
    torch.set_num_threads(1)
    value=torch.randn(2,1,33,33,dtype=torch.float64,requires_grad=True)
    descriptor=velocity_moments(value)
    assert descriptor.shape==(2,20)
    torch.testing.assert_close(velocity_moments(value*20),descriptor*20)
    parity=value.new_tensor([1,-1,1,1,-1]).repeat(4)
    torch.testing.assert_close(velocity_moments(value.flip(-1)),descriptor*parity,rtol=1e-10,atol=1e-10)
    constant=velocity_moments(torch.ones_like(value)).reshape(2,4,5)
    torch.testing.assert_close(constant[:,:,0],torch.ones_like(constant[:,:,0]))
    torch.testing.assert_close(constant[:,:,1:],torch.zeros_like(constant[:,:,1:]),atol=1e-10,rtol=0)
    upper=value.new_zeros(1,1,33,33);upper[0,0,8,16]=1
    # The upper half is positive y, consistent with the parent spatial moments.
    assert velocity_moments(upper).reshape(1,4,5)[0,:,2].sum()>0
    descriptor.square().sum().backward()
    assert torch.isfinite(value.grad).all() and value.grad.abs().sum()>0


def test_descriptors_exclude_preoperative_invalid_pixels():
    value=torch.randn(2,1,32,32,requires_grad=True)
    mask=(torch.rand_like(value)>.2).float()
    expected=velocity_moments(value,mask)
    changed=value.detach().clone();changed[mask==0]=99999
    torch.testing.assert_close(expected,velocity_moments(changed,mask))
    expected.square().sum().backward()
    assert torch.count_nonzero(value.grad[mask==0])==0
    torch.testing.assert_close(velocity_moments(value,torch.zeros_like(mask)),torch.zeros_like(expected))


def test_coupling_keeps_parent_rng_parameters_and_initial_predictions():
    pytest.importorskip('neuralop');torch.set_num_threads(1)
    torch.manual_seed(15);parent=build_model(arguments(False),statistics()).eval();parent_rng=torch.get_rng_state()
    torch.manual_seed(15);model=build_model(arguments(),statistics()).eval()
    assert torch.equal(parent_rng,torch.get_rng_state())
    dofs=lambda m:sum(p.numel()*(2 if p.is_complex() else 1) for p in m.parameters())
    assert dofs(model)-dofs(parent)==2560
    for name,value in parent.state_dict().items():
        other=model.state_dict()[name]
        if name=='sia_readout.0.weight':
            assert torch.count_nonzero(other[:,-20:])==0
            other=other[:,:-20]
        torch.testing.assert_close(value,other,rtol=0,atol=0)
    pre=torch.rand(1,1,32,32);clinical=torch.zeros(1,1)
    a=parent.sample(pre,clinical=clinical);b=model.sample(pre,clinical=clinical)
    for key in ('pred','sia','sia_vector'):torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
    assert model.velocity_feature_scale==20.
    assert saved_config_value({},'sia_velocity_coupling') is False
    assert saved_config_value({},'endpoint_image_weight')==.5


def test_preoperative_only_graph_and_cross_task_gradient():
    pytest.importorskip('neuralop');torch.set_num_threads(1)
    model=build_model(arguments(),statistics()).eval()
    with torch.no_grad():
        model.sia_readout[0].weight[:,-20:].normal_(0,.05)
        model.sia_readout[-1].weight.normal_(0,.01)
    pre=torch.rand(1,1,32,32,requires_grad=True);post=torch.rand_like(pre,requires_grad=True)
    clinical=torch.tensor([[.2]],requires_grad=True);gt=torch.tensor([[.6,-.3]])
    mask=(torch.rand_like(pre)>.2).float()
    a=model(pre,post,gt,clinical=clinical,mask=mask,t=torch.tensor([.2]))
    b=model(pre,post+10,gt,clinical=clinical,mask=1-mask,t=torch.tensor([.8]),noise=torch.ones(1,2)*100)
    torch.testing.assert_close(a['q_clean'],b['q_clean'],rtol=0,atol=0)
    torch.testing.assert_close(a['velocity_initial'],b['velocity_initial'],rtol=0,atol=0)
    q_grads=torch.autograd.grad(a['q_clean'].sum(),(pre,clinical,post),retain_graph=True,allow_unused=True)
    assert q_grads[2] is None and all(v is not None and v.abs().sum()>0 for v in q_grads[:2])
    sia_grads=torch.autograd.grad(a['loss_sia']+a['loss_axis'],tuple(model.unet.fno.parameters()),retain_graph=True,allow_unused=True)
    assert any(v is not None and torch.isfinite(v).all() and v.abs().sum()>0 for v in sia_grads)
    anchor=masked_mse(a['velocity_initial'],post-pre,mask)
    torch.testing.assert_close(a['loss_image_initial'],anchor)
    torch.testing.assert_close(a['loss_image'],.5*a['loss_image_normalized']+.5*anchor)
    torch.testing.assert_close(a['loss'],a['loss_image']+a['loss_sia']+a['loss_axis'])


def test_inference_reuses_initial_field_and_saved_weights(tmp_path):
    pytest.importorskip('neuralop');torch.set_num_threads(1)
    args=arguments();model=build_model(args,statistics()).eval()
    pre=torch.rand(1,1,32,32);clinical=torch.zeros(1,1)
    calls=[];handle=model.unet.register_forward_hook(lambda *unused:calls.append(1))
    before=model.sample(pre,clinical=clinical);handle.remove()
    assert len(calls)==args.sample_steps
    path=tmp_path/'temporary.pth'
    torch.save(dict(pipeline_version=PIPELINE_VERSION,config=vars(args),stats=statistics(),
                    model=model.state_dict(),ema=None),path)
    restored,_,_,_=load_checkpoint(path,'cpu',cpu_threads=1)
    after=restored.sample(pre,clinical=clinical)
    for key in ('pred','sia','sia_vector'):torch.testing.assert_close(before[key],after[key],rtol=0,atol=0)
    path.unlink()


def test_maintained_batch_paths_use_only_preoperative_mask():
    pytest.importorskip('neuralop');torch.set_num_threads(1)
    args=arguments();model=build_model(args,statistics()).eval()
    with torch.no_grad():
        model.sia_readout[0].weight[:,-20:].normal_(0,.05)
        model.sia_readout[-1].weight.normal_(0,.01)
    pre=torch.rand(1,1,32,32);pre_mask=(torch.rand_like(pre)>.2).float()
    batch=dict(input=pre,target=torch.rand_like(pre),condition=torch.zeros(1,1),
               mask_pre=pre_mask,mask_post=torch.ones_like(pre),sia_vector=torch.tensor([[.6,-.3]]),label=[1])
    predicted=sample_batch(model,batch,args,'cpu')
    expected=model.sample(pre,clinical=batch['condition'],pre_mask=pre_mask)
    torch.testing.assert_close(predicted['sia_vector'],expected['sia_vector'],rtol=0,atol=0)
    a=forward_batch(model,batch,'cpu')
    batch['target']=batch['target']+10;batch['mask_post']=1-pre_mask*.5
    b=forward_batch(model,batch,'cpu')
    torch.testing.assert_close(a['q_clean'],b['q_clean'],rtol=0,atol=0)
    torch.testing.assert_close(a['q_clean'],expected['sia_vector'],rtol=0,atol=0)


@pytest.mark.parametrize('extra',[
    ['--sia_prediction','axial_direct'],['--image_backbone','unet'],
    ['--endpoint_image_weight','0'],['--endpoint_image_weight','1'],['--endpoint_image_weight','nan'],
])
def test_coupling_scope_and_endpoint_weight(extra):
    with pytest.raises(ValueError):arguments(extra=extra)
