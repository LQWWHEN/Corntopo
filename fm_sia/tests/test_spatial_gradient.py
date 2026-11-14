import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'cddpm'))
from model.spatial_gradient import masked_gradient_mse
from train import get_args, build_model, saved_config_value, PIPELINE_VERSION
from predict import load_checkpoint


def arguments(weight=.25, extra=()):
    return get_args(['--excel_path','unused','--data_root','unused','--sia_prediction','polar_direct',
        '--conditions','pre_age_years','--fusion_method','film','--lambda_sia','1','--sample_steps','2',
        '--image_backbone','fno','--fno_modes','4','4','--fno_hidden_channels','8','--fno_layers','1',
        '--image_gradient_weight',str(weight),*extra])


def statistics():
    return dict(sia=dict(mean=[.2,.3],std=[.5,.6],polar_median_magnitude=.5,
                        polar_initial_phase=.4,polar_axis_training_count=2),image=dict(scale=20.))


def test_signed_slopes_equal_case_average_and_missing_boundaries():
    pred=torch.tensor([[[[0.,1.,2.],[0.,1.,2.]]],[[[0.,2.,999.],[0.,2.,999.]]]],dtype=torch.float64,requires_grad=True)
    target=torch.zeros_like(pred);mask=torch.ones_like(pred);mask[1,:,:,2]=0
    # First case: four unit horizontal slopes among7 edges. Second: two slopes2 among4.
    expected=(4/7+8/4)/2
    loss=masked_gradient_mse(pred,target,mask)
    assert loss.item()==pytest.approx(expected)
    loss.backward();assert torch.count_nonzero(pred.grad[1,:,:,2])==0
    altered=pred.detach().clone();altered[1,:,:,2]=-10000
    torch.testing.assert_close(loss,masked_gradient_mse(altered,target,mask))
    # Opposite signed slopes are not interchangeable with equal absolute gradient magnitudes.
    ramp=torch.arange(4,dtype=torch.float64)[None,None,None,:]
    assert masked_gradient_mse(ramp,-ramp).item()==4
    assert masked_gradient_mse(ramp,ramp).item()==0


def test_constant_offsets_no_wrap_and_empty_edges():
    step=torch.tensor([[[[0.,0.,1.]]]],requires_grad=True)
    assert masked_gradient_mse(step,torch.zeros_like(step)).item()==.5
    assert masked_gradient_mse(step+10,step).item()==0
    mask=torch.tensor([[[[1.,0.,1.]]]])
    loss=masked_gradient_mse(step,torch.zeros_like(step),mask)
    loss.backward();assert loss.item()==0 and torch.count_nonzero(step.grad)==0


def test_rotations_reflections_and_autograd_finite_difference():
    torch.manual_seed(19)
    pred=torch.randn(2,1,4,5,dtype=torch.float64,requires_grad=True)
    target=torch.randn_like(pred);mask=(torch.rand_like(pred)>.2).double()
    reference=masked_gradient_mse(pred,target,mask)
    for transform in [lambda v:v.rot90(1,(-2,-1)),lambda v:v.flip(-1)]:
        torch.testing.assert_close(reference,masked_gradient_mse(transform(pred),transform(target),transform(mask)))
    assert torch.autograd.gradcheck(lambda p:masked_gradient_mse(p,target,mask),(pred,))


def test_parent_weights_rng_sampling_and_loss_decomposition():
    pytest.importorskip('neuralop');torch.set_num_threads(1)
    torch.manual_seed(19);parent=build_model(arguments(0),statistics()).eval();rng=torch.get_rng_state()
    torch.manual_seed(19);model=build_model(arguments(),statistics()).eval()
    assert torch.equal(rng,torch.get_rng_state())
    for k,v in parent.state_dict().items():torch.testing.assert_close(v,model.state_dict()[k],rtol=0,atol=0)
    pre=torch.rand(2,1,32,32);post=torch.rand_like(pre);clinical=torch.zeros(2,1)
    mask=(torch.rand_like(pre)>.2).float();gt=torch.tensor([[.5,.2],[-.1,.3]])
    kw=dict(clinical=clinical,mask=mask,t=torch.tensor([.2,.8]))
    a=parent(pre,post,gt,**kw);b=model(pre,post,gt,**kw)
    for key in ['velocity_pred','q_clean','loss_sia','loss_axis','loss_image_normalized']:
        torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
    assert 'loss_image_gradient' not in a
    torch.testing.assert_close(b['loss_image_gradient'],masked_gradient_mse(b['velocity_pred'],post-pre,mask))
    torch.testing.assert_close(b['loss_image'],a['loss_image']+.25*b['loss_image_gradient'])
    torch.testing.assert_close(b['loss'],b['loss_image']+b['loss_sia']+b['loss_axis'])
    gradient=torch.autograd.grad(b['loss_image_gradient'],model.condition_encoder.in_conv.weight)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum()>0
    # Target information is confined to losses and the ordinary FM interpolation path.
    changed=model(pre,post+7,gt+8,**kw)
    torch.testing.assert_close(b['q_clean'],changed['q_clean'],rtol=0,atol=0)
    before=parent.sample(pre,clinical=clinical);after=model.sample(pre,clinical=clinical)
    for key in ['pred','sia','sia_vector']:torch.testing.assert_close(before[key],after[key],rtol=0,atol=0)
    assert saved_config_value({},'image_gradient_weight')==0


@pytest.mark.parametrize('legacy',[False,True])
def test_saved_configuration_and_legacy_reload(tmp_path,legacy):
    pytest.importorskip('neuralop');torch.set_num_threads(1)
    args=arguments(0 if legacy else .25);model=build_model(args,statistics()).eval()
    pre=torch.rand(1,1,32,32);clinical=torch.zeros(1,1);before=model.sample(pre,clinical=clinical)
    config=vars(args).copy()
    if legacy:config.pop('image_gradient_weight')
    path=tmp_path/'temporary.pth'
    try:
        torch.save(dict(pipeline_version=PIPELINE_VERSION,config=config,stats=statistics(),model=model.state_dict(),ema=None),path)
        restored,_,_,_=load_checkpoint(path,'cpu',cpu_threads=1)
        assert restored.image_gradient_weight==(0 if legacy else .25)
        after=restored.sample(pre,clinical=clinical)
        for key in ['pred','sia','sia_vector']:torch.testing.assert_close(before[key],after[key],rtol=0,atol=0)
    finally:path.unlink(missing_ok=True)


@pytest.mark.parametrize('extra',[
    ['--image_gradient_weight','-1'],['--image_gradient_weight','nan'],['--sia_prediction','axial_direct'],
    ['--sia_spatial_conditioning'],['--sia_velocity_coupling'],['--sia_rank_weight','.1'],
    ['--augmentation','d4'],['--reconstruction_domain','full'],
])
def test_gradient_configuration_scope(extra):
    with pytest.raises(ValueError):arguments(extra=extra)
