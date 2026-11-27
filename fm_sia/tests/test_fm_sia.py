"""Mathematical and integration regression tests; no patient data required."""
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'cddpm'))
from model.fm_sia import JointFlowMatching, fixed_sia_noise
from model.fm_common import ode_step
from model.unet import ConditionalUNet
from model.encoder import ConditionEncoder
from metrics import MetricAccumulator
from transforms import DataTransform
from dataloader import fit_image_stats, CornealTopoDataset, validate_table
from train import get_args

torch.set_num_threads(1)


class IdentityCondition(nn.Module):
    def forward(self,x): return x


class Oracle(nn.Module):
    def __init__(self,vx,vz):
        super().__init__(); self.vx=vx; self.vz=vz; self.times=[]
    def forward(self,x,t,cond,clinical=None,sia_state=None):
        self.times.append(t.detach().clone())
        return self.vx.expand_as(x),self.vz.expand_as(sia_state)


def test_exact_targets_and_endpoint_nontrivial_normalization():
    pre=torch.zeros(2,1,8,8); post=pre+0.3
    q=torch.tensor([[0.6,0.8],[-0.3,0.4]])
    mean=torch.tensor([0.2,-0.1]); std=torch.tensor([0.5,2.0])
    eps=torch.tensor([[0.7,-1.0],[0.2,0.3]])
    velocity=(q-mean)/std-eps
    oracle=Oracle(torch.tensor(0.3),velocity)
    model=JointFlowMatching(IdentityCondition(),oracle,mean,std,sample_steps=7)
    out=model(pre,post,q,t=torch.tensor([0.2,0.85]),noise=eps)
    assert out['loss'].item()<1e-12
    torch.testing.assert_close(out['sia_clean'],torch.tensor([1.0,0.5]))
    oracle.times=[]
    pred=model.sample(pre,noise=eps)
    torch.testing.assert_close(pred['pred'],post)
    torch.testing.assert_close(pred['sia_vector'],q)
    assert len(oracle.times)==7
    assert oracle.times[-1][0].item()==pytest.approx(6/7)


@pytest.mark.parametrize('method',['euler','heun','rk4'])
def test_ode_simultaneous_coupling(method):
    # dx/dt=z, dz/dt=-x: both derivatives read the old joint state.
    x,z=torch.tensor(1.),torch.tensor(0.)
    state=(x,z)
    field=lambda y,t:(y[1],-y[0])
    for k in range(100): state=ode_step(field,state,k/100,.01,method)
    assert state[0].item()==pytest.approx(np.cos(1),abs=.006)
    assert state[1].item()==pytest.approx(-np.sin(1),abs=.006)


def test_physical_zero_sia_origin_matches_velocity_target_and_both_ode_endpoints():
    pre = torch.zeros(2, 1, 8, 8)
    q = torch.tensor([[.6, .8], [-.4, .3]])
    mean, std = torch.tensor([.2, -.1]), torch.tensor([.5, 2.])
    oracle = Oracle(torch.tensor(.25), q/std)
    model = JointFlowMatching(IdentityCondition(), oracle, mean, std,
                              zero_induced_source=True, sample_steps=5)
    for noise in (torch.ones_like(q), -torch.ones_like(q)):
        initial = model.initial_sia_state(noise)
        torch.testing.assert_close(model.decode(initial)[0], torch.zeros_like(q))
        result = model(pre, pre+.25, q, t=torch.tensor([0., .8]), noise=noise)
        assert result['loss'].item() < 1e-12
        prediction = model.sample(pre, noise=noise)
        torch.testing.assert_close(prediction['sia_vector'], q)
        torch.testing.assert_close(prediction['sia'], q.norm(dim=-1))
        torch.testing.assert_close(prediction['pred'], pre+.25)


def test_source_velocity_uses_the_same_draw_along_the_entire_joint_path():
    pre = torch.zeros(2, 1, 8, 8)
    mean, std = torch.tensor([-.3, .2]), torch.tensor([.4, 1.5])
    q = torch.tensor([[.2, -.7], [.6, .3]])
    endpoint = (q-mean)/std
    oracle = Oracle(torch.tensor(.25), endpoint)
    model = JointFlowMatching(IdentityCondition(), oracle, mean, std,
                              subtract_source_velocity=True, sample_steps=7)
    for draw in (torch.ones_like(q), -torch.ones_like(q)):
        output = model(pre, pre+.25, q, t=torch.tensor([0., .7]), noise=draw)
        torch.testing.assert_close(output['sia_velocity_pred'], endpoint-draw)
        assert output['loss'].item() < 1e-12
        predicted = model.sample(pre, noise=draw)
        torch.testing.assert_close(predicted['sia_vector'], q)
        torch.testing.assert_close(predicted['pred'], pre+.25)


def test_optical_velocity_couples_flat_axis_cylinder_at_each_integration_step():
    from train import build_model, sample_batch
    flags = ['--excel_path','unused','--data_root','unused','--conditions','pre_age_years',
             '--fusion_method','film','--sample_steps','4']
    args = get_args([*flags, '--optical_sia_velocity'])
    stats = {'sia':{'mean':[.2, -.1], 'std':[.5, 1.5]}, 'image':{'scale':20.}}
    torch.manual_seed(9)
    parent = build_model(get_args(flags), stats)
    torch.manual_seed(9)
    model = build_model(args, stats)
    assert model.state_dict().keys() == parent.state_dict().keys()
    for name, value in parent.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name], value, atol=0, rtol=0)
    coords = (torch.arange(32) + .5) / 16 - 1
    y, x = torch.meshgrid(-coords, coords, indexing='ij')
    angle = torch.atan2(y, x)
    power = .06 + .02 * torch.cos(2*angle) - .01 * torch.sin(2*angle)
    mask = torch.stack(((x >= -.15) | (y < 0), x.square() + y.square() > .5))[:, None].float()
    pre = torch.zeros_like(mask)
    class Field(nn.Module):
        def __init__(self):
            super().__init__()
            self.amplitude = nn.Parameter(torch.ones(()))
            self.feedback = 0.
        def forward(self, image, time, condition, clinical=None, sia_state=None):
            return self.amplitude * power.expand_as(image), self.feedback * sia_state
    field = Field()
    model.unet, model.condition_encoder = field, IdentityCondition()
    noise = torch.tensor([[.2, -.4],[-.5, .6]])
    norm = torch.tensor(stats['sia']['std'])
    mean = torch.tensor(stats['sia']['mean'])
    cylinder = torch.tensor([[-.8, .4],[0., 0.]])
    target = mean + noise*norm + cylinder
    out = model(pre, pre+power, target, noise=noise, t=torch.tensor([.2, .6]),
                mask=mask, pre_mask=mask)
    torch.testing.assert_close(out['sia_velocity_pred'], cylinder / norm)
    assert out['loss'].item() < 1e-12
    # SIA supervision can update image velocity even with the image loss excluded.
    altered = model(pre, pre+power, target+.2, noise=noise, pre_mask=mask)
    altered['loss_vector'].backward()
    assert field.amplitude.grad.abs().item() > 0
    field.feedback = .5
    result = model.sample(pre, noise=noise, pre_mask=mask)
    expected = noise.clone()
    for _ in range(4):
        expected += (.5 * expected + cylinder/norm) / 4
    torch.testing.assert_close(result['sia_vector'], mean + norm*expected)
    torch.testing.assert_close(result['optical_sia_vector'], cylinder)
    torch.testing.assert_close(result['pred'], pre+power)
    assert result['optical_resolved'].tolist() == [True,False]
    batch = dict(input=pre, condition=torch.ones(2,1), mask_pre=mask, label=[1,2])
    sampled = sample_batch(model, batch, args, 'cpu')
    torch.testing.assert_close(sampled['optical_sia_vector'], cylinder)
    torch.testing.assert_close(sampled['sia_vector'].norm(dim=-1), sampled['sia'])


def test_transported_radius_preserves_parent_weights_and_consistent_endpoints():
    from train import build_model, forward_batch, sample_batch
    flags = ['--excel_path','unused','--data_root','unused','--zero_induced_source',
             '--conditions','pre_age_years','--fusion_method','film','--sample_steps','2']
    stats = {'sia':{'mean':[.2, -.1], 'std':[.5, 2.]}}
    torch.manual_seed(42)
    parent = build_model(get_args(flags), stats)
    torch.manual_seed(42)
    args = get_args([*flags,'--transported_radius'])
    augmented = build_model(args, stats)
    for name,value in parent.named_parameters():
        torch.testing.assert_close(augmented.get_parameter(name), value, rtol=0, atol=0)
    branch = augmented.unet.sia_interaction
    assert sum(p.numel() for p in augmented.parameters())-sum(p.numel() for p in parent.parameters()) == 2*args.sia_embed_dim+1
    pre = torch.rand(2, 1, 32, 32)
    q = torch.tensor([[.3,.4],[-.4,.3]])
    batch = dict(input=pre,target=pre+.25,condition=torch.ones(2,1),mask_pre=torch.ones_like(pre),
                 mask_post=torch.ones_like(pre),sia_vector=q,label=[1,2])
    out = forward_batch(augmented,batch,'cpu')
    out['loss'].backward()
    assert branch.radius_velocity.weight.grad.abs().sum()>0
    augmented.eval()
    result = sample_batch(augmented,batch,args,'cpu')
    torch.testing.assert_close(result['sia_vector'].norm(dim=-1), result['sia'])
    origin = augmented.initial_sia_state(torch.ones(2,3))
    target = torch.cat((q, q.norm(dim=-1,keepdim=True)),dim=-1)
    velocity = (target-augmented.sia_mean)/augmented.sia_std-origin
    augmented.unet = Oracle(torch.tensor(.25),velocity)
    restored = augmented.sample(pre,noise=torch.randn(2,3))
    torch.testing.assert_close(restored['sia_vector'],q)
    torch.testing.assert_close(restored['pred'],pre+.25)
    # Change only the integrated radius. Direction and reported norm must agree.
    velocity[:, 2] *= 2
    torch.testing.assert_close(augmented.sample(pre)['sia_vector'],2*q)


def test_eye_centered_flow_fits_training_eyes_and_keeps_the_original_learned_conditions(tmp_path):
    from clinical_features import EYE_SIGN, PRE_EYE, prepare_clinical_features
    from dataloader import feature_stats
    from predict import predict_case
    from train import build_model, fit_sia_stats
    flags = ['--excel_path','unused','--data_root','unused','--fusion_method','film',
             '--sample_steps','2','--image_size','32','--conditions','pre_age_years']
    parent_args = get_args(flags)
    args = get_args([*flags,EYE_SIGN,'--center_sia_by_eye'])
    frame = prepare_clinical_features(pd.DataFrame({
        'label_index':[1,2,3,4],PRE_EYE:['OS','OS','OD','OD'],'pre_age_years':[30.,40.,50.,40.],
        'delta_front_X':[-.7,-.3,1.2,99.],'delta_front_Y':[.3,.1,-.2,-99.]}),args.conditions)
    train_ids = [1,2,3]
    sia = fit_sia_stats(args,frame,train_ids)
    np.testing.assert_allclose(sia['eye_centers'],[[-.5,.2],[1.2,-.2]])
    assert sia['eye_counts']==[2,1]
    frame.loc[3,['delta_front_X','delta_front_Y']]=[-999.,999.]
    assert fit_sia_stats(args,frame,train_ids)==sia
    one_eye = fit_sia_stats(args,frame,[1,2])
    assert one_eye['eye_counts']==[2,0] and one_eye['eye_centers'][1]==one_eye['mean']
    stats=dict(sia=sia,clinical=feature_stats(frame,train_ids,args.conditions),image=dict(minimum=0.,scale=1.))
    torch.manual_seed(17)
    parent = build_model(parent_args,stats)
    torch.manual_seed(17)
    model = build_model(args,stats).eval()
    assert dict(parent.named_parameters()).keys()==dict(model.named_parameters()).keys()
    for name,w in parent.named_parameters():
        torch.testing.assert_close(model.get_parameter(name),w,rtol=0,atol=0)
    pre = torch.ones(2,1,32,32)
    raw_clinical = torch.tensor([[40.,-1.],[40.,1.]])
    clinical = (raw_clinical-torch.tensor(stats['clinical']['mean']))/torch.tensor(stats['clinical']['std'])
    learned,centers = model.conditions_and_center(clinical)
    torch.testing.assert_close(learned,clinical[:,:1])
    torch.testing.assert_close(centers,torch.tensor(sia['eye_centers']))
    noise = torch.tensor([[.2,-.7],[.2,-.7]])
    result = model.sample(pre,clinical=clinical,noise=noise)
    torch.testing.assert_close(result['pred'][0],result['pred'][1])
    torch.testing.assert_close(result['sia_vector'][1]-result['sia_vector'][0],centers[1]-centers[0])
    speed = torch.tensor([.3,-.2])
    model.unet = Oracle(torch.tensor(.125),speed)
    target = centers+model.sia_std*(noise+speed)
    loss = model(pre,pre+.125,target,clinical=clinical,t=torch.tensor([.2,.75]),noise=noise)
    assert loss['loss'].item()<1e-12
    torch.testing.assert_close(loss['sia_vector_clean'],target)
    pred = model.sample(pre,clinical=clinical,noise=noise)
    torch.testing.assert_close(pred['sia_vector'],target)
    torch.testing.assert_close(pred['sia'],target.norm(dim=-1))
    map_path=tmp_path/'pre.npy'
    np.save(map_path,pre[0,0].numpy())
    info,_=predict_case(model,args,stats,'cpu',map_path,[40.,1.],sample_id='synthetic')
    draw=fixed_sia_noise(['synthetic'],args.eval_seed)[0]
    expected=centers[1]+model.sia_std*(draw+speed)
    torch.testing.assert_close(torch.tensor([info['pred_delta_X'],info['pred_delta_Y']]),expected)
    assert info['pred_SIA_D']==pytest.approx(float(expected.norm()))


def test_zero_norm_gradient_and_auxiliary_backprop():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__(); self.bias=nn.Parameter(torch.tensor([.3,.4]))
        def forward(self,x,t,cond,clinical=None,sia_state=None):
            return torch.zeros_like(x),self.bias[None].expand(x.shape[0],-1)
    tiny=Tiny(); m=JointFlowMatching(IdentityCondition(),tiny,[0,0],[1,1])
    x=torch.zeros(1,1,8,8)
    out=m(x,x,torch.tensor([[.6,.8]]),t=torch.tensor([0.]),noise=torch.zeros(1,2))
    out['loss_sia'].backward()
    assert tiny.bias.grad.abs().sum()>0 and torch.isfinite(tiny.bias.grad).all()
    tiny.bias.data.zero_(); tiny.zero_grad()
    m(x,x,torch.zeros(1,2),t=torch.tensor([0.]),noise=torch.zeros(1,2))['loss'].backward()
    assert torch.isfinite(tiny.bias.grad).all()


def test_fixed_noise_batch_invariance():
    a=fixed_sia_noise([10,20,30],123)
    b=torch.cat([fixed_sia_noise([30],123),fixed_sia_noise([10,20],123)])
    torch.testing.assert_close(a,b[[1,2,0]])
    assert not torch.equal(a,fixed_sia_noise([10,20,30],123,1))


def test_masked_metrics_all_images_and_batch_invariance():
    target=torch.zeros(3,1,12,12); pred=target.clone()
    pred[0]=.1; pred[1]=.2; pred[2]=.4
    mask=torch.ones_like(target); mask[:,:,:,0]=0
    pred[:,:,:,0]=999
    a=MetricAccumulator(); a.update(pred,target,mask)
    b=MetricAccumulator()
    b.update(pred[:2],target[:2],mask[:2]); b.update(pred[2:],target[2:],mask[2:])
    assert a.compute()==pytest.approx(b.compute())
    assert a.ssim_count==3
    assert a.compute()['RMSE']==pytest.approx(np.sqrt((.01+.04+.16)/3))


def test_label_transform_preserves_norm():
    q=torch.tensor([.6,.8])
    for h in [False,True]:
        for v in [False,True]:
            for angle in [0,90,180,270]:
                transformed=DataTransform.transform_vector(q,dict(hflip=h,vflip=v,angle=angle))
                assert transformed.norm()==pytest.approx(1.)
    torch.testing.assert_close(DataTransform.transform_vector(q,dict(hflip=True,vflip=False,angle=90)),torch.tensor([-.6,.8]))


def test_full_unet_joint_backward_and_sample():
    encoder=ConditionEncoder()
    net=ConditionalUNet(discrete=False,clinical_dim=4,fusion_method='cross_attention',sia_embed_dim=16)
    model=JointFlowMatching(encoder,net,[.1,.2],[.3,.8],sample_steps=2)
    pre=torch.rand(1,1,16,16); post=torch.rand_like(pre); clinical=torch.randn(1,4)
    out=model(pre,post,torch.tensor([[.6,.8]]),clinical=clinical)
    out['loss'].backward()
    for module in [net.sia_interaction.velocity,net.sia_interaction.film,encoder.in_conv]:
        grads=[p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum().item() for g in grads)>0
    model.eval()
    pred=model.sample(pre,clinical=clinical,noise=fixed_sia_noise([1]))
    assert pred['pred'].shape==pre.shape
    assert pred['sia_vector'].shape==(1,2)
    torch.testing.assert_close(pred['sia'],pred['sia_vector'].norm(dim=1))


def test_train_only_normalization_and_minimum_pixel_valid(tmp_path):
    root=tmp_path/'raw'/'FRONT'; root.mkdir(parents=True)
    for label,offset in [(1,0),(2,1000)]:
        for phase in ['pre','post']:
            image=np.arange(64,dtype=np.float32).reshape(8,8)+offset
            image[0,0]=np.nan
            np.save(root/f'{label}_{phase}.npy',image)
    args=SimpleNamespace(data_root=tmp_path,input_maps=['FRONT'],output_maps=['FRONT'],
        image_size=8,invalid_zero=False,conditions=[],targets=[],model='fm')
    stats=fit_image_stats(args,[1])
    assert stats['maximum']==63
    df=pd.DataFrame({'label_index':[1,2]})
    ds=CornealTopoDataset(args,None,tmp_path,dataframe=df,image_size=8,image_stats=stats)
    item=ds[0]
    assert item['input'][0,0,1]==0  # valid training minimum, not a missing pixel
    assert item['mask_pre'][0,0,1]==1 and item['mask_pre'][0,0,0]==0
    assert ds[1]['input'].max()>1  # no test-dependent rescaling or clipping


def test_label_validation_and_leakage_guard():
    args=get_args(['--excel_path','unused','--data_root','unused'])
    df=pd.DataFrame(dict(label_index=[1],delta_front_X=[.6],delta_front_Y=[.8],front_SIA_D=[.5]))
    with pytest.raises(ValueError,match='norm mismatch'): validate_table(df,args)
    df.front_SIA_D=1.; df['sia_valid']=False
    with pytest.raises(ValueError,match='sia_valid'): validate_table(df,args)
    with pytest.raises(ValueError,match='Outcome'): get_args(['--excel_path','x','--data_root','x','--conditions','front_SIA_D'])
    with pytest.raises(ValueError,match='CDDPM-only'): get_args(['--excel_path','x','--data_root','x','--sampler','ddim'])
