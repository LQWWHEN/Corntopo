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
