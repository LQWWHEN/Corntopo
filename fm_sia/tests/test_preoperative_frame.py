import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'cddpm'))
from model.axial_sia import axial_moments,preoperative_frame_moments
from train import get_args,build_model,saved_config_value

torch.set_num_threads(1)


def test_relative_frame_quarter_turn_and_reflection():
    torch.manual_seed(12)
    feature=torch.randn(2,4,16,16)
    q=torch.tensor([[.6,.8],[-.4,.3]])
    first=preoperative_frame_moments(feature,q).reshape(2,3,4)
    rotated=preoperative_frame_moments(feature.rot90(1,(-2,-1)),-q).reshape(2,3,4)
    torch.testing.assert_close(first,rotated,atol=1e-6,rtol=1e-5)
    reflected=q*torch.tensor([1.,-1.])
    flipped=preoperative_frame_moments(feature.flip(-1),reflected).reshape(2,3,4)
    torch.testing.assert_close(first[:,:2],flipped[:,:2],atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(first[:,2],-flipped[:,2],atol=1e-6,rtol=1e-5)


def test_zero_frame_and_no_new_trainable_parameters():
    feature=torch.randn(2,4,16,16,requires_grad=True);q=torch.zeros(2,2,requires_grad=True)
    out=preoperative_frame_moments(feature,q)
    horizontal=preoperative_frame_moments(feature,torch.tensor([[1.,0.],[1.,0.]]))
    torch.testing.assert_close(out,horizontal);out.sum().backward()
    assert torch.isfinite(q.grad).all() and torch.isfinite(feature.grad).all()
    cli=['--excel_path','x','--data_root','x','--sia_prediction','axial_direct',
        '--conditions','pre_front_X','pre_front_Y','--fusion_method','film','--sample_steps','1']
    args=get_args(cli);stats={'sia':{'mean':[.2,.1],'std':[.3,.4]},'clinical':{'mean':[.5,.6],'std':[2.,3.]}}
    torch.manual_seed(5);a=build_model(args,stats)
    args.sia_feature_fusion='preop_frame';torch.manual_seed(5);b=build_model(args,stats)
    for (an,ap),(bn,bp) in zip(a.named_parameters(),b.named_parameters()):
        assert an==bn;torch.testing.assert_close(ap,bp)
    with torch.no_grad():b.sia_readout[-1].weight.normal_(0,.01)
    pre=torch.rand(1,1,32,32,requires_grad=True);post=torch.rand_like(pre,requires_grad=True);clinical=torch.tensor([[.2,-.1]],requires_grad=True)
    first=b(pre,post,torch.tensor([[.3,-.2]]),clinical=clinical,t=torch.tensor([.1]))
    second=b(pre,post+30,torch.tensor([[.3,-.2]]),clinical=clinical,t=torch.tensor([.9]))
    torch.testing.assert_close(first['q_clean'],second['q_clean'])
    gradients=torch.autograd.grad(first['q_clean'].sum(),(pre,clinical,post),allow_unused=True)
    assert gradients[2] is None and all(g is not None and torch.isfinite(g).all() and g.abs().sum()>0 for g in gradients[:2])


def test_frame_cli_contract():
    assert saved_config_value({},'sia_feature_fusion')=='lab'
    with pytest.raises(ValueError,match='Preoperative frame requires'):
        get_args(['--excel_path','x','--data_root','x','--sia_feature_fusion','preop_frame'])
