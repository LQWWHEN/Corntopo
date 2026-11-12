"""Deterministic SIA acceptance: preoperative-only graph and saved inference."""
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'cddpm'))
from train import get_args, build_model, fit_sia_stats, saved_config_value, PIPELINE_VERSION
from predict import load_checkpoint
from task_balancing import build_task_balancer

torch.set_num_threads(1)


def arguments():
    return get_args(['--excel_path','unused','--data_root','unused',
        '--sia_prediction','direct','--sia_mode','scalar','--lambda_sia','0',
        '--conditions','pre_age_years','--fusion_method','film','--sample_steps','2'])


def test_direct_defaults_and_training_only_median():
    args=arguments()
    frame=pd.DataFrame(dict(label_index=[1,2,3],front_SIA_D=[.2,.6,999.]))
    stats=fit_sia_stats(args,frame,[1,2])
    assert stats['median']==pytest.approx([.4])
    frame.loc[2,'front_SIA_D']=999999.
    assert fit_sia_stats(args,frame,[1,2])==stats
    assert saved_config_value({},'sia_prediction')=='flow'
    assert get_args(['--excel_path','x','--data_root','x']).sia_prediction=='flow'
    with pytest.raises(ValueError,match='Direct SIA requires'):
        get_args(['--excel_path','x','--data_root','x','--sia_prediction','direct'])


def test_direct_preoperative_only_gradient_and_checkpoint(tmp_path):
    torch.manual_seed(10)
    args=arguments()
    stats={'sia':{'mean':[.5],'std':[.3],'median':[.4]}}
    model=build_model(args,stats)
    assert model.unet.sia_interaction is None
    pre=torch.rand(1,1,16,16,requires_grad=True)
    post=torch.rand_like(pre,requires_grad=True)
    clinical=torch.tensor([[.2]],requires_grad=True)
    target=torch.tensor([.8])
    first=model(pre,post,clinical=clinical,sia_magnitude=target)
    torch.testing.assert_close(first['sia_clean'],torch.tensor([.4]))
    # Nonzero readout verifies actual shared gradients after the initial step.
    with torch.no_grad(): model.sia_readout[-1].weight.normal_(0,.01)
    a=model(pre,post,clinical=clinical,sia_magnitude=target,t=torch.tensor([.1]))
    b=model(pre,post+100,clinical=clinical,sia_magnitude=target,t=torch.tensor([.9]),noise=torch.randn(1,1))
    torch.testing.assert_close(a['sia_clean'],b['sia_clean'])
    gradients=torch.autograd.grad(a['sia_clean'].sum(),(pre,clinical,post),retain_graph=True,allow_unused=True)
    assert gradients[2] is None
    assert all(g is not None and torch.isfinite(g).all() and g.abs().sum()>0 for g in gradients[:2])
    total,_=build_task_balancer(args,model,'cpu').combine(a)
    torch.testing.assert_close(total,a['loss_image']+args.lambda_scalar*a['loss_scalar'])
    total.backward()
    for parameter in [model.condition_encoder.in_conv.weight,model.sia_readout[-1].weight,model.unet.out.weight]:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum()>0
    model.eval()
    x=pre.detach();c=clinical.detach()
    prediction=model.sample(x,clinical=c,noise=torch.zeros(1,1))
    alternative=model.sample(x,clinical=c,noise=torch.full((1,1),100.))
    torch.testing.assert_close(prediction['sia'],alternative['sia'])
    torch.testing.assert_close(prediction['pred'],alternative['pred'])
    assert (prediction['sia']>=0).all() and 'sia_vector' not in prediction
    path=tmp_path/'direct.pth'
    torch.save(dict(pipeline_version=PIPELINE_VERSION,config=vars(args),stats=stats,
                    model=model.state_dict()),path)
    restored,_,_,_=load_checkpoint(path)
    after=restored.sample(x,clinical=c)
    torch.testing.assert_close(after['sia'],prediction['sia'])
    torch.testing.assert_close(after['pred'],prediction['pred'])
