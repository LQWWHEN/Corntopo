import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'cddpm'))
from clinical_features import EYE_SIGN
from model.eye_frame import reflect_left_image,reflect_left_vector
from predict import load_checkpoint,predict_case
from train import get_args,fit_sia_stats,build_model,PIPELINE_VERSION,seed_everything

torch.set_num_threads(1)


def arguments(backbone='fno',extra=()):
    return get_args(['--excel_path','unused','--data_root','unused','--sia_prediction','polar_direct',
        '--eye_frame','right','--lambda_sia','1','--conditions','pre_age_years',EYE_SIGN,
        '--fusion_method','film','--sample_steps','2','--image_backbone',backbone,
        '--fno_modes','4','4','--fno_hidden_channels','8','--fno_layers','1',*extra])


def statistics(args):
    frame=pd.DataFrame(dict(label_index=[1,2,3],delta_front_X=[1.,0.,999.],
                            delta_front_Y=[0.,1.,999.],pre_eye_sign=[1.,-1.,-1.]))
    return frame,dict(sia=fit_sia_stats(args,frame,[1,2]),clinical=dict(mean=[50.,.1],std=[10.,.9]))


def test_polar_canonical_initialization_uses_training_targets_and_preserves_world_statistics():
    args=arguments();frame,stats=statistics(args)
    sia=stats['sia']
    assert sia['polar_median_magnitude']==1 and sia['polar_axis_training_count']==2
    assert sia['polar_initial_phase']==pytest.approx(-math.pi/4)
    assert sia['mean']==pytest.approx([.5,.5]) and sia['canonical_mean']==pytest.approx([.5,-.5])
    native=arguments(extra=['--eye_frame','native'])
    assert fit_sia_stats(native,frame,[1,2])['polar_initial_phase']==pytest.approx(math.pi/4)
    frame.loc[2,['delta_front_X','delta_front_Y',EYE_SIGN]]=[-1e6,1e6,1.]
    assert fit_sia_stats(args,frame,[1,2])==sia


@pytest.mark.parametrize('backbone',['unet','fno'])
def test_polar_eye_frame_matches_manual_geometry_and_has_no_postoperative_sia_input(backbone):
    if backbone=='fno':pytest.importorskip('neuralop')
    args=arguments(backbone);_,stats=statistics(args)
    model=build_model(args,stats)
    pre=torch.rand(2,1,32,32,requires_grad=True);post=torch.rand_like(pre,requires_grad=True)
    clinical=torch.tensor([[.2,1.],[-.1,-1.1/.9]],requires_grad=True)
    sign=model.eye_sign(clinical);torch.testing.assert_close(sign,torch.tensor([1.,-1.]))
    target=torch.tensor([[.4,-.2],[.3,.8]]);mask=(torch.rand_like(pre)>.2).float();t=torch.tensor([.2,.7])
    initial=model(pre,post,target,clinical=clinical,mask=mask,t=t)
    expected=torch.tensor([[1.,-1.],[1.,1.]])/math.sqrt(2)
    torch.testing.assert_close(initial['q_clean'],expected)
    torch.testing.assert_close(initial['sia_clean'],torch.ones(2))
    with torch.no_grad():model.sia_readout[-1].weight.normal_(0,.01)
    args.eye_frame='native';native=build_model(args,stats)
    native.load_state_dict({k:v for k,v in model.state_dict().items() if k not in ('eye_mean','eye_std')})
    world=model(pre,post,target,clinical=clinical,mask=mask,t=t)
    canonical=native(reflect_left_image(pre,sign),reflect_left_image(post,sign),reflect_left_vector(target,sign),
        clinical=clinical,mask=reflect_left_image(mask,sign),t=t)
    for key in ('loss','loss_image','loss_sia','loss_axis'):
        torch.testing.assert_close(world[key],canonical[key])
    torch.testing.assert_close(world['velocity_pred'],reflect_left_image(canonical['velocity_pred'],sign))
    torch.testing.assert_close(world['q_clean'],reflect_left_vector(canonical['q_clean'],sign))
    torch.testing.assert_close(world['loss'],world['loss_image']+world['loss_sia']+world['loss_axis'])
    changed=model(pre,post+10,target*3,clinical=clinical,t=1-t)
    torch.testing.assert_close(world['q_clean'],changed['q_clean'])
    gradients=torch.autograd.grad(world['q_clean'].sum(),(pre,clinical,post),retain_graph=True,allow_unused=True)
    assert gradients[2] is None
    assert all(g is not None and torch.isfinite(g).all() and g.abs().sum()>0 for g in gradients[:2])
    world['loss'].backward()
    assert model.condition_encoder.in_conv.weight.grad.abs().sum()>0
    model.eval();native.eval()
    output=model.sample(pre.detach(),clinical=clinical.detach())
    reference=native.sample(reflect_left_image(pre.detach(),sign),clinical=clinical.detach())
    torch.testing.assert_close(output['pred'],reflect_left_image(reference['pred'],sign))
    torch.testing.assert_close(output['sia_vector'],reflect_left_vector(reference['sia_vector'],sign))
    torch.testing.assert_close(output['sia'],output['sia_vector'].norm(dim=-1))
    torch.testing.assert_close(output['sia_state'],(output['sia_vector']-model.sia_mean)/model.sia_std)


def test_canonical_polar_saved_reload_keeps_world_image_magnitude_and_axis(tmp_path):
    pytest.importorskip('neuralop');args=arguments();_,stats=statistics(args)
    torch.set_num_threads(1);seed_everything(args.seed,strict=args.strict_determinism)
    model=build_model(args,stats).eval()
    with torch.no_grad():model.sia_readout[-1].weight.normal_(0,.01)
    pre=torch.rand(2,1,32,32);clinical=torch.tensor([[0.,1.],[0.,-1.1/.9]])
    before=model.sample(pre,clinical=clinical)
    path=tmp_path/'temporary.pth'
    torch.save(dict(pipeline_version=PIPELINE_VERSION,config=vars(args),stats=stats,model=model.state_dict(),ema=None),path)
    restored,_,_,_=load_checkpoint(path,'cpu',cpu_threads=1)
    after=restored.sample(pre,clinical=clinical)
    for key in ('pred','sia','sia_vector','sia_state'):torch.testing.assert_close(before[key],after[key],rtol=0,atol=0)
    path.unlink()


def test_preoperative_only_prediction_api_returns_os_axis_in_acquisition_frame(tmp_path):
    pytest.importorskip('neuralop');args=arguments(extra=['--image_size','32']);_,stats=statistics(args)
    stats['image']=dict(minimum=0.,scale=1.)
    model=build_model(args,stats).eval();pre=np.ones((32,32),dtype=np.float32)
    path=tmp_path/'pre.npy';np.save(path,pre)
    result,arrays=predict_case(model,args,stats,torch.device('cpu'),path,[50.,-1.])
    assert result['pred_SIA_D']==pytest.approx(1.)
    assert result['pred_axis_deg']==pytest.approx(22.5)
    assert result['pred_delta_X']==pytest.approx(1/math.sqrt(2))
    assert result['pred_delta_Y']==pytest.approx(1/math.sqrt(2))
    assert arrays['post_prediction_raw'].shape==(32,32)


@pytest.mark.parametrize('extra',[
    ['--sia_rank_weight','.1'],['--sia_velocity_coupling'],
    ['--conditions',EYE_SIGN,'pre_front_X'],['--augmentation','d4'],
])
def test_canonical_polar_rejects_unimplemented_geometry_combinations(extra):
    with pytest.raises(ValueError):arguments(extra=extra)
