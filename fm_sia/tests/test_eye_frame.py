import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'cddpm'))
from clinical_features import prepare_clinical_features, PRE_EYE, EYE_SIGN
from model.eye_frame import reflect_left_image, reflect_left_vector
from model.fm_common import masked_mse
from metrics import axis_details
from train import get_args, build_model, fit_sia_stats, saved_config_value

torch.set_num_threads(1)


def args_for_frame(backbone='unet'):
    return get_args(['--excel_path','unused','--data_root','unused',
                     '--sia_prediction','axial_direct','--eye_frame','right',
                     '--conditions','pre_age_years',EYE_SIGN,'--fusion_method','film',
                     '--lambda_sia','1','--sample_steps','2','--image_backbone',backbone,
                     '--fno_modes','4','4','--fno_hidden_channels','8','--fno_layers','1'])


def test_eye_mapping_and_initialization_use_only_pre_eye_and_training_targets():
    frame = pd.DataFrame({'label_index':[1,2,3], PRE_EYE:['OD',' os ','OS'],
                          EYE_SIGN:[999.,999.,999.], 'delta_front_X':[.2,.4,999.],
                          'delta_front_Y':[.4,.6,999.]})
    original = frame.copy(deep=True)
    mapped = prepare_clinical_features(frame,[EYE_SIGN])
    pd.testing.assert_frame_equal(frame,original)
    np.testing.assert_array_equal(mapped[EYE_SIGN],[1.,-1.,-1.])
    args = args_for_frame()
    stats = fit_sia_stats(args,mapped,[1,2])
    assert stats['mean'] == pytest.approx([.3,.5])
    assert stats['canonical_mean'] == pytest.approx([.3,-.1])
    mapped.loc[2,['delta_front_X','delta_front_Y',EYE_SIGN]] = [-1e6,1e6,1.]
    assert fit_sia_stats(args,mapped,[1,2]) == stats
    frame['delta_front_Y'] = -999.
    np.testing.assert_array_equal(prepare_clinical_features(frame,[EYE_SIGN])[EYE_SIGN],[1.,-1.,-1.])
    frame.loc[0,PRE_EYE] = 'unknown'
    with pytest.raises(ValueError,match='OD or OS'):
        prepare_clinical_features(frame,[EYE_SIGN])


def test_reflection_is_involution_and_transforms_periodic_axis_and_mask():
    sign = torch.tensor([1.,-1.])
    image = torch.arange(2*3*5*7,dtype=torch.float32).reshape(2,3,5,7)
    flipped = reflect_left_image(image,sign)
    torch.testing.assert_close(flipped[0],image[0])
    torch.testing.assert_close(flipped[1],image[1].flip(-1))
    torch.testing.assert_close(reflect_left_image(flipped,sign),image)
    theta = np.array([15.,37.]);q = torch.tensor(np.column_stack((np.cos(2*np.deg2rad(theta)),np.sin(2*np.deg2rad(theta)))),dtype=torch.float32)
    mirrored = reflect_left_vector(q,sign)
    torch.testing.assert_close(reflect_left_vector(mirrored,sign),q)
    torch.testing.assert_close(mirrored.norm(dim=-1),q.norm(dim=-1))
    np.testing.assert_allclose(axis_details(mirrored.numpy(),mirrored.numpy())['pred'],[15.,143.],atol=1e-5)
    target = image*.01;mask = (image[:,:1] % 3 != 0).float()
    torch.testing.assert_close(masked_mse(image,target,mask),
        masked_mse(reflect_left_image(image,sign),reflect_left_image(target,sign),reflect_left_image(mask,sign)))


@pytest.mark.parametrize('backbone',['unet','fno'])
def test_canonical_joint_graph_returns_world_predictions_and_no_postoperative_sia(backbone):
    if backbone == 'fno':
        pytest.importorskip('neuralop')
    args = args_for_frame(backbone)
    stats = {'sia':{'mean':[.2,.4],'std':[.5,.6],'canonical_mean':[.3,-.1]},
             'clinical':{'mean':[50.,.1],'std':[10.,.9]}}
    torch.manual_seed(7)
    model = build_model(args,stats)
    clinical = torch.tensor([[.2,1.],[-.1,-1.1/.9]],requires_grad=True)
    sign = model.eye_sign(clinical)
    torch.testing.assert_close(sign,torch.tensor([1.,-1.]))
    pre = torch.rand(2,1,32,32,requires_grad=True)
    post = torch.rand_like(pre,requires_grad=True)
    q = torch.tensor([[.4,-.2],[.3,.8]])
    mask = (torch.rand_like(pre)>.2).float()
    t = torch.tensor([.2,.7])
    initial = model(pre,post,q,clinical=clinical,mask=mask,t=t)
    torch.testing.assert_close(initial['q_clean'],torch.tensor([[.3,-.1],[.3,.1]]))
    with torch.no_grad(): model.sia_readout[-1].weight.normal_(0,.01)
    args.eye_frame = 'native'
    native = build_model(args,stats)
    native.load_state_dict({k:v for k,v in model.state_dict().items() if k not in ('eye_mean','eye_std')})
    world = model(pre,post,q,clinical=clinical,mask=mask,t=t)
    canonical = native(reflect_left_image(pre,sign),reflect_left_image(post,sign),reflect_left_vector(q,sign),
                       clinical=clinical,mask=reflect_left_image(mask,sign),t=t)
    for key in ('loss','loss_image','loss_vector','loss_sia'):
        torch.testing.assert_close(world[key],canonical[key])
    torch.testing.assert_close(world['velocity_pred'],reflect_left_image(canonical['velocity_pred'],sign))
    torch.testing.assert_close(world['q_clean'],reflect_left_vector(canonical['q_clean'],sign))
    changed = model(pre,post+10,q,clinical=clinical,t=1-t)
    torch.testing.assert_close(world['q_clean'],changed['q_clean'])
    gradients = torch.autograd.grad(world['q_clean'].sum(),(pre,clinical,post),retain_graph=True,allow_unused=True)
    assert gradients[2] is None
    assert all(g is not None and torch.isfinite(g).all() and g.abs().sum()>0 for g in gradients[:2])
    world['loss'].backward()
    assert model.condition_encoder.in_conv.weight.grad.abs().sum()>0
    model.eval();native.eval()
    a = model.sample(pre.detach(),clinical=clinical.detach())
    b = native.sample(reflect_left_image(pre.detach(),sign),clinical=clinical.detach())
    torch.testing.assert_close(a['pred'],reflect_left_image(b['pred'],sign))
    torch.testing.assert_close(a['sia_vector'],reflect_left_vector(b['sia_vector'],sign))
    torch.testing.assert_close(a['sia'],a['sia_vector'].norm(dim=-1))
    torch.testing.assert_close(a['sia_state'],(a['sia_vector']-model.sia_mean)/model.sia_std)
    assert not args.save_checkpoints


def test_eye_frame_cli_requires_matched_clinical_geometry_and_preserves_old_default():
    assert saved_config_value({},'eye_frame') == 'native'
    assert get_args(['--excel_path','x','--data_root','x']).eye_frame == 'native'
    with pytest.raises(ValueError,match='Right-eye frame requires'):
        get_args(['--excel_path','x','--data_root','x','--eye_frame','right'])
    with pytest.raises(ValueError,match='scalar clinical'):
        get_args(['--excel_path','x','--data_root','x','--sia_prediction','axial_direct',
                  '--eye_frame','right','--conditions',EYE_SIGN,'pre_front_X'])
