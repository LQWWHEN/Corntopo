import sys
from pathlib import Path
import pytest
import torch
from torch import nn
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'cddpm'))
from model.axial_sia import AxialSIAFlowMatching
from train import get_args, build_model, saved_config_value

torch.set_num_threads(1)


class ConstantEncoder(nn.Module):
    def forward(self, x):
        return [x.new_ones((len(x),512,4,4))]


class PhysicalVelocity(nn.Module):
    sia_interaction = None
    clinical_encoder = None

    def __init__(self, scale):
        super().__init__()
        self.physical_velocity = nn.Parameter(torch.tensor(.7))
        self.scale = scale

    def forward(self, x, t, cond, clinical=None):
        return torch.ones_like(x)*self.physical_velocity/self.scale


def test_physical_loss_and_gradient_invariant_to_normalization_range():
    # Same physical task under two artificial training-fold affine encodings.
    # Missing pixels contain a large distractor; only the aperture contributes.
    pre = torch.tensor([[[[42.,43.],[44.,45.]]]])
    post = pre+torch.tensor([[[[.2,-.4],[.3,100.]]]])
    mask = torch.tensor([[[[1.,1.],[1.,0.]]]])
    results=[]
    for minimum,scale in [(20.,30.),(10.,60.)]:
        model=AxialSIAFlowMatching(ConstantEncoder(),PhysicalVelocity(scale),[.2,.1],[1.,1.],
                                  image_loss_units='diopter',image_scale=scale,sia_ref=1.)
        out=model((pre-minimum)/scale,(post-minimum)/scale,torch.tensor([[.2,.1]]),mask=mask,t=torch.tensor([.5]))
        out['loss_image'].backward()
        results.append((out['loss_image'].detach(),model.unet.physical_velocity.grad))
    expected=((torch.tensor([.2,-.4,.3])-.7)**2).mean()
    for value,gradient in results:
        torch.testing.assert_close(value,expected,atol=2e-6,rtol=1e-5)
        torch.testing.assert_close(gradient,2*(.7-torch.tensor([.2,-.4,.3])).mean(),atol=2e-6,rtol=1e-5)


def test_loss_units_preserve_initial_weights_sampling_and_sia():
    cli=['--excel_path','x','--data_root','x','--sia_prediction','axial_direct','--lambda_sia','1','--sample_steps','1']
    stats={'sia':{'mean':[.2,.1],'std':[.3,.4]},'image':{'minimum':20.,'scale':30.}}
    args=get_args(cli)
    torch.manual_seed(4);a=build_model(args,stats)
    args.image_loss_units='diopter'
    torch.manual_seed(4);b=build_model(args,stats)
    assert a.state_dict().keys()==b.state_dict().keys()
    for key,value in a.state_dict().items():torch.testing.assert_close(value,b.state_dict()[key])
    pre=torch.rand(1,1,32,32);post=torch.rand_like(pre);q=torch.tensor([[.5,-.3]])
    oa=a(pre,post,q,t=torch.tensor([.2]));ob=b(pre,post,q,t=torch.tensor([.2]))
    for key in ['loss_vector','loss_sia','q_clean','velocity_pred']:torch.testing.assert_close(oa[key],ob[key])
    torch.testing.assert_close(ob['loss_image'],900*oa['loss_image'])
    a.eval();b.eval()
    for key,value in a.sample(pre).items():torch.testing.assert_close(value,b.sample(pre)[key])


def test_physical_front_only_and_historical_config_default():
    assert saved_config_value({},'image_loss_units')=='normalized'
    with pytest.raises(ValueError,match='Diopter image loss requires'):
        get_args(['--excel_path','x','--data_root','x','--image_loss_units','diopter'])
    with pytest.raises(ValueError,match='Diopter image loss requires'):
        get_args(['--excel_path','x','--data_root','x','--sia_prediction','axial_direct',
                  '--image_loss_units','diopter','--input_maps','BACK','--output_maps','BACK'])
