import sys
from pathlib import Path
import pytest
import torch
pytest.importorskip('neuralop')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'cddpm'))
from train import get_args,build_model,cpu_weight_snapshot,saved_config_value

torch.set_num_threads(1)


def model_and_args():
    args=get_args(['--excel_path','x','--data_root','x','--sia_prediction','axial_direct',
        '--image_backbone','fno','--conditions','pre_age_years','--fusion_method','film',
        '--lambda_sia','1','--sample_steps','2','--fno_modes','4','4',
        '--fno_hidden_channels','8','--fno_layers','2'])
    model=build_model(args,{'sia':{'mean':[.2,.1],'std':[.3,.4]}})
    return model,args


def test_joint_fno_gradients_and_preoperative_sia_dependency():
    torch.manual_seed(9);model,args=model_and_args()
    with torch.no_grad():model.sia_readout[-1].weight.normal_(0,.01)
    pre=torch.rand(1,1,32,32,requires_grad=True);post=torch.rand_like(pre,requires_grad=True)
    clinical=torch.tensor([[.2]],requires_grad=True)
    out=model(pre,post,torch.tensor([[.5,-.2]]),clinical=clinical,t=torch.tensor([.4]))
    dq=torch.autograd.grad(out['q_clean'].sum(),(pre,clinical,post),retain_graph=True,allow_unused=True)
    assert dq[2] is None and all(g is not None and g.abs().sum()>0 for g in dq[:2])
    out['loss'].backward()
    for layer in model.unet.condition_projection:
        assert torch.isfinite(layer.weight.grad).all() and layer.weight.grad.abs().sum()>0
    complex_grad=[p.grad for p in model.unet.fno.parameters() if p.is_complex()]
    assert complex_grad and all(g is not None and torch.isfinite(g).all() and g.abs().sum()>0 for g in complex_grad)
    for p in [model.condition_encoder.in_conv.weight,model.sia_readout[-1].weight]:
        assert torch.isfinite(p.grad).all() and p.grad.abs().sum()>0


def test_cached_condition_and_memory_weight_selection_roundtrip():
    torch.manual_seed(3);model,args=model_and_args();model.eval()
    pre=torch.rand(1,1,32,32);clinical=torch.tensor([[.2]])
    with torch.no_grad():
        cond=model.condition_encoder(pre);cached=model.unet.prepare_condition(cond,clinical)
        for time in [0.,.7]:
            t=torch.tensor([time]);a=model.unet(pre,t,cond,clinical);b=model.unet(pre,t,cached,clinical)
            torch.testing.assert_close(a,b)
        expected=model.sample(pre,clinical=clinical)
    snapshot=cpu_weight_snapshot(model)
    assert snapshot and all(torch.is_tensor(v) for v in snapshot.values())
    assert '_metadata' not in snapshot and not args.save_checkpoints
    restored,_=model_and_args();restored.load_state_dict(snapshot,strict=True);restored.eval()
    actual=restored.sample(pre,clinical=clinical)
    for k,v in expected.items():torch.testing.assert_close(actual[k],v)
    assert actual['pred'].shape==pre.shape and actual['sia_vector'].shape==(1,2)
    torch.testing.assert_close(actual['sia'],actual['sia_vector'].norm(dim=-1))
    changed=dict(snapshot,unexpected=torch.zeros(1))
    with pytest.raises(RuntimeError,match='Unexpected key'):restored.load_state_dict(changed,strict=True)


def test_backbone_scope_and_historical_default():
    assert saved_config_value({},'image_backbone')=='unet'
    with pytest.raises(ValueError,match='Conditional FNO backbone requires'):
        get_args(['--excel_path','x','--data_root','x','--image_backbone','fno'])
