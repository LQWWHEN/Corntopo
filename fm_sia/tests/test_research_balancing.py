import copy
import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'cddpm'))
from task_balancing import MultiTaskBalancer
from train import get_args, saved_config_value

def losses(w):
    image=(w-1).square().sum()
    vector=(2*w+3).square().sum()
    sia=(w+.2).square().mean()
    return dict(loss_image=image,loss_vector=vector,loss_sia=sia,
                loss=image+vector+.1*sia)

def test_fixed_preserves_original_loss_and_gradients():
    w=torch.nn.Parameter(torch.tensor([.3,.8])); out=losses(w)
    original=torch.autograd.grad(out['loss'],w,retain_graph=True)[0]
    total,_=MultiTaskBalancer().combine(out)
    assert total is out['loss']
    torch.testing.assert_close(torch.autograd.grad(total,w)[0],original)

def test_gradnorm_gradient_isolation_and_resume():
    w=torch.nn.Parameter(torch.tensor([.3,.8]))
    b=MultiTaskBalancer(method='gradnorm')
    total,logs=b.combine(losses(w),[w])
    assert w.grad is None and b.weights.grad is not None
    assert logs['gradnorm_image']>0 and logs['gradnorm_sia_task']>0
    total.backward(); assert torch.isfinite(w.grad).all()
    b.step(); assert (b.weights>0).all()
    torch.testing.assert_close(b.weights.sum(),torch.tensor(2.))
    clone=MultiTaskBalancer(method='gradnorm');clone.load_state_dict(copy.deepcopy(b.state_dict()))
    for control in [b,clone]:
        control.combine(losses(w),[w]);control.step()
    torch.testing.assert_close(b.weights,clone.weights)

def test_validation_only_opt_in_preserves_old_defaults():
    base=['--excel_path','unused','--data_root','unused']
    assert not get_args(base).validation_only
    assert get_args(base+['--validation_only']).validation_only
    assert saved_config_value({},'validation_only') is False
