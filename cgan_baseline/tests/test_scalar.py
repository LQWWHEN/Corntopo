import copy
import numpy as np
import pytest
import torch
from dataloader import validate_table, fit_image_stats, CornealTopoDataset
from metrics import MetricAccumulator
from sia_metrics import SIAMetricAccumulator
from models import create_model
from train import evaluate,make_loader
from test_contracts import args_for,fixture_data


def test_scalar_metrics_exact_vector_magnitude_parity():
    gen=torch.Generator().manual_seed(7)
    p=torch.rand(3,1,16,16,generator=gen);y=torch.rand(3,1,16,16,generator=gen)
    mask=torch.ones_like(y);mask[:,:,:3]=0
    qp=torch.tensor([[.1,.2],[.3,.4],[-.1,.2]])
    qt=torch.tensor([[.2,.3],[.2,.2],[-.3,.1]])
    sp=qp.norm(dim=1)
    original=MetricAccumulator(5);original.update(p,y,mask,qp,qt,sp)
    scalar=SIAMetricAccumulator(5);scalar.update(p,y,mask,q_true=qt,s_pred=sp)
    vector=SIAMetricAccumulator(5);vector.update(p,y,mask,qp,qt,sp)
    assert vector.compute()==original.compute()
    assert scalar.compute()=={k:v for k,v in original.compute().items()
                             if not k.startswith(('Vector_','DeltaX_','DeltaY_'))}
    with pytest.raises(ValueError,match='mix'):scalar.update(p,y,mask,qp,qt,sp)


def test_scalar_only_labels_and_d4_invariance(tmp_path):
    df,folds=fixture_data(tmp_path);df=df.drop(columns=['delta_front_X','delta_front_Y'])
    a=args_for(tmp_path,'--sia_mode','scalar','--augmentation','d4')
    report=validate_table(df,a)
    assert report['sia_label_source']=='front_SIA_D'
    image=fit_image_stats(a,folds[0]['train'])
    from transforms import DataTransform
    ds=CornealTopoDataset(a,a.excel_path,a.data_root,dataframe=df,labels=folds[0]['train'],
                         image_size=128,image_stats=image,transforms=DataTransform())
    for _ in range(4):
        item=ds[0]
        assert item['sia_vector'].numel()==0
        assert item['sia_magnitude'].item()==pytest.approx(df.set_index('label_index').loc[item['label'],'front_SIA_D'])
    ds.transforms=None
    model=create_model(a,{'sia':None})
    metrics=evaluate(model,make_loader(ds,a,torch.device('cpu')),torch.device('cpu'),a,{'image':image})
    assert 'SIA_MAE' in metrics and 'Vector_MAE' not in metrics
    a.sia_mode='vector'
    with pytest.raises(ValueError,match='both physical'):validate_table(df,a)
    a.sia_mode='scalar';df.loc[0,'front_SIA_D']=-1
    with pytest.raises(ValueError,match='nonnegative'):validate_table(df,a)


@pytest.mark.parametrize('loss',['l1','huber'])
def test_scalar_loss_formula_and_no_vector_weight(tmp_path,loss):
    a=args_for(tmp_path,'--sia_mode','scalar','--sia_loss',loss,'--sia_ref','2',
        '--lambda_sia','3','--lambda_vector','100','--lambda_gan','0','--lambda_L1','0','--lambda_grad','0')
    m=create_model(a,{'sia':None}).eval();pre=torch.zeros(2,1,128,128)
    out={'pred':pre,'sia':torch.tensor([1.,3.],requires_grad=True)}
    true=torch.tensor([0.,1.]);mask=torch.ones_like(pre)
    result=m.generator_loss(pre,pre,out,mask,s_true=true)
    error=torch.tensor([.5,1.])
    expected=error.abs().mean() if loss=='l1' else torch.nn.functional.huber_loss(error,torch.zeros_like(error))
    assert result['loss_sia'].item()==pytest.approx(expected.item())
    assert result['loss_vector'].item()==0
    assert result['loss'].item()==pytest.approx(3*expected.item())
    result['loss'].backward();assert out['sia'].grad is not None


def test_scalar_default_and_checkpoint_mode_guard(tmp_path):
    a=args_for(tmp_path);assert a.sia_mode=='scalar'
    scalar=create_model(a,{'sia':None})
    a=copy.deepcopy(a);a.sia_mode='vector'
    vector=create_model(a,{'sia':{'mean':[0.,0.],'std':[1.,1.]}})
    with pytest.raises(RuntimeError):vector.load_state_dict(scalar.state_dict())
    a=args_for(tmp_path,'--model','CPTP');assert a.sia_mode=='off'
