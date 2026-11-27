from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'cddpm'))
from clinical_features import PRE_EYE, PRE_AXIS, PRE_ASTIG, PRE_VECTOR, EYE_SIGN, prepare_clinical_features
from dataloader import feature_stats
from model.preop_vector_source import fit_reference, crossfit_references
from predict import predict_case
from train import build_model, fit_sia_stats, forward_batch, get_args, make_dataset


def test_patient_excluded_clinical_source_and_full_endpoint(tmp_path):
    torch.set_num_threads(1)
    flags=['--excel_path','unused','--data_root',str(tmp_path),'--fusion_method','film',
           '--sample_steps','2','--image_size','32','--conditions','pre_age_years']
    args=get_args([*flags,EYE_SIGN,*PRE_VECTOR,'--preop_vector_source'])
    table=pd.DataFrame({'label_index':np.arange(8),PRE_EYE:['OS','OD']*4,
                        PRE_ASTIG:np.linspace(.2,1.6,8),PRE_AXIS:np.arange(8)*11.,
                        'pre_age_years':np.arange(8)+40.})
    for side in ('pre','post'):
        table[side+'_PATIENT_PatientID']=[1,1,2,2,3,3,4,4]
        for field in ('LastName','FirstName','DateOfBirth'):
            table[side+'_PATIENT_'+field]=None
    table=prepare_clinical_features(table,args.conditions)
    pre=table[list(PRE_VECTOR)].to_numpy()
    eye=(table[EYE_SIGN].to_numpy()>0).astype(int)
    target=np.array([[.3,.1],[-.15,.2]])[eye]-.25*pre
    labels=list(range(6))
    table[args.sia_columns]=target
    assert fit_sia_stats(args,table,labels)['preop_reference']['slope']==pytest.approx(-.25)
    # A patient's other eye cannot influence its training reference.
    initial=crossfit_references(table.iloc[:6],args.sia_columns)
    table.loc[1,args.sia_columns]=[3.,4.]
    crossfit=crossfit_references(table.iloc[:6],args.sia_columns)
    for label in (0,1):
        np.testing.assert_allclose(crossfit[label],initial[label],atol=1e-7)
    sia=fit_sia_stats(args,table,labels)
    table.loc[6:,args.sia_columns]=[-900.,800.]
    assert fit_sia_stats(args,table,labels)==sia
    fit_zero=fit_reference(np.ones((3,2)),target[:3],np.ones(3))
    assert fit_zero['slope']==0 and fit_zero['target'][0]==fit_zero['target'][1]
    stats=dict(sia=sia,clinical=feature_stats(table,labels,args.conditions),image=dict(minimum=0.,scale=1.))
    torch.manual_seed(13)
    base=build_model(get_args(flags),stats)
    torch.manual_seed(13)
    model=build_model(args,stats)
    assert dict(base.named_parameters()).keys()==dict(model.named_parameters()).keys()
    for key,p in base.named_parameters():
        torch.testing.assert_close(model.get_parameter(key),p,atol=0,rtol=0)
    location=tmp_path/'raw/FRONT'
    location.mkdir(parents=True)
    for i in range(8):
        for phase in ('pre','post'):
            np.save(location/f'{i}_{phase}.npy',np.ones((32,32),np.float32))
    training=make_dataset(args,table,labels,stats,training=True)
    valid=make_dataset(args,table,[6,7],stats)
    assert 'training_reference' not in valid[0]
    batch=next(iter(DataLoader(training,batch_size=2)))
    np.testing.assert_allclose(batch['training_reference'].numpy(),[initial[0],initial[1]])
    learned,center=model.conditions_and_center(batch['condition'])
    torch.testing.assert_close(learned,batch['condition'][:,:1])
    assert not torch.allclose(center,batch['training_reference'])
    class ConstantField(nn.Module):
        def __init__(self, speed):
            super().__init__()
            self.speed=speed
        def forward(self,x,t,condition,clinical=None,sia_state=None):
            assert clinical.shape[-1]==1
            return torch.zeros_like(x), self.speed.expand_as(sia_state)
    speed=torch.tensor([.12,-.3])
    model.condition_encoder=nn.Identity()
    model.unet=ConstantField(speed)
    batch['sia_vector']=batch['training_reference']+model.sia_std*speed
    assert forward_batch(model,batch,'cpu')['loss'].item()<1e-12
    for draw in (torch.ones((2,2)),-torch.ones((2,2))):
        out=model.sample(batch['input'],clinical=batch['condition'],noise=draw)
        torch.testing.assert_close(out['sia_vector'],center+model.sia_std*speed)
        torch.testing.assert_close(out['sia'],out['sia_vector'].norm(dim=-1))
    info,_=predict_case(model,args,stats,'cpu',location/'0_pre.npy',
                        table.loc[0,args.conditions].to_numpy(float),sample_id='synthetic')
    np.testing.assert_allclose([info['pred_delta_X'],info['pred_delta_Y']],out['sia_vector'][0],atol=1e-6)
