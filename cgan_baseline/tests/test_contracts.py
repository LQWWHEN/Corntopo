import argparse
import copy
import json
import numpy as np
import pandas as pd
import pytest
import torch
from dataloader import feature_stats, fit_image_stats, CornealTopoDataset
from metrics import MetricAccumulator
from models import create_model
from options.train_options import TrainOptions
from patient_cv import _labels_and_groups, load_cv_splits
from transforms import DataTransform
from train import metric_reference_images, train_one_epoch


def args_for(tmp_path, *extra):
    return TrainOptions().parse(['--excel_path',str(tmp_path/'data.csv'),'--data_root',str(tmp_path),
        '--split_file',str(tmp_path/'split.json'),'--save_dir',str(tmp_path/'results'),
        '--device','cpu','--num_workers','0','--epochs','2','--batch_size','8',
        '--ngf','4','--ndf','4',*extra])


def fixture_data(tmp_path):
    directory=tmp_path/'raw'/'FRONT';directory.mkdir(parents=True)
    yy,xx=np.mgrid[:128,:128]; valid=(xx-63.5)**2+(yy-63.5)**2<60**2
    rows=[]
    for i in range(20):
        label=101+7*i;patient=i//2
        row=dict(label_index=label,age=30+patient,delta_front_X=.1*(patient+1),delta_front_Y=-.05*(patient+1))
        row['front_SIA_D']=float(np.hypot(row['delta_front_X'],row['delta_front_Y']))
        for side in ('pre','post'):
            row.update({f'{side}_PATIENT_PatientID':f'p{patient}', f'{side}_PATIENT_LastName':f'l{patient}',
                        f'{side}_PATIENT_FirstName':'f', f'{side}_PATIENT_DateOfBirth':'1980-01-01'})
            raw=(40+xx*.01+yy*.005+i*.02+(side=='post')*.05).astype(np.float32)
            raw[~valid]=np.nan
            np.save(directory/f'{label}_{side}.npy',raw)
        rows.append(row)
    df=pd.DataFrame(rows);df.to_csv(tmp_path/'data.csv',index=False)
    labels,groups,fp=_labels_and_groups(df)
    folds=[]
    for k in range(10):
        test=labels[2*k:2*k+2];v=(k+1)%10;val=labels[2*v:2*v+2]
        folds.append(dict(fold=k+1,train=[x for x in labels if x not in test+val],val=val,test=test))
    split=dict(version=1,seed=42,n_splits=10,dataset_fingerprint=fp,folds=folds)
    (tmp_path/'split.json').write_text(json.dumps(split))
    return df,folds


def test_metrics_domains_pooling_and_batch_invariance():
    p=torch.zeros(2,1,16,16); y=p.clone(); mask=torch.zeros_like(p); mask[:,:,4:12,4:12]=1
    p[0]=.1;p[1]=.2
    acc=MetricAccumulator(5); acc.update(p,y,mask)
    result=acc.compute()
    assert result['PSNR_valid']==pytest.approx(-10*np.log10(.025))
    assert result['MAE_raw_valid']==pytest.approx(.75)
    other=MetricAccumulator(5)
    for i in range(2):other.update(p[i:i+1],y[i:i+1],mask[i:i+1])
    assert other.compute()==result
    p[mask==0]=10
    full=MetricAccumulator(5);full.update(p,y,mask)
    assert full.compute()['MAE_valid']==result['MAE_valid']
    assert full.compute()['MAE_full']>result['MAE_full']
    assert full.compute()['SSIM_valid']==pytest.approx(result['SSIM_valid'])


def test_split_identity_and_fail_closed(tmp_path):
    df,folds=fixture_data(tmp_path)
    assert load_cv_splits(df.sample(frac=1,random_state=1),tmp_path/'split.json')==folds
    with pytest.raises(FileNotFoundError):load_cv_splits(df,tmp_path/'missing.json')
    assert not (tmp_path/'missing.json').exists()
    s=json.loads((tmp_path/'split.json').read_text())
    f=s['folds'][0];f['train'][0],f['test'][0]=f['test'][0],f['train'][0]
    (tmp_path/'bad.json').write_text(json.dumps(s))
    with pytest.raises(ValueError,match='泄漏'):load_cv_splits(df,tmp_path/'bad.json')


def test_fold_normalization_and_local_reference(tmp_path):
    df,folds=fixture_data(tmp_path);a=args_for(tmp_path,'--conditions','age','--fusion_method','film')
    train=folds[0]['train'];image=fit_image_stats(a,train);clinical=feature_stats(df,train,a.conditions)
    changed=df.copy();changed.loc[changed.label_index.isin(folds[0]['test']),'age']=10000
    assert clinical==feature_stats(changed,train,a.conditions)
    d=CornealTopoDataset(a,a.excel_path,a.data_root,dataframe=df,labels=folds[0]['test'],image_size=128,
        image_stats=image,clinical_mean=clinical['mean'],clinical_std=clinical['std'])
    item=d[0];assert item['label']==folds[0]['test'][0];assert item['sia_vector'].shape==(2,)
    raw=np.load(tmp_path/'raw'/'FRONT'/f"{item['label']}_pre.npy")
    assert torch.equal(item['mask_pre'][0].bool(),torch.from_numpy(np.isfinite(raw)))
    # A normalized minimum is valid, not missing.
    a.image_normalization='local';d.image_normalization='local';local=d[0]
    assert ((local['input']==0)&local['mask_pre'].bool()).any()
    batch={k:torch.as_tensor([local[k]]) for k in ('image_minimum','image_scale')}
    batch['mask_post']=local['mask_post'].unsqueeze(0)
    p,y=metric_reference_images(local['target'].unsqueeze(0),local['target'].unsqueeze(0),batch,{'image':image})
    assert torch.allclose(y.float(),item['target'].unsqueeze(0),atol=2e-6)


@pytest.mark.parametrize('model,sia',[('CPTP','off'),('CPTP','scalar'),('CPTP','vector'),('mCPTP','scalar'),('mCPTP','vector')])
def test_sia_optimizer_clinical_and_checkpoint(tmp_path,model,sia):
    a=args_for(tmp_path,'--model',model,'--sia_mode',sia,'--conditions','age','--fusion_method','film')
    stats={'sia':{'mean':[.2,-.1],'std':[.3,.5]}}
    m=create_model(a,stats); m.eval()
    pre=torch.rand(1,1,128,128);c=torch.ones(1,1)
    first=m(pre,c)
    with torch.no_grad():
        different=m(pre,c+10)
    assert not torch.equal(first['pred'],different['pred'])
    assert any(k.startswith('netG.cond_encoder') for k in m.state_dict())
    parameters=list(m.generator_parameters()); ids={id(x) for x in parameters}
    if sia=='vector':
        assert all(id(x) in ids for x in m.netSIA.parameters())
        assert torch.allclose(first['sia_vector'],m.sia_mean+m.sia_std*first['sia_state'])
        assert torch.allclose(first['sia'],first['sia_vector'].norm(dim=-1))
    if sia=='scalar':
        assert first['sia'].shape==(1,) and (first['sia']>=0).all()
        assert 'sia_vector' not in first and 'sia_state' not in first
        assert all(id(x) in ids for x in m.netSIA.parameters())
    optimizer={'G':torch.optim.AdamW(parameters,lr=.001),'D':torch.optim.AdamW(m.netD.parameters(),lr=.001)}
    before={k:v.clone() for k,v in m.state_dict().items()}
    batch=dict(input=pre,target=torch.rand_like(pre),condition=c,mask_pre=torch.ones_like(pre),
               mask_post=torch.ones_like(pre),sia_vector=torch.tensor([[.8,.6]]) if sia=='vector' else torch.empty(1,0),
               sia_magnitude=torch.tensor([1.]))
    train_one_epoch(m,[batch],optimizer,torch.device('cpu'))
    if sia!='off':
        for prefix in ('netSIA.proj_layers','netSIA.attn_fc','netSIA.reg_head'):
            assert any(not torch.equal(v,before[k]) for k,v in m.state_dict().items() if k.startswith(prefix))
    m.eval();expected=m(pre,c)
    torch.save(m.state_dict(),tmp_path/'roundtrip.pth')
    fresh=create_model(a,stats);fresh.load_state_dict(torch.load(tmp_path/'roundtrip.pth',weights_only=True));fresh.eval()
    for key,val in expected.items():assert torch.equal(val,fresh(pre,c)[key])


def test_d4_and_outcome_rejection(tmp_path):
    q=torch.tensor([2.,3.]); t=DataTransform()
    assert torch.equal(t.transform_vector(q,dict(hflip=True,vflip=False,angle=90)),torch.tensor([-2.,3.]))
    for col in ('post_K1','delta_front_X','front_SIA_D','label_index'):
        with pytest.raises(ValueError):args_for(tmp_path,'--conditions',col,'--fusion_method','film')
    with pytest.raises(ValueError):args_for(tmp_path,'--model','CPTP','--best_metric','SIA_MAE')
