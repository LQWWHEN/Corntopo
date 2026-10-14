import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import torch
import train
from test_contracts import args_for, fixture_data


@pytest.mark.parametrize('mode',['scalar','vector'])
def test_three_fold_cli_and_independent_evaluation(tmp_path,mode):
    fixture_data(tmp_path)
    root=Path(__file__).resolve().parents[1]
    common=['--excel_path',str(tmp_path/'data.csv'),'--data_root',str(tmp_path),
        '--split_file',str(tmp_path/'split.json'),'--save_dir',str(tmp_path/'results'),
        '--device','cpu','--num_workers','0','--epochs','1','--batch_size','8',
        '--ngf','4','--ndf','4','--sia_mode',mode,'--conditions','age','--fusion_method','film','--use_ema','--save_predictions']
    split_before=(tmp_path/'split.json').read_bytes()
    r=subprocess.run([sys.executable,str(root/'train.py'),*common],capture_output=True,text=True)
    assert r.returncode==0,r.stdout+r.stderr
    summary=json.loads((tmp_path/'results/cv_summary.json').read_text())
    assert summary['requested_folds']==summary['completed_folds']==3
    assert [r['fold'] for r in summary['fold_results']]==[1,2,3]
    assert summary['aggregate']['best']['SIA_MAE']['n']==3
    assert ('Vector_MAE' in summary['aggregate']['best']) == (mode=='vector')
    assert summary['sia_mode']==mode
    assert (tmp_path/'results/fold_01/best_sia.pth').exists()
    sample=np.load(tmp_path/'results/fold_01/test_results/best/000000_sia_samples.npz')
    assert set(sample.files)==({'s','q'} if mode=='vector' else {'s'})
    assert summary['aggregate']['best']['PSNR_full']['n']==3
    assert summary['aggregate']['best']['PSNR_valid']['n']==3
    assert (tmp_path/'split.json').read_bytes()==split_before
    r=subprocess.run([sys.executable,str(root/'test.py'),'--save_dir',str(tmp_path/'results'),
                      '--device','cpu','--batch_size','1'],capture_output=True,text=True)
    assert r.returncode==0,r.stdout+r.stderr
    retest=json.loads((tmp_path/'results/reevaluation/cv_summary.json').read_text())
    for first,second in zip(summary['fold_results'],retest['fold_results']):
        for selection in ('best','final'):
            for key,value in first[selection].items():
                assert second[selection][key]==pytest.approx(value,abs=1e-6),key
    # Both optimizer states, regressor weights and statistics must be checkpointed.
    state=torch.load(tmp_path/'results/fold_01/best.pth',weights_only=False,map_location='cpu')
    assert set(state['optimizer'])=={'G','D'}
    assert any(k.startswith('netSIA.proj_layers') for k in state['model'])
    assert set(state['stats'])=={'image','clinical','sia'}
    # Refuse silent overwrite of completed experiments.
    r=subprocess.run([sys.executable,str(root/'train.py'),*common],capture_output=True,text=True)
    assert r.returncode!=0 and 'already contains checkpoints' in r.stderr


@pytest.mark.parametrize('mode',['scalar','vector'])
def test_resume_matches_uninterrupted(tmp_path,monkeypatch,mode):
    df,folds=fixture_data(tmp_path);a=args_for(tmp_path,'--conditions','age','--fusion_method','film',
        '--use_ema','--dropout','--augmentation','d4','--sia_mode',mode)
    # Fill the fingerprints main() normally records.
    a.excel_sha256=hashlib.sha256(Path(a.excel_path).read_bytes()).hexdigest()
    a.split_sha256=hashlib.sha256(Path(a.split_file).read_bytes()).hexdigest()
    a.save_dir=str(tmp_path/'continuous')
    train.run_fold(a,df,folds[0],torch.device('cpu'))
    b=copy.deepcopy(a);b.save_dir=str(tmp_path/'interrupted')
    real_save=train.torch_save_atomic
    def stop_after_epoch_one(payload,path):
        real_save(payload,path)
        if Path(path).name=='last.pth' and payload['epoch']==0:
            raise RuntimeError('simulated interruption')
    monkeypatch.setattr(train,'torch_save_atomic',stop_after_epoch_one)
    with pytest.raises(RuntimeError,match='simulated interruption'):
        train.run_fold(b,df,folds[0],torch.device('cpu'))
    monkeypatch.setattr(train,'torch_save_atomic',real_save)
    b.resume=True
    train.run_fold(b,df,folds[0],torch.device('cpu'))
    one=torch.load(Path(a.save_dir)/'fold_01/last.pth',weights_only=False,map_location='cpu')
    two=torch.load(Path(b.save_dir)/'fold_01/last.pth',weights_only=False,map_location='cpu')
    for domain in ('model','ema'):
        for key,value in one[domain].items():assert torch.equal(value,two[domain][key]),key
    assert one['best']==two['best']
