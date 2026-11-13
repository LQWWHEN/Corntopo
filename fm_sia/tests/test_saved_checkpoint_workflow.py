"""Exercise saved validation selection, test isolation, and scoped cleanup."""
import hashlib
import json
from pathlib import Path
import sys
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'cddpm'))
import train
import predict
import test_saved_run as saved
from checkpoint_lifecycle import finalize_checkpoints


def prepare(tmp_path, monkeypatch):
    run=tmp_path/'train'; table=tmp_path/'data';table.write_text('synthetic')
    split=tmp_path/'split.json';fold=dict(fold=1,train=[1],val=[2],test=[3])
    train.json_save(split,dict(folds=[fold]))
    args=train.get_args(['--excel_path',str(table),'--data_root',str(tmp_path),'--model','fm',
        '--save_dir',str(run),'--split_file',str(split),'--epochs','2','--eval_every','1',
        '--k_folds','1','--validation_only','--save_checkpoints'])
    args.excel_sha256=hashlib.sha256(table.read_bytes()).hexdigest()
    model=torch.nn.Linear(1,1,bias=False)
    with torch.no_grad():model.weight.zero_()
    monkeypatch.setattr(train,'build_model',lambda *a:model)
    monkeypatch.setattr(predict,'build_model',lambda *a:torch.nn.Linear(1,1,bias=False))
    monkeypatch.setattr(train,'fit_image_stats',lambda *a:{'scale':1.})
    monkeypatch.setattr(train,'feature_stats',lambda *a:None)
    labels_seen=[]
    def dataset(a,d,labels,s,training=False):
        labels_seen.append(labels);return labels
    for module in (train,saved):
        monkeypatch.setattr(module,'make_dataset',dataset)
        monkeypatch.setattr(module,'make_loader',lambda ds,*a,**kw:ds)
    monkeypatch.setattr(train,'visualize_epoch',lambda *a,**kw:None)
    monkeypatch.setattr(saved,'load_dataframe',lambda *a:None)
    monkeypatch.setattr(saved,'validate_table',lambda *a:None)
    monkeypatch.setattr(saved,'validate_maps',lambda *a:None)
    def step(model,loader,optimizer,*a,**kw):
        optimizer.step()
        with torch.no_grad():model.weight.add_(1)
        return {'loss':1.}
    monkeypatch.setattr(train,'train_one_epoch',step)
    def evaluate(model,*a,**kw):
        w=model.weight.item();return dict(MAE=w,RMSE=w,PSNR=40-w,SSIM=1-w/10)
    monkeypatch.setattr(train,'evaluate',evaluate)
    monkeypatch.setattr(saved,'evaluate',evaluate)
    result=train.run_fold(args,None,fold,torch.device('cpu'))
    train.json_save(run/'config.json',vars(args))
    train.write_summary([result],run/'cv_summary.json',args,split)
    return run, labels_seen


def test_selected_checkpoint_test_and_cleanup(tmp_path,monkeypatch):
    run,labels=prepare(tmp_path,monkeypatch);out=tmp_path/'test'
    saved.test_run(run,out,'cpu')
    s=saved.read(out/'cv_summary.json')
    assert s['fold_results'][0]['best']['MAE']==1
    assert s['fold_results'][0]['final']['MAE']==2
    assert saved.read(out/'phase_timing.json')['training_seconds']==0
    assert labels[-4:]==[[2],[2],[3],[3]]
    assert list(run.rglob('*.pth'))  # Retained until result audit.
    audit=tmp_path/'audit.json';train.json_save(audit,dict(audit='PASS'))
    unrelated=tmp_path/'pretrained.pth';unrelated.write_bytes(b'preserve')
    result=finalize_checkpoints(run,audit,'tested',out)
    assert result['status']=='PASS' and result['remaining_checkpoints']==0
    assert not list(run.rglob('*.pth')) and unrelated.exists()
    assert (run/'fold_01/history.json').exists() and (out/'cv_summary.json').exists()


def test_validation_mismatch_never_constructs_test_loader(tmp_path,monkeypatch):
    run,labels=prepare(tmp_path,monkeypatch)
    path=run/'fold_01/validation_results/best.json';m=saved.read(path);m['MAE']=99
    train.json_save(path,m)
    with pytest.raises(ValueError,match='validation differs'):
        saved.test_run(run,tmp_path/'test','cpu')
    assert [3] not in labels and list(run.rglob('*.pth'))


def test_cleanup_preserves_failed_audit_and_accepts_rejection(tmp_path,monkeypatch):
    run,_=prepare(tmp_path,monkeypatch);audit=tmp_path/'audit.json'
    train.json_save(audit,dict(audit='FAIL',selected_for_frozen_test=False))
    with pytest.raises(ValueError,match='audit must pass'):
        finalize_checkpoints(run,audit,'rejected')
    assert list(run.rglob('*.pth'))
    train.json_save(audit,dict(audit='PASS',selected_for_frozen_test=False))
    assert finalize_checkpoints(run,audit,'rejected')['remaining_checkpoints']==0
