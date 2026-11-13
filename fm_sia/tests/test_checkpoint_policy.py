import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'cddpm'))
import train


def test_default_and_resume_require_explicit_persistence():
    cli=['--excel_path','unused','--data_root','unused']
    assert train.get_args(cli).save_checkpoints is False
    with pytest.raises(ValueError,match='save_checkpoints'):
        train.get_args(cli+['--resume'])
    assert train.get_args(cli+['--save_checkpoints','--resume']).save_checkpoints
    assert train.saved_config_value({},'save_checkpoints') is True


@pytest.mark.parametrize('persist',[False,True])
def test_selected_weights_survive_later_training_without_default_disk_writes(tmp_path,monkeypatch,persist):
    args=train.get_args(['--excel_path','unused','--data_root','unused','--model','fm',
        '--save_dir',str(tmp_path),'--epochs','2','--eval_every','1','--validation_only']+
        (['--save_checkpoints'] if persist else []))
    model=torch.nn.Linear(1,1,bias=False)
    with torch.no_grad():model.weight.zero_()
    monkeypatch.setattr(train,'build_model',lambda *a:model)
    monkeypatch.setattr(train,'fit_image_stats',lambda *a:{'scale':1.})
    monkeypatch.setattr(train,'feature_stats',lambda *a:None)
    monkeypatch.setattr(train,'make_dataset',lambda a,d,labels,s,training=False:labels)
    monkeypatch.setattr(train,'make_loader',lambda ds,*a,**kw:ds)
    monkeypatch.setattr(train,'visualize_epoch',lambda *a,**kw:None)
    def training_step(model,loader,optimizer,*a,**kw):
        optimizer.step()
        with torch.no_grad():model.weight.add_(1)
        return {'loss':1.}
    monkeypatch.setattr(train,'train_one_epoch',training_step)
    monkeypatch.setattr(train,'evaluate',lambda model,*a,**kw:{'MAE':model.weight.item()})
    if not persist:
        monkeypatch.setattr(train.torch,'save',lambda *a,**kw:pytest.fail('Default run wrote a checkpoint'))
    result=train.run_fold(args,None,dict(fold=1,train=[1],val=[2],test=[3]),torch.device('cpu'))
    assert result['best']['MAE']==1 and result['final']['MAE']==2
    assert bool(list(tmp_path.rglob('*.pth')))==persist
    assert (tmp_path/'fold_01/history.json').exists()
