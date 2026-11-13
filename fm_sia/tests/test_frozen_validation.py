import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'cddpm'))
from train import verify_frozen_validation


def reference(tmp_path):
    args=SimpleNamespace(epochs=200,best_metric='MAE',save_dir='new',validation_only=False,
                         validation_reference=str(tmp_path))
    stats={'image':{'minimum':20.,'scale':30.}}
    history=[{'epoch':1,'validation':{'MAE':.02,'SIA_MAE':.35}},
             {'epoch':200,'validation':{'MAE':.016,'SIA_MAE':.33}}]
    (tmp_path/'config.json').write_text(json.dumps(dict(vars(args),save_dir='old',validation_only=True,validation_reference=None)))
    folder=tmp_path/'fold_01';folder.mkdir()
    (folder/'normalization.json').write_text(json.dumps(stats))
    (folder/'history.json').write_text(json.dumps(history))
    return args,stats,history


def test_frozen_validation_accepts_only_matching_scientific_run(tmp_path):
    args,stats,history=reference(tmp_path)
    actual=copy.deepcopy(history);actual[0]['train_seconds']=999
    result=verify_frozen_validation(tmp_path,1,args,stats,actual)
    assert result['status']=='PASS' and result['selected_epoch']==200
    assert result['metrics_compared']==4 and result['max_absolute_metric_difference']==0


@pytest.mark.parametrize('change', ['metric','normalization','configuration','epochs'])
def test_frozen_validation_refuses_mismatch_before_test(tmp_path,change):
    args,stats,history=reference(tmp_path)
    if change=='metric':history[0]['validation']['SIA_MAE']+=.01
    elif change=='normalization':stats['image']['scale']+=1
    elif change=='configuration':args.epochs=100
    else:history.pop(0)
    with pytest.raises(ValueError,match='Frozen'):
        verify_frozen_validation(tmp_path,1,args,stats,history)
