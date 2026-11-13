import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'cddpm'))
from clinical_features import astigmatism_vector,prepare_clinical_features,PRE_ASTIG,PRE_AXIS,PRE_VECTOR
from dataloader import feature_stats
from train import get_args,load_dataframe


def test_preoperative_direction_representation_and_period():
    q=astigmatism_vector(np.ones(4),[0,45,90,135])
    np.testing.assert_allclose(q,[[1,0],[0,1],[-1,0],[0,-1]],atol=1e-12)
    np.testing.assert_allclose(q,astigmatism_vector(np.ones(4),[180,225,270,315]),atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(q,axis=1),1)
    np.testing.assert_allclose(astigmatism_vector(0,[0,90]),0)


def test_features_ignore_postoperative_and_cached_values(tmp_path):
    frame=pd.DataFrame({'label_index':[1,2,3],PRE_ASTIG:[1.,2.,3.],PRE_AXIS:[0.,45.,90.],
        'pre_front_X':[999.]*3,'pre_front_Y':[999.]*3,'post_value':[99.]*3,'delta_front_X':[12.]*3})
    original=frame.copy(deep=True)
    first=prepare_clinical_features(frame,PRE_VECTOR)
    pd.testing.assert_frame_equal(frame,original)
    expected=first[list(PRE_VECTOR)].copy()
    frame['post_value']=-100;frame['delta_front_X']=100;frame['pre_front_X']=-999
    second=prepare_clinical_features(frame,PRE_VECTOR)
    pd.testing.assert_frame_equal(expected,second[list(PRE_VECTOR)])
    stats=feature_stats(first,[1,2],PRE_VECTOR)
    first.loc[2,list(PRE_VECTOR)]=999
    assert stats==feature_stats(first,[1,2],PRE_VECTOR)
    path=tmp_path/'pre.csv';frame.to_csv(path,index=False);before=path.read_bytes()
    actual=load_dataframe(path,PRE_VECTOR)
    pd.testing.assert_frame_equal(actual[list(PRE_VECTOR)],expected)
    assert path.read_bytes()==before
    assert prepare_clinical_features(frame,[]) is frame


def test_unimplemented_directional_augmentation_rejected():
    cli=['--excel_path','x','--data_root','x','--conditions',*PRE_VECTOR]
    assert get_args(cli).conditions==list(PRE_VECTOR)
    with pytest.raises(ValueError,match='D4 does not transform'):
        get_args(cli+['--augmentation','d4'])
    with pytest.raises(ValueError,match='Outcome'):
        get_args(['--excel_path','x','--data_root','x','--conditions','post_front_X'])
