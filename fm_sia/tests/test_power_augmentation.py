import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'cddpm'))
from power_augmentation import PairedSphericalShift,PRE_KMAX
from clinical_features import astigmatism_vector
from train import get_args,make_dataset,saved_config_value


def arguments(extra=()):
    return get_args(['--excel_path','unused','--data_root','unused','--sia_prediction','polar_direct',
        '--augmentation','spherical_shift','--conditions','pre_age_years',PRE_KMAX,
        '--fusion_method','film','--lambda_sia','1','--image_size','8',*extra])


def sample():
    pre=torch.arange(64,dtype=torch.float32).reshape(1,8,8)/100
    post=pre+.1;mask_pre=torch.ones_like(pre);mask_post=torch.ones_like(pre)
    mask_pre[0,0,0]=0;mask_post[0,0,1]=0
    return dict(input=pre*mask_pre,target=post*mask_post,mask_pre=mask_pre,mask_post=mask_post,
                condition=torch.tensor([.2,.4]),sia_vector=torch.tensor([.3,.4]),sia_magnitude=torch.tensor(.5),
                label=1,targets=torch.empty(0))


def test_shift_preserves_physical_velocity_masks_sia_and_source_sample():
    original=sample();before={k:v.clone() if torch.is_tensor(v) else v for k,v in original.items()}
    torch.manual_seed(7);out=PairedSphericalShift(.25,10.,1,2.)(original)
    delta=(out['condition'][1]-before['condition'][1])*2
    assert 0<abs(float(delta))<=.25
    for key,mask in [('input','mask_pre'),('target','mask_post')]:
        valid=before[mask].bool()
        torch.testing.assert_close((out[key]-before[key])[valid],torch.full_like(out[key][valid],delta/10))
        torch.testing.assert_close(out[key][~valid],before[key][~valid],rtol=0,atol=0)
    common=(out['mask_pre']*out['mask_post']).bool()
    torch.testing.assert_close((out['target']-out['input'])[common],(before['target']-before['input'])[common])
    for key in ('sia_vector','sia_magnitude','mask_pre','mask_post','targets'):
        torch.testing.assert_close(out[key],before[key],rtol=0,atol=0)
    assert out['condition'][0]==before['condition'][0]
    for key,value in before.items():
        if torch.is_tensor(value):torch.testing.assert_close(original[key],value,rtol=0,atol=0)
    # Independent meridional-power calculation: adding sphere preserves cylinder and axis.
    angles=np.array([0.,37.,179.]);flat=np.array([42.,43.,44.]);steep=flat+np.array([1.,2.,3.])
    np.testing.assert_allclose(astigmatism_vector(steep-flat,angles),
                               astigmatism_vector((steep+float(delta))-(flat+float(delta)),angles))


def test_zero_shift_is_identity_and_does_not_consume_randomness():
    item=sample();state=torch.get_rng_state().clone()
    assert PairedSphericalShift(0,10.,1,2.)(item) is item
    torch.testing.assert_close(torch.get_rng_state(),state,rtol=0,atol=0)


def test_random_shift_is_reproducible_and_changes_across_accesses():
    transform=PairedSphericalShift(.25,10.,1,2.);item=sample()
    torch.manual_seed(12);a=transform(item);b=transform(item)
    assert not torch.equal(a['condition'],b['condition'])
    torch.manual_seed(12);c=transform(item)
    for key in ('input','target','condition'):torch.testing.assert_close(a[key],c[key],rtol=0,atol=0)


def test_only_training_dataset_is_augmented_and_legacy_config_stays_unchanged(tmp_path):
    args=arguments();args.data_root=str(tmp_path)
    folder=tmp_path/'raw/FRONT';folder.mkdir(parents=True)
    pre=np.full((8,8),44.,dtype=np.float32);pre[0,0]=np.nan
    post=np.full((8,8),45.,dtype=np.float32);post[0,1]=np.nan
    np.save(folder/'1_pre.npy',pre);np.save(folder/'1_post.npy',post)
    df=pd.DataFrame({'label_index':[1],'pre_age_years':[52.],PRE_KMAX:[46.],
                     'delta_front_X':[.3],'delta_front_Y':[.4]})
    stats=dict(image=dict(minimum=40.,scale=10.),clinical=dict(mean=[50.,45.],std=[10.,2.]))
    train=make_dataset(args,df,[1],stats,training=True)
    validation=make_dataset(args,df,[1],stats)
    assert train.paired_transform is not None and validation.paired_transform is None
    state=torch.get_rng_state().clone();original=validation[0]
    torch.testing.assert_close(torch.get_rng_state(),state,rtol=0,atol=0)
    augmented=train[0];delta=(augmented['condition'][1]-original['condition'][1])*2
    assert abs(float(delta))<=.25
    torch.testing.assert_close(augmented['sia_vector'],original['sia_vector'],rtol=0,atol=0)
    assert augmented['sia_magnitude']==original['sia_magnitude']
    torch.testing.assert_close(validation[0]['input'],original['input'],rtol=0,atol=0)
    np.testing.assert_array_equal(np.load(folder/'1_pre.npy'),pre)
    args.augmentation='none';del args.spherical_shift_D
    old=make_dataset(args,df,[1],stats,training=True)
    assert old.paired_transform is None
    torch.testing.assert_close(old[0]['condition'],original['condition'],rtol=0,atol=0)
    assert saved_config_value({},'spherical_shift_D')==.25


@pytest.mark.parametrize('extra',[
    ['--spherical_shift_D','0'],['--spherical_shift_D','nan'],['--spherical_shift_D','-1'],
    ['--conditions','pre_age_years'],['--conditions',PRE_KMAX,'pre_GENERAL DATA_K1'],
    ['--sia_prediction','axial_direct'],['--input_maps','BACK','--output_maps','BACK'],
])
def test_cli_requires_matched_front_power_inputs_and_finite_strength(extra):
    with pytest.raises(ValueError):arguments(extra)
