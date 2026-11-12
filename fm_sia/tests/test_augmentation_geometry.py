"""D4 geometry acceptance tests on synthetic fields, with no patient data."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'cddpm'))
from transforms import DataTransform


@pytest.mark.parametrize('hflip', [False, True])
@pytest.mark.parametrize('vflip', [False, True])
@pytest.mark.parametrize('angle', [0, 90, 180, 270])
def test_spin_two_vector_matches_transformed_quadratic_field(hflip, vflip, angle):
    # The spin-two basis x²-y², 2xy gives an independent geometric oracle.
    axis = torch.linspace(-1, 1, 17, dtype=torch.float64)
    y, x = torch.meshgrid(axis, axis, indexing='ij')
    q = torch.tensor([0.6, 0.8], dtype=torch.float64)
    field = q[0]*(x*x-y*y) + q[1]*(2*x*y)
    params = dict(hflip=hflip, vflip=vflip, angle=angle)
    transform = DataTransform()
    q_new = transform.transform_vector(q, params)
    predicted_field = q_new[0]*(x*x-y*y) + q_new[1]*(2*x*y)
    torch.testing.assert_close(transform(field, params), predicted_field)
    torch.testing.assert_close(q_new.norm(), q.norm())
    torch.testing.assert_close(q, torch.tensor([0.6, 0.8], dtype=torch.float64))


def test_paired_maps_masks_and_scalar_label_share_geometry(tmp_path):
    import numpy as np
    import pandas as pd
    from types import SimpleNamespace
    from dataloader import CornealTopoDataset, fit_image_stats

    class FixedReflection(DataTransform):
        def get_params(self, seed=None):
            return dict(hflip=True, vflip=False, angle=0)

    folder = tmp_path/'raw'/'FRONT'
    folder.mkdir(parents=True)
    pre = np.arange(64, dtype=np.float32).reshape(8, 8)
    post = pre+2
    pre[1, 2] = np.nan
    post[2, 3] = np.nan
    np.save(folder/'1_pre.npy', pre)
    np.save(folder/'1_post.npy', post)
    args = SimpleNamespace(data_root=tmp_path, input_maps=['FRONT'], output_maps=['FRONT'],
        image_size=8, invalid_zero=False, conditions=['pre_age_years'], targets=[],
        model='fm_sia', sia_mode='vector', sia_columns=['delta_front_X', 'delta_front_Y'],
        sia_magnitude_column='front_SIA_D')
    frame = pd.DataFrame(dict(label_index=[1], pre_age_years=[50.],
        delta_front_X=[.6], delta_front_Y=[.8], front_SIA_D=[1.]))
    options = dict(dataframe=frame, image_size=8, image_stats=fit_image_stats(args, [1]))
    plain = CornealTopoDataset(args, None, tmp_path, **options)[0]
    aug = CornealTopoDataset(args, None, tmp_path, transforms=FixedReflection(), **options)[0]
    for key in ['input', 'target', 'mask_pre', 'mask_post']:
        torch.testing.assert_close(aug[key], plain[key].flip(-1))
    torch.testing.assert_close(aug['sia_vector'], torch.tensor([.6, -.8]))
    torch.testing.assert_close(aug['sia_magnitude'], plain['sia_magnitude'])
    torch.testing.assert_close(aug['condition'], plain['condition'])
