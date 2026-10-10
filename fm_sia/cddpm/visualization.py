"""Separate, fixed-scale pre/gt/pred/error PNGs for corneal topography.

The jet palette, black invalid pixels, lower origin and borderless 512-pixel
rendering follow the available dataset heatmap exporter. Numeric predictions
and metric inputs are never modified by display clipping.
"""
from pathlib import Path
import numpy as np
from PIL import Image
from matplotlib import colormaps


def visualization_spec(image_stats, error_max=1.0):
    if not np.isfinite(error_max) or error_max <= 0:
        raise ValueError('visual_error_max must be positive and finite')
    minimum, scale = float(image_stats['minimum']), float(image_stats['scale'])
    if not np.isfinite([minimum, scale]).all() or scale <= 0:
        raise ValueError('Invalid visualization image scale')
    return dict(cmap='jet', invalid_color='black', origin='lower', size_pixels=512,
                map_min=minimum, map_max=minimum+scale,
                error_min=0.0, error_max=float(error_max),
                units='original map units (D when FRONT stores corneal power)',
                error_definition='absolute difference between prediction and GT',
                masks=dict(pre='pre valid',gt='post valid',pred='pre valid',error='common pre/post valid'),
                scale_policy='fixed training-fold map range; fixed error range; display-only clipping')


def save_heatmap(values, path, valid, vmin, vmax, size=512):
    """Save a borderless RGB PNG without per-image min/max normalization."""
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if values.ndim != 2 or valid.shape != values.shape:
        raise ValueError('Heatmap values and validity must be matching 2D arrays')
    if not np.isfinite([vmin,vmax]).all() or vmax <= vmin:
        raise ValueError('Invalid heatmap display bounds')
    valid = valid & np.isfinite(values)
    normalized = np.clip((np.where(valid,values,vmin)-vmin)/(vmax-vmin),0,1)
    rgb = colormaps['jet'](normalized,bytes=True)[...,:3].copy()
    rgb[~valid] = 0
    # The dataset exporter uses origin='lower'. Flip only for PNG rendering.
    picture = Image.fromarray(np.flipud(rgb))
    picture = picture.resize((size,size),resample=Image.Resampling.NEAREST)
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    picture.save(path)


def save_visuals(output_dir, stem, pre, gt, pred, mask_pre, mask_post, image_stats, error_max=1.0):
    """Save all four files using the same range across patients and epochs."""
    spec = visualization_spec(image_stats,error_max)
    pre,gt,pred = [np.asarray(x,dtype=np.float64) for x in (pre,gt,pred)]
    mask_pre,mask_post = [np.asarray(x,dtype=bool) for x in (mask_pre,mask_post)]
    if not (pre.shape==gt.shape==pred.shape==mask_pre.shape==mask_post.shape):
        raise ValueError('Visualization image/mask shapes differ')
    scale,minimum = image_stats['scale'],image_stats['minimum']
    arrays = dict(pre=pre*scale+minimum,gt=gt*scale+minimum,
                  pred=pred*scale+minimum,error=np.abs(pred-gt)*scale)
    masks = dict(pre=mask_pre,gt=mask_post,pred=mask_pre,error=mask_pre & mask_post)
    for name,values in arrays.items():
        limits = (0.0,error_max) if name=='error' else (spec['map_min'],spec['map_max'])
        save_heatmap(values,Path(output_dir)/f'{stem}_{name}.png',masks[name],*limits)
