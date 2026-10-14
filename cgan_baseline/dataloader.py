"""Raw maps with global or pre-only case normalization and physical SIA labels."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

DATASET_API_VERSION = 'fm_sia_v1'


def center_crop(arr, crop_size=128):
    if arr.ndim != 2:
        raise ValueError(f'Expected a 2D numeric map, got {arr.shape}')
    if crop_size is None:
        return arr
    h, w = arr.shape
    if min(h, w) < crop_size:
        raise ValueError(f'{arr.shape} smaller than crop_size={crop_size}')
    top, left = (h-crop_size)//2, (w-crop_size)//2
    return arr[top:top+crop_size, left:left+crop_size]


def read_raw_map(data_root, map_name, label, phase, image_size, invalid_zero=False):
    path = Path(data_root)/'raw'/map_name/f'{label}_{phase}.npy'
    if not path.is_file():
        raise FileNotFoundError(path)
    image = center_crop(np.load(path, allow_pickle=False).astype(np.float32), image_size)
    mask = np.isfinite(image)
    if invalid_zero:
        mask &= image != 0
    if not mask.any():
        raise ValueError(f'No valid pixels: {path}')
    return image, mask


def numeric_features(df, columns):
    missing = set(columns)-set(df.columns)
    if missing:
        raise ValueError(f'Excel missing columns: {sorted(missing)}')
    values = df.loc[:, columns].apply(pd.to_numeric, errors='raise').to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f'Non-finite values in columns: {columns}')
    return values


def feature_stats(df, train_labels, columns):
    if not columns:
        return None
    train_df = df[df.label_index.isin(train_labels)]
    if len(train_df) != len(train_labels):
        raise ValueError('Training labels do not uniquely match dataframe')
    values = numeric_features(train_df, columns)
    mean, std = values.mean(0), values.std(0)
    # A constant component remains well-defined in physical units.
    std[std < 1e-8] = 1.0
    return dict(columns=list(columns), mean=mean.tolist(), std=std.tolist(), count=len(values))


def local_image_stats(image, mask):
    """A single affine transform derived only from valid preoperative pixels.

    Reuse this transform for pre, post and prediction. A constant preoperative
    map uses a 1-original-unit scale, retaining invertibility without division
    by zero. Ground truth must never be used to obtain these parameters.
    """
    image, mask = np.asarray(image), np.asarray(mask,dtype=bool)
    if image.shape != mask.shape:
        raise ValueError('Image and mask shapes differ')
    values=image[mask & np.isfinite(image)]
    if not values.size:
        raise ValueError('No valid preoperative pixels for local normalization')
    minimum,maximum=float(values.min()),float(values.max())
    constant=maximum-minimum < 1e-8
    scale=1.0 if constant else maximum-minimum
    return dict(minimum=minimum,maximum=minimum+scale,scale=scale,
                observed_maximum=maximum,constant_fallback=constant,
                fit_on='case_pre_valid_pixels',mode='local_pre_minmax',clip=False)


def select_image_stats(mode, pre, mask_pre, global_stats):
    if mode == 'local':
        return local_image_stats(pre,mask_pre)
    if mode != 'global':
        raise ValueError(f'Unknown image_normalization: {mode}')
    return global_stats


def fit_image_stats(args, train_labels):
    minimum, maximum = np.inf, -np.inf
    for label in train_labels:
        for phase, names in [('pre', args.input_maps), ('post', args.output_maps)]:
            for name in names:
                image, mask = read_raw_map(args.data_root, name, label, phase,
                                           args.image_size, args.invalid_zero)
                minimum = min(minimum, float(image[mask].min()))
                maximum = max(maximum, float(image[mask].max()))
    if not np.isfinite([minimum, maximum]).all() or maximum-minimum < 1e-8:
        raise ValueError('Training images have no usable shared intensity range')
    return dict(mode='train_minmax', minimum=minimum, maximum=maximum,
                scale=maximum-minimum, clip=False, fit_on='train_pre_and_post_valid_pixels',
                normalization_mode=getattr(args,'image_normalization','global'),
                local_rule='case pre min/max; constant pre uses scale=1 original unit',
                input_maps=args.input_maps, output_maps=args.output_maps)


def validate_table(df, args):
    if 'label_index' not in df or df.label_index.isna().any() or df.label_index.duplicated().any():
        raise ValueError('label_index must be present, nonempty and unique')
    numeric_features(df, args.conditions)
    result = dict(rows=len(df), conditions=list(args.conditions))
    if getattr(args, 'sia_mode', 'off') != 'off':
        has_vector = all(c in df for c in args.sia_columns)
        if args.sia_mode == 'vector' and not has_vector:
            raise ValueError('Vector mode requires both physical X/Y columns')
        q = numeric_features(df, args.sia_columns) if has_vector else None
        if q is not None:
            s = np.linalg.norm(q, axis=1)
            result['sia_label_source'] = 'norm of physical X/Y (same as FM_SIA)'
        else:
            s = numeric_features(df, [args.sia_magnitude_column])[:, 0]
            result['sia_label_source'] = args.sia_magnitude_column
        if (s < 0).any():
            raise ValueError('SIA magnitude labels must be nonnegative')
        skip_sia_checks = getattr(args, 'skip_sia_checks', False)
        if q is None:
            result['sia_magnitude_check'] = 'scalar labels only; vector consistency unavailable'
        elif skip_sia_checks:
            result['sia_magnitude_check'] = 'skipped (--skip_sia_checks); SIA computed from X/Y'
        elif args.sia_magnitude_column in df:
            given = numeric_features(df, [args.sia_magnitude_column])[:, 0]
            mismatch = ~np.isclose(given, s, atol=args.sia_label_atol, rtol=1e-5)
            if mismatch.any():
                labels = df.loc[mismatch, 'label_index'].tolist()[:10]
                raise ValueError(f'SIA norm mismatch at labels {labels}; check X/Y versus J0/J45 scaling')
            result['sia_magnitude_check'] = 'passed'
        else:
            result['sia_magnitude_check'] = 'column absent; computed from physical X/Y'
        if 'sia_valid' in df:
            valid = df.sia_valid.map(lambda v: str(v).strip().lower() in {'true', '1', '1.0'})
            result['sia_invalid_or_unknown_count'] = int((~valid).sum())
            if not valid.all() and not skip_sia_checks:
                labels = df.loc[~valid, 'label_index'].tolist()[:10]
                raise ValueError(f'Invalid/unknown sia_valid at labels {labels}; resolve quality flags before training')
            result['sia_quality_check'] = 'skipped (--skip_sia_checks)' if skip_sia_checks else 'passed'
        else:
            result['sia_quality_check'] = 'sia_valid absent; numerical validation only'
        result['sia_min_D'], result['sia_max_D'] = float(s.min()), float(s.max())
    if 'pre_exam_date_used' in df and 'post_exam_date_used' in df:
        gap = (pd.to_datetime(df.post_exam_date_used, errors='coerce')-
               pd.to_datetime(df.pre_exam_date_used, errors='coerce')).dt.days
        result['exam_gap_days'] = dict(min=float(gap.min()) if gap.notna().any() else None,
                                       max=float(gap.max()) if gap.notna().any() else None,
                                       note='exam gap is not postoperative follow-up duration')
    return result


def validate_maps(df, args):
    for label in df.label_index:
        masks = []
        for phase, names in [('pre', args.input_maps), ('post', args.output_maps)]:
            for name in names:
                _, mask = read_raw_map(args.data_root, name, label, phase, args.image_size, args.invalid_zero)
                masks.append(mask)
        if not (masks[0] & masks[1]).any():
            raise ValueError(f'No common pre/post valid region at label {label}')


class CornealTopoDataset(Dataset):
    def __init__(self, opt, excel_path, data_root, input_maps=None, output_maps=None,
                 data_type='raw', image_size=None, normalize=True, transforms=None,
                 seed=42, indices=None, labels=None, dataframe=None,
                 clinical_mean=None, clinical_std=None, image_stats=None):
        if data_type != 'raw':
            raise ValueError('This physically scaled pipeline requires raw .npy maps')
        if indices is not None and labels is not None:
            raise ValueError('Use indices or labels, not both')
        self.opt, self.data_root = opt, data_root
        self.df = dataframe if dataframe is not None else pd.read_excel(excel_path)
        if self.df.label_index.duplicated().any():
            raise ValueError('Duplicate label_index')
        self.rows = self.df.set_index('label_index', drop=False)
        all_labels = sorted(self.df.label_index.tolist())
        self.labels = list(labels) if labels is not None else (
            [all_labels[i] for i in indices] if indices is not None else all_labels)
        if len(set(self.labels)) != len(self.labels) or not set(self.labels) <= set(all_labels):
            raise ValueError('Invalid sample labels')
        self.input_maps = input_maps or getattr(opt, 'input_maps', ['FRONT'])
        self.output_maps = output_maps or getattr(opt, 'output_maps', ['FRONT'])
        self.image_size, self.transforms = image_size, transforms
        self.normalize, self.image_stats = normalize, image_stats
        self.image_normalization = getattr(opt,'image_normalization','global')
        if self.image_normalization not in ('global','local'):
            raise ValueError('image_normalization must be global or local')
        if normalize and image_stats is None:
            raise ValueError('Supply training-fold image_stats for the shared evaluation/display reference')
        self.invalid_zero = getattr(opt, 'invalid_zero', False)
        self.conditions = getattr(opt, 'conditions', [])
        self.targets = getattr(opt, 'targets', [])
        self.sia_columns = getattr(opt, 'sia_columns', ['delta_front_X', 'delta_front_Y'])
        self.use_sia = getattr(opt, 'sia_mode', 'off') != 'off'
        self.has_sia_vector = self.use_sia and all(c in self.df for c in self.sia_columns)
        self.sia_magnitude_column = getattr(opt, 'sia_magnitude_column', 'front_SIA_D')
        if (clinical_mean is None) != (clinical_std is None):
            raise ValueError('Clinical mean and std must be provided together')
        self.clinical_mean = None if clinical_mean is None else np.asarray(clinical_mean, np.float32)
        self.clinical_std = None if clinical_std is None else np.asarray(clinical_std, np.float32)
        if self.clinical_mean is not None:
            if self.clinical_mean.shape != (len(self.conditions),) or self.clinical_std.shape != self.clinical_mean.shape or not np.isfinite(self.clinical_mean).all() or not np.isfinite(self.clinical_std).all() or (self.clinical_std <= 0).any():
                raise ValueError('Invalid clinical normalization statistics')

    def load_maps(self, names, label, phase, *, normalize=None, image_stats=None):
        if normalize is None: normalize=self.normalize
        image_stats=self.image_stats if image_stats is None else image_stats
        images, masks = [], []
        for name in names:
            image, mask = read_raw_map(self.data_root, name, label, phase, self.image_size, self.invalid_zero)
            if normalize:
                image = (image-image_stats['minimum'])/image_stats['scale']
            images.append(np.where(mask, image, 0).astype(np.float32))
            masks.append(mask.astype(np.float32))
        return torch.from_numpy(np.stack(images)), torch.from_numpy(np.stack(masks))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        row = self.rows.loc[label]
        # Read raw pre first; derive the affine transform without post information.
        raw_pre, raw_mask = self.load_maps(self.input_maps,label,'pre',normalize=False)
        case_stats=local_image_stats(raw_pre.numpy(),raw_mask.numpy())
        norm = case_stats if self.image_normalization=='local' else self.image_stats
        if not self.normalize: norm=dict(minimum=0.0,scale=1.0)
        # Keep the existing NumPy float32 arithmetic for backward-compatible global mode.
        pre_array=raw_pre.numpy()
        pre=torch.from_numpy(np.where(raw_mask.numpy().astype(bool),
            (pre_array-norm['minimum'])/norm['scale'],0).astype(np.float32))
        mask_pre=raw_mask
        post, mask_post = self.load_maps(self.output_maps, label, 'post',
                                         normalize=self.normalize,image_stats=norm)
        clinical = np.asarray([float(row[c]) for c in self.conditions], np.float32)
        if self.clinical_mean is not None:
            clinical = (clinical-self.clinical_mean)/self.clinical_std
        q = torch.tensor([float(row[c]) for c in self.sia_columns]) if self.has_sia_vector else torch.empty(0)
        magnitude = q.norm() if self.has_sia_vector else (torch.tensor(float(row[self.sia_magnitude_column])) if self.use_sia else torch.tensor(0.0))
        if self.transforms is not None:
            params = self.transforms.get_params()  # worker RNG, not a repeated fixed seed
            pre, post, mask_pre, mask_post = [self.transforms(x, params) for x in (pre, post, mask_pre, mask_post)]
            if self.has_sia_vector:
                q = self.transforms.transform_vector(q, params)
        return dict(input=pre, target=post, mask_pre=mask_pre, mask_post=mask_post,
                    image_minimum=float(norm['minimum']), image_scale=float(norm['scale']),
                    case_minimum=float(case_stats['minimum']),case_scale=float(case_stats['scale']),
                    condition=torch.from_numpy(clinical), sia_vector=q, sia_magnitude=magnitude,
                    targets=torch.tensor([float(row[c]) for c in self.targets]), label=label)


def build_dataloader(excel_path, data_root, input_maps, output_maps, data_type='raw',
                     batch_size=4, shuffle=True, num_workers=4, image_size=None, opt=None, **kwargs):
    ds = CornealTopoDataset(opt, excel_path, data_root, input_maps, output_maps,
                           data_type, image_size, **kwargs)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
