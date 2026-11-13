"""Patient-level CV for image models and coupled FRONT / SIA flow matching."""
import argparse
import hashlib
import json
import math
import os
import random
import shutil
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, default_collate
from tqdm import tqdm
from dataloader import (CornealTopoDataset, feature_stats, fit_image_stats,
                        validate_table, validate_maps, sia_target_columns)
from transforms import DataTransform
from patient_cv import load_or_create_cv_splits
from metrics import MetricAccumulator, axis_details
from visualization import save_visuals, visualization_spec
from model.encoder import ConditionEncoder
from model.unet import ConditionalUNet
from model.ema import EMA
from model.fm_sia import JointFlowMatching, fixed_sia_noise
from clinical_features import prepare_clinical_features, PRE_VECTOR
from task_balancing import (BALANCE_DEFAULTS, build_task_balancer,
                            gradnorm_shared_parameters)

PIPELINE_VERSION = 'front_fm_sia_v1'


def load_dataframe(path, conditions=()):
    frame = pd.read_csv(path) if Path(path).suffix.lower() == '.csv' else pd.read_excel(path)
    return prepare_clinical_features(frame, conditions)


def get_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--excel_path', required=True)
    p.add_argument('--data_root', required=True)
    p.add_argument('--input_maps', nargs='+', default=['FRONT'])
    p.add_argument('--output_maps', nargs='+', default=['FRONT'])
    p.add_argument('--image_size', type=int, default=128)
    p.add_argument('--model', choices=['cddpm','bbdm','fm','fm_sia','fno','uno'], default='fm_sia')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--cpu_threads', type=int, default=2)
    p.add_argument('--time_steps', type=int, default=None)
    p.add_argument('--save_dir', default='./checkpoints')
    p.add_argument('--device', default='cuda')
    p.add_argument('--sampler', choices=['ddpm','ddim'], default='ddpm', help='CDDPM only')
    p.add_argument('--ode_solver', choices=['euler','heun','rk4'], default='euler')
    p.add_argument('--sample_steps', type=int, default=100)
    p.add_argument('--ddim_eta', type=float, default=0.0)
    p.add_argument('--k_folds', type=int, default=10)
    p.add_argument('--split_file', default=None)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--eval_seed', type=int, default=2026)
    p.add_argument('--eval_samples', type=int, default=1, help='Fixed noise samples per input for FM-SIA')
    p.add_argument('--use_ema', action='store_true')
    p.add_argument('--ema_decay', type=float, default=0.995)
    p.add_argument('--objective', default='grad')
    p.add_argument('--conditions', nargs='+', default=[])
    p.add_argument('--targets', nargs='+', default=[], help='Legacy extra fields; SIA uses the selected mode-specific label columns')
    p.add_argument('--fusion_method', choices=['none','concat','add','self_attention','cross_attention','film'], default='none')
    p.add_argument('--clinical_embed_dim', type=int, default=128)
    p.add_argument('--sia_columns', nargs=2, default=['delta_front_X','delta_front_Y'])
    p.add_argument('--sia_mode', choices=['vector','scalar'], default='vector',
                   help='FM-SIA target: vector=DeltaX/DeltaY; scalar=direct magnitude in D')
    p.add_argument('--sia_prediction', choices=['flow','direct','axial_direct'], default='flow',
                   help='SIA flow, direct scalar, or spatial-moment direct vector head')
    p.add_argument('--sia_feature_fusion', choices=['lab','preop_frame'], default='lab',
                   help='Axial direct readout: lab moments or preoperative-axis-conditioned moments')
    p.add_argument('--image_loss_units', choices=['normalized','diopter'], default='normalized',
                   help='Axial direct FRONT velocity loss: normalized pixels or physical D / sia_ref')
    p.add_argument('--image_backbone', choices=['unet','fno'], default='unet',
                   help='Axial direct FM-SIA velocity network; historical models retain UNet')
    p.add_argument('--sia_magnitude_column', default='front_SIA_D')
    p.add_argument('--sia_label_atol', type=float, default=1e-4)
    p.add_argument('--skip_sia_checks', action='store_true',
                   help='Skip sia_valid and vector/magnitude consistency checks; retain numeric target validity')
    p.add_argument('--sia_embed_dim', type=int, default=32)
    p.add_argument('--sia_num_heads', type=int, default=4)
    p.add_argument('--lambda_vector', type=float, default=1.0)
    p.add_argument('--lambda_scalar', type=float, default=1.0,
                   help='Scalar SIA loss weight: standardized velocity MSE (flow) or physical MAE/sia_ref (direct)')
    p.add_argument('--lambda_sia', type=float, default=0.1)
    p.add_argument('--loss_balance', choices=['fixed','gradnorm','dwa'], default='fixed',
                   help='FM-SIA training loss weighting; fixed preserves the original lambda weights')
    p.add_argument('--balance_tasks', choices=['branches','terms'], default='branches',
                   help='branches=image vs combined SIA; terms=image/state/magnitude independently')
    p.add_argument('--gradnorm_alpha', type=float, default=1.5,
                   help='GradNorm relative training-rate exponent; 0 balances gradient norms only')
    p.add_argument('--gradnorm_lr', type=float, default=1e-3,
                   help='Separate GradNorm weight optimizer learning rate')
    p.add_argument('--dwa_temperature', type=float, default=2.0,
                   help='DWA softmax temperature, using two completed training epochs')
    p.add_argument('--sia_ref', type=float, default=1.0, help='Physical reference scale in D')
    p.add_argument('--huber_delta', type=float, default=1.0)
    p.add_argument('--invalid_zero', action='store_true', help='Use only if raw zeros explicitly mean missing pixels')
    p.add_argument('--reconstruction_domain', choices=['valid','full'], default='valid',
                   help='Training image loss only: valid=common pre/post mask; full=all pixels including filled background')
    p.add_argument('--augmentation', choices=['none','d4'], default='none')
    p.add_argument('--best_metric', choices=['MAE','SIA_MAE'], default='MAE')
    p.add_argument('--eval_every', type=int, default=10)
    p.add_argument('--checkpoint_every', type=int, default=10)
    p.add_argument('--save_checkpoints', action='store_true', help='Explicit opt-in to persist model/optimizer checkpoints; default is memory-only selection')
    p.add_argument('--resume', action='store_true', help='Resume last.pth separately within each fold')
    p.add_argument('--preflight_only', action='store_true')
    p.add_argument('--validation_only', action='store_true',
                   help='Development run: no test loader or test evaluation; report validation checkpoints only')
    p.add_argument('--save_predictions', action='store_true', help='Save all physical image predictions as .npy')
    p.add_argument('--visualize_n', type=int, default=5,
                   help='Legacy compatibility only: epoch=1, validation=5, test=all are fixed')
    p.add_argument('--visual_error_max', type=float, default=1.0,
                   help='Fixed error PNG color maximum in original map units (D for FRONT power)')
    p.add_argument('--fno_modes', type=int, nargs=2, default=[32,32])
    p.add_argument('--fno_hidden_channels', type=int, default=64)
    p.add_argument('--fno_layers', type=int, default=4)
    p.add_argument('--fno_in_channels', type=int, default=1)
    p.add_argument('--fno_out_channels', type=int, default=1)
    p.add_argument('--fno_domain_padding', type=float, default=0.05)
    args = p.parse_args(argv)
    check_args(args)
    return args


def check_args(a):
    if a.resume and not a.save_checkpoints:
        raise ValueError("--resume requires explicit --save_checkpoints")
    if a.image_backbone == 'fno' and (a.model != 'fm_sia' or a.sia_prediction != 'axial_direct'):
        raise ValueError('Conditional FNO backbone requires axial_direct FM-SIA')
    if a.image_backbone == 'fno' and (min(*a.fno_modes,a.fno_hidden_channels,a.fno_layers)<1
                                     or not math.isfinite(a.fno_domain_padding) or a.fno_domain_padding<0):
        raise ValueError('Invalid conditional FNO dimensions or padding')
    if a.image_loss_units == 'diopter' and (a.model != 'fm_sia' or a.sia_prediction != 'axial_direct'
                                           or a.input_maps != ['FRONT'] or a.output_maps != ['FRONT']):
        raise ValueError('Diopter image loss requires axial_direct FM-SIA with FRONT maps')
    if a.sia_feature_fusion == 'preop_frame' and (a.sia_prediction != 'axial_direct' or not set(PRE_VECTOR).issubset(a.conditions)):
        raise ValueError('Preoperative frame requires axial_direct and both pre_front_X/Y conditions')
    if a.sia_prediction == 'axial_direct' and (a.model != 'fm_sia' or a.sia_mode != 'vector'
                                              or a.loss_balance != 'fixed'):
        raise ValueError('Axial direct SIA requires fm_sia, vector mode and fixed balance')
    if a.sia_prediction == 'direct' and (a.model != 'fm_sia' or a.sia_mode != 'scalar'
                                       or a.loss_balance != 'fixed' or a.lambda_sia != 0):
        raise ValueError('Direct SIA requires fm_sia, scalar mode, fixed balance and lambda_sia=0')
    if a.loss_balance != 'fixed' and a.model != 'fm_sia':
        raise ValueError('Adaptive task balancing requires --model fm_sia')
    if (not all(math.isfinite(x) for x in (a.gradnorm_alpha, a.gradnorm_lr, a.dwa_temperature))
            or a.gradnorm_alpha < 0 or min(a.gradnorm_lr, a.dwa_temperature) <= 0):
        raise ValueError('Invalid GradNorm/DWA settings')
    if not math.isfinite(a.visual_error_max) or a.visual_error_max <= 0:
        raise ValueError('visual_error_max must be positive and finite')
    if len(a.input_maps) != 1 or len(a.output_maps) != 1:
        raise ValueError('Current encoder/UNet supports one input and one output map')
    if a.model in ('fm','fm_sia') and a.input_maps != a.output_maps:
        raise ValueError('Pre-to-post FM requires identical input/output map semantics')
    if a.image_size < 8 or a.image_size % 8:
        raise ValueError('image_size must be a multiple of 8 and >=8')
    if not 1 <= a.k_folds <= 10:
        raise ValueError('k_folds must be in [1,10]')
    if min(a.batch_size, a.epochs, a.sample_steps, a.eval_samples, a.eval_every, a.checkpoint_every, a.cpu_threads) < 1 or a.num_workers < 0 or not math.isfinite(a.lr) or a.lr <= 0:
        raise ValueError('Invalid training or sampling counts/lr')
    if len(set(a.conditions)) != len(a.conditions):
        raise ValueError('Duplicate clinical condition names')
    # Prevent the common leakage routes, even when clinical fusion is disabled.
    forbidden = [c for c in a.conditions if c == 'label_index' or c.lower().startswith(('post_', 'delta_'))
                 or 'sia' in c.lower() or c in a.sia_columns or c == a.sia_magnitude_column]
    if forbidden:
        raise ValueError(f'Outcome/identifier fields cannot be input conditions: {forbidden}')
    if a.fusion_method != 'none' and not a.conditions:
        raise ValueError('Clinical fusion requires --conditions')
    if a.fusion_method != 'none' and a.model in ('fno','uno'):
        raise ValueError('Neural operator clinical fusion is not implemented')
    if a.augmentation == 'd4' and any(c in PRE_VECTOR or any(w in c.lower() for w in ('axis','angle','meridian','location','eye','side','j0','j45')) for c in a.conditions):
        raise ValueError('D4 does not transform directional/eye-side clinical inputs; use --augmentation none')
    if a.time_steps is None:
        a.time_steps = 1000 if a.model == 'cddpm' else 50000
    if a.sampler == 'ddim' and a.model != 'cddpm':
        raise ValueError('DDIM is CDDPM-only. For FM-SIA remove --sampler ddim; use --ode_solver euler')
    if a.model == 'cddpm' and (a.time_steps < 2 or (a.sampler == 'ddim' and a.sample_steps > a.time_steps)):
        raise ValueError('Invalid CDDPM/DDIM steps')
    if not 0 <= a.ddim_eta <= 1 or not 0 <= a.ema_decay < 1:
        raise ValueError('Invalid eta or EMA decay')
    if a.model != 'fm_sia' and a.best_metric == 'SIA_MAE':
        raise ValueError('SIA checkpoint selection requires fm_sia')
    if a.sia_mode != 'vector' and a.model != 'fm_sia':
        raise ValueError('--sia_mode scalar requires --model fm_sia')
    active_weight = a.lambda_vector if a.sia_mode == 'vector' else a.lambda_scalar
    values = [active_weight, a.lambda_sia, a.sia_ref, a.huber_delta, a.sia_label_atol]
    if not all(math.isfinite(x) for x in values) or active_weight <= 0 or a.lambda_sia < 0 or min(a.sia_ref, a.huber_delta, a.sia_label_atol) <= 0:
        raise ValueError('Invalid SIA loss settings')
    if a.sia_num_heads < 1 or a.sia_embed_dim < 1 or a.sia_embed_dim % a.sia_num_heads:
        raise ValueError('sia_embed_dim must be divisible by sia_num_heads')
    if len(set(a.sia_columns)) != 2:
        raise ValueError('Two distinct X/Y columns required')


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed); np.random.seed(seed)


def json_save(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def resolve_device(name, cpu_threads=2):
    if name.startswith('cuda') and not torch.cuda.is_available():
        print('CUDA unavailable; using CPU')
        name = 'cpu'
    if name == 'cpu':
        torch.set_num_threads(cpu_threads)
    return torch.device(name)


def build_model(args, stats=None):
    if args.model in ('fno','uno'):
        from model.no import FourierNeuralOperator, UnetNeuralOperator
        if args.model == 'fno':
            return FourierNeuralOperator(in_channels=args.fno_in_channels, out_channels=args.fno_out_channels,
                n_layers=args.fno_layers, hidden_channels=args.fno_hidden_channels, modes=args.fno_modes)
        return UnetNeuralOperator(in_channels=args.fno_in_channels, out_channels=args.fno_out_channels,
                                   hidden_channels=args.fno_hidden_channels)
    direct_sia = args.model == 'fm_sia' and getattr(args, 'sia_prediction', 'flow') in ('direct','axial_direct')
    encoder = ConditionEncoder()
    if getattr(args, 'image_backbone', 'unet') == 'fno':
        from model.conditional_fno import ConditionalFNO
        unet = ConditionalFNO(clinical_dim=len(args.conditions),clinical_embed_dim=args.clinical_embed_dim,
            fusion_method=args.fusion_method,modes=args.fno_modes,hidden_channels=args.fno_hidden_channels,
            n_layers=args.fno_layers,domain_padding=args.fno_domain_padding)
    else:
        unet = ConditionalUNet(discrete=args.model not in ('fm','fm_sia'), clinical_dim=len(args.conditions),
            clinical_embed_dim=args.clinical_embed_dim, fusion_method=args.fusion_method,
            sia_embed_dim=args.sia_embed_dim if args.model == 'fm_sia' and not direct_sia else None,
            sia_num_heads=args.sia_num_heads,
            sia_state_dim=1 if getattr(args, 'sia_mode', 'vector') == 'scalar' else 2)
    if args.model == 'fm_sia':
        if not stats or not stats.get('sia'):
            raise ValueError('FM-SIA requires saved/fitted training-fold SIA statistics')
        if getattr(args, 'sia_prediction', 'flow') == 'axial_direct':
            from model.axial_sia import AxialSIAFlowMatching
            frame = getattr(args, 'sia_feature_fusion', 'lab')
            frame_args = dict(feature_fusion=frame)
            if frame == 'preop_frame':
                indices = [args.conditions.index(name) for name in PRE_VECTOR]
                frame_args.update(clinical_vector_indices=indices,
                    clinical_vector_mean=[stats['clinical']['mean'][i] for i in indices],
                    clinical_vector_std=[stats['clinical']['std'][i] for i in indices])
            return AxialSIAFlowMatching(encoder, unet, stats['sia']['mean'], stats['sia']['std'],
                sample_steps=args.sample_steps, method=args.ode_solver, lambda_vector=args.lambda_vector,
                lambda_sia=args.lambda_sia, sia_ref=args.sia_ref, huber_delta=args.huber_delta,
                image_loss_units=getattr(args, 'image_loss_units', 'normalized'),
                image_scale=stats['image']['scale'] if getattr(args, 'image_loss_units', 'normalized') == 'diopter' else 1.,
                **frame_args)
        if direct_sia:
            from model.direct_sia import DirectSIAFlowMatching
            return DirectSIAFlowMatching(encoder, unet, stats['sia']['mean'], stats['sia']['std'],
                initial_sia=stats['sia']['median'][0], sample_steps=args.sample_steps,
                method=args.ode_solver, lambda_scalar=args.lambda_scalar, sia_ref=args.sia_ref)
        return JointFlowMatching(encoder, unet, stats['sia']['mean'], stats['sia']['std'],
            sample_steps=args.sample_steps, method=args.ode_solver, lambda_vector=args.lambda_vector,
            lambda_sia=args.lambda_sia, sia_ref=args.sia_ref, huber_delta=args.huber_delta,
            sia_mode=getattr(args, 'sia_mode', 'vector'), lambda_scalar=getattr(args, 'lambda_scalar', 1.0))
    if args.model == 'fm':
        from model.fm import FlowMatching
        return FlowMatching(encoder, unet, method=args.ode_solver, sample_steps=args.sample_steps)
    if args.model == 'bbdm':
        from model.bbdm import BBDM
        return BBDM(encoder, unet, timesteps=args.time_steps, objective=args.objective, sample_steps=args.sample_steps)
    from model.cddpm import CDDPM
    return CDDPM(encoder, unet, timesteps=args.time_steps, sampler=args.sampler,
                 sample_steps=args.sample_steps, ddim_eta=args.ddim_eta)


def make_dataset(args, dataframe, labels, stats, training=False):
    clinical = stats.get('clinical')
    return CornealTopoDataset(args, args.excel_path, args.data_root, image_size=args.image_size,
        dataframe=dataframe, labels=labels, input_maps=args.input_maps, output_maps=args.output_maps,
        image_stats=stats['image'], clinical_mean=clinical['mean'] if clinical else None,
        clinical_std=clinical['std'] if clinical else None,
        transforms=DataTransform() if training and args.augmentation == 'd4' else None)


def make_loader(dataset, args, device, shuffle=False, generator=None):
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers,
        pin_memory=device.type == 'cuda', worker_init_fn=seed_worker, generator=generator)


def saved_config_value(config, key):
    """Recover historical defaults for checkpoints without newer options."""
    if key == 'save_checkpoints' and key not in config:
        return True
    if key == 'sia_prediction' and key not in config:
        return 'flow'
    if key == 'sia_feature_fusion' and key not in config:
        return 'lab'
    if key == 'image_loss_units' and key not in config:
        return 'normalized'
    if key == 'image_backbone' and key not in config:
        return 'unet'
    if key == 'validation_only' and key not in config:
        return False
    if key in BALANCE_DEFAULTS and key not in config:
        return BALANCE_DEFAULTS[key]
    if key == 'reconstruction_domain' and key not in config:
        return 'valid' if config.get('model') in ('fm', 'fm_sia') else 'full'
    if key not in config and key in ('sia_mode', 'lambda_scalar'):
        return {'sia_mode': 'vector', 'lambda_scalar': 1.0}[key]
    return config.get(key)


def forward_batch(model, batch, device, reconstruction_domain='valid'):
    if reconstruction_domain not in ('valid', 'full'):
        raise ValueError('reconstruction_domain must be valid or full')
    pre, post, clinical = [batch[k].to(device) for k in ('input','target','condition')]
    mask = ((batch['mask_pre']*batch['mask_post']).to(device)
            if reconstruction_domain == 'valid' else None)
    if isinstance(model, JointFlowMatching):
        if model.sia_mode == 'scalar':
            return model(pre, post, clinical=clinical, mask=mask,
                         sia_magnitude=batch['sia_magnitude'].to(device))
        return model(pre, post, batch['sia_vector'].to(device), clinical=clinical, mask=mask)
    from model.fm import FlowMatching
    if isinstance(model, FlowMatching):
        return model(pre, post, clinical=clinical, mask=mask)
    return (model(pre, post, clinical=clinical, mask=mask) if hasattr(model, 'sample')
            else model(pre, post, mask=mask))


def train_one_epoch(model, loader, optimizer, device, ema=None, reconstruction_domain='valid',
                    balancer=None):
    model.train(); totals = {}; n = 0
    shared = (gradnorm_shared_parameters(model)
              if balancer is not None and balancer.method == 'gradnorm' else ())
    for batch in tqdm(loader, desc='Train'):
        optimizer.zero_grad(set_to_none=True)
        output = forward_batch(model, batch, device, reconstruction_domain)
        loss = output['loss']
        balance_logs = {}
        if balancer is not None:
            loss, balance_logs = balancer.combine(output, shared)
            output['loss'] = loss
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite training loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if balancer is not None:
            balancer.step()
        if ema is not None:
            ema.update(model)
        count = batch['input'].shape[0]; n += count
        for key, value in output.items():
            if key.startswith('loss'):
                totals[key] = totals.get(key, 0.0) + value.detach().item()*count
        for key, value in balance_logs.items():
            totals[key] = totals.get(key, 0.0) + value.detach().item()*count
    if n == 0:
        raise ValueError('Empty training loader')
    metrics = {k: v/n for k, v in totals.items()}
    if balancer is not None:
        balancer.end_epoch(metrics)
    return metrics


@torch.no_grad()
def sample_batch(model, batch, args, device):
    pre, clinical = batch['input'].to(device), batch['condition'].to(device)
    if isinstance(model, JointFlowMatching):
        outputs = [model.sample(pre, clinical=clinical, noise=fixed_sia_noise(
            batch['label'], args.eval_seed, m, model.state_dim).to(pre)) for m in range(args.eval_samples)]
        if model.sia_mode == 'scalar':
            s = torch.stack([o['sia'] for o in outputs])
            raw = torch.stack([o['sia_unclipped'] for o in outputs])
            return dict(pred=torch.stack([o['pred'] for o in outputs]).mean(0), sia=s.mean(0),
                        sia_samples=s, sia_sample_std=s.std(0, unbiased=False),
                        sia_unclipped=raw.mean(0), sia_unclipped_samples=raw)
        # E[||q||] differs from ||E[q]||. Keep both explicitly for M>1.
        q = torch.stack([o['sia_vector'] for o in outputs])
        s = torch.stack([o['sia'] for o in outputs])
        return dict(pred=torch.stack([o['pred'] for o in outputs]).mean(0), sia_vector=q.mean(0),
                    sia=s.mean(0), sia_samples=s, sia_vector_samples=q,
                    sia_sample_std=s.std(0, unbiased=False), sia_from_mean_vector=q.mean(0).norm(dim=-1))
    pred = model.sample(pre, clinical=clinical) if hasattr(model, 'sample') else model(pre)['pred']
    return dict(pred=pred)


def labels_list(labels):
    return labels.detach().cpu().tolist() if torch.is_tensor(labels) else list(labels)


def save_batch_visuals(output_dir, stem, batch, pred, j, args, stats):
    save_visuals(output_dir, stem, batch['input'][j,0].detach().cpu().numpy(),
        batch['target'][j,0].detach().cpu().numpy(), pred[j,0].detach().cpu().numpy(),
        batch['mask_pre'][j,0].detach().cpu().numpy(), batch['mask_post'][j,0].detach().cpu().numpy(),
        stats['image'], error_max=getattr(args,'visual_error_max',1.0))


@torch.no_grad()
def visualize_epoch(model, dataset, device, args, stats, epoch, output_dir):
    """One fixed validation case, complete sampling, no effect on training RNG."""
    if not len(dataset):
        raise ValueError('Cannot visualize an empty validation set')
    was_training = model.training
    numpy_rng, python_rng = np.random.get_state(), random.getstate()
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    try:
        model.eval()
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(args.eval_seed)
            np.random.seed(args.eval_seed); random.seed(args.eval_seed)
            batch = default_collate([dataset[0]])
            out = sample_batch(model,batch,args,device)
            stem = f'epoch_{epoch:04d}'
            save_batch_visuals(output_dir,stem,batch,out['pred'],0,args,stats)
            label = labels_list(batch['label'])[0]
            json_save(Path(output_dir)/f'{stem}.json',dict(epoch=epoch,label_index=label,
                partition='val',selection='first validation sample, unchanged across epochs',
                eval_seed=args.eval_seed,eval_samples=args.eval_samples,stem=stem,
                visualization=visualization_spec(stats['image'],getattr(args,'visual_error_max',1.0))))
    finally:
        model.train(was_training)
        np.random.set_state(numpy_rng); random.setstate(python_rng)


@torch.no_grad()
def evaluate(model, loader, device, args, stats, output_dir=None, *, partition='test'):
    if partition not in ('val','test'):
        raise ValueError('Evaluation partition must be val or test')
    model.eval()
    acc = MetricAccumulator(stats['image']['scale'])
    rows, count, visual_count = [], 0, 0
    output_dir = Path(output_dir) if output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        json_save(output_dir/'visualization.json',dict(
            **visualization_spec(stats['image'],getattr(args,'visual_error_max',1.0)),
            partition=partition,sample_selection='first 5 samples' if partition=='val' else 'all samples'))
    # Isolate evaluation RNG from training. FM-SIA additionally uses per-label noise.
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(args.eval_seed)
        for batch in tqdm(loader, desc=partition.capitalize()):
            out = sample_batch(model, batch, args, device)
            target = batch['target'].to(device)
            mask = (batch['mask_pre']*batch['mask_post']).to(device)
            q_true = batch['sia_vector'].to(device) if 'sia_vector' in out else None
            s_true = (q_true.norm(dim=-1) if q_true is not None else
                      batch['sia_magnitude'].to(device) if 'sia' in out else None)
            acc.update(out['pred'], target, mask, out.get('sia_vector'), q_true, out.get('sia'), s_true)
            for j, label in enumerate(labels_list(batch['label'])):
                row = dict(label_index=label)
                if 'sia' in out:
                    st, sp = s_true[j].item(), out['sia'][j].item()
                    row.update(sia_mode=getattr(model, 'sia_mode', 'vector'), true_SIA_D=st,
                        pred_SIA_D=sp, SIA_error_D=sp-st, SIA_absolute_error_D=abs(sp-st),
                        SIA_sample_std_D=out['sia_sample_std'][j].item(), samples=args.eval_samples)
                    if q_true is not None:
                        qt, qp = q_true[j].cpu().numpy(), out['sia_vector'][j].cpu().numpy()
                        axes = axis_details(qp[None], qt[None])
                        row.update(true_delta_X=float(qt[0]), true_delta_Y=float(qt[1]),
                            pred_delta_X=float(qp[0]), pred_delta_Y=float(qp[1]),
                            SIA_from_mean_vector_D=out['sia_from_mean_vector'][j].item(),
                            true_axis_deg=float(axes['true'][0]), pred_axis_deg=float(axes['pred'][0]),
                            axis_error_deg=float(axes['error'][0]), axis_gt_ge_0p1D=bool(st >= .1))
                    else:
                        row.update(pred_SIA_unclipped_D=out['sia_unclipped'][j].item(),
                            SIA_negative_sample_fraction=float((out['sia_unclipped_samples'][:, j] < 0).float().mean()))
                stem = f'{count:06d}'
                row.update(sample_index=count,png_stem='')
                if output_dir:
                    if args.save_predictions:
                        raw = out['pred'][j].cpu().numpy()*stats['image']['scale']+stats['image']['minimum']
                        # Inference uses pre validity only; never use target validity to generate predictions.
                        raw = np.where(batch['mask_pre'][j].numpy().astype(bool), raw, np.nan)
                        np.save(output_dir/f'{stem}_pred_raw.npy', raw[0])
                        if 'sia' in out:
                            samples = dict(s=out['sia_samples'][:, j].cpu().numpy())
                            if 'sia_vector_samples' in out:
                                samples['q'] = out['sia_vector_samples'][:, j].cpu().numpy()
                            else:
                                samples['s_unclipped'] = out['sia_unclipped_samples'][:, j].cpu().numpy()
                            np.savez(output_dir/f'{stem}_sia_samples.npz', **samples)
                    if partition=='test' or count < 5:
                        save_batch_visuals(output_dir,stem,batch,out['pred'],j,args,stats)
                        row['png_stem'] = stem
                        visual_count += 1
                rows.append(row)
                count += 1
    metrics = acc.compute()
    if output_dir:
        pd.DataFrame(rows).to_csv(output_dir/'predictions.csv', index=False, encoding='utf-8-sig')
        json_save(output_dir/'metrics.json', metrics)
        json_save(output_dir/'evaluation.json', dict(samples=count, ssim_valid_images=acc.ssim_count,
            partition=partition,visualized_samples=visual_count,visualization_files=4*visual_count,
            ssim_full_images=acc.full_ssim_count,
            valid_pixels=acc.pixels, full_pixels=acc.full_pixels,
            metric_reporting_version='full_and_valid_v1',
            reconstruction_domain=saved_config_value(vars(args), 'reconstruction_domain'),
            pixel_domains=dict(valid='common finite pre/post region',
                full='all pixels in the supplied image tensor, including normalized zero-filled invalid targets'),
            metric_aliases='unsuffixed image metrics are valid-region aliases; raw denotes original units',
            psnr_data_range=1.0, psnr_aggregation='pooled pixel MSE per domain; floor 1e-12',
            ssim_aggregation='equal mean over image-channel scores with eligible windows',
            eval_seed=args.eval_seed, eval_samples=args.eval_samples, steps=args.sample_steps,
            solver=args.ode_solver, pixel_domain='common finite pre/post region',
            image_scale=stats['image'], sia_mode=getattr(args, 'sia_mode', 'vector'),
            sia_prediction=getattr(args, 'sia_prediction', 'flow'),
            axis_protocol=dict(period_degrees=180, zero_epsilon_D=1e-8, stable_gt_threshold_D=.1,
                undefined_prediction_penalty_degrees=90, vector_for_axis='mean predicted physical vector'),
            SIA_point_estimator=('deterministic preoperative-only spatial-moment vector head' if getattr(args, 'sia_prediction', 'flow') == 'axial_direct'
                                 else 'deterministic preoperative-only magnitude head' if getattr(args, 'sia_prediction', 'flow') == 'direct'
                                 else 'mean of nonnegative projected scalar samples' if getattr(args, 'sia_mode', 'vector') == 'scalar'
                                 else 'mean of per-sample norms'),
            sample_std_is_calibrated_uncertainty=False))
    return metrics


def checkpoint_payload(args, fold, stats, model, ema, optimizer, scheduler, epoch, best, generator,
                       balancer=None):
    return dict(pipeline_version=PIPELINE_VERSION, config=vars(args), model_type=args.model,
        fold=fold, stats=stats, epoch=epoch, model=model.state_dict(),
        ema=ema.ema_model.state_dict() if ema else None, optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(), best=best.copy(),
        task_balancer=balancer.state_dict() if balancer is not None else None,
        rng=dict(torch=torch.get_rng_state(), numpy=np.random.get_state(), python=random.getstate(),
                 cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                 loader=generator.get_state()))


def torch_save_atomic(payload, path):
    path = Path(path); temporary = path.with_name(path.name+'.tmp')
    torch.save(payload, temporary); os.replace(temporary, path)


def cpu_weight_snapshot(model):
    """Detached in-memory weights; preserve state_dict version metadata."""
    state = model.state_dict()
    for key, value in state.items():
        state[key] = value.detach().to(device='cpu', copy=True)
    return state


def checkpoint_link(source, destination):
    temporary = destination.with_name(destination.name+'.linktmp')
    if temporary.exists(): temporary.unlink()
    os.link(source, temporary)
    os.replace(temporary, destination)


def restore_rng(rng, generator):
    torch.set_rng_state(rng['torch']); np.random.set_state(rng['numpy']); random.setstate(rng['python'])
    generator.set_state(rng['loader'])
    if rng['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng['cuda'])


def fit_sia_stats(args, df, labels):
    if args.model != 'fm_sia':
        return None
    columns = sia_target_columns(args)
    stats = feature_stats(df, labels, columns)
    if getattr(args, 'sia_prediction', 'flow') == 'direct':
        stats['median'] = df[df.label_index.isin(labels)][columns].median().tolist()
    return stats


def run_fold(args, df, fold, device):
    fold_started = time.perf_counter()
    number = fold['fold']; folder = Path(args.save_dir)/f'fold_{number:02d}'
    folder.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed+number)
    generator = torch.Generator().manual_seed(args.seed+number)
    last = folder/'last.pth'
    if (folder/'history.json').exists() and not args.resume:
        raise FileExistsError('Experiment history already exists; choose a new save_dir')
    if args.resume and not last.exists():
        raise FileNotFoundError('Cannot resume without last.pth; checkpoint-free runs must restart in a new directory')
    ckpt = None
    memory_best = None
    if args.resume and last.exists():
        ckpt = torch.load(last, map_location='cpu', weights_only=False)
        if ckpt['pipeline_version'] != PIPELINE_VERSION or ckpt['fold'] != fold:
            raise ValueError('Checkpoint pipeline/fold does not match')
        allowed = {'resume','device','num_workers','cpu_threads','save_dir','visualize_n','visual_error_max','save_predictions','k_folds','preflight_only','skip_sia_checks'}
        changed = [k for k,v in vars(args).items() if k not in allowed and saved_config_value(ckpt['config'], k) != v]
        if changed:
            raise ValueError(f'Resume configuration changed: {changed}; use a new save_dir')
        stats = ckpt['stats']
    else:
        if any(folder.glob('*.pth')):
            raise FileExistsError(f'{folder} already contains checkpoints; use --resume or a new save_dir')
        stats = dict(image=fit_image_stats(args, fold['train']),
                     clinical=feature_stats(df, fold['train'], args.conditions),
                     sia=fit_sia_stats(args, df, fold['train']))
    json_save(folder/'normalization.json', stats)
    partitions = ('train','val') if args.validation_only else ('train','val','test')
    datasets = {part: make_dataset(args, df, fold[part], stats, training=part=='train') for part in partitions}
    loaders = {part: make_loader(ds, args, device, shuffle=part=='train', generator=generator if part=='train' else None)
               for part,ds in datasets.items()}
    model = build_model(args, stats).to(device)
    balancer = build_task_balancer(args, model, device)
    ema = EMA(model, args.ema_decay) if args.use_ema else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best = dict(MAE=float('inf'))
    if args.model == 'fm_sia': best['SIA_MAE'] = float('inf')
    start = 0
    if ckpt:
        model.load_state_dict(ckpt['model'])
        if ema: ema.ema_model.load_state_dict(ckpt['ema'])
        optimizer.load_state_dict(ckpt['optimizer']); scheduler.load_state_dict(ckpt['scheduler'])
        if balancer is not None:
            if ckpt.get('task_balancer') is not None:
                balancer.load_state_dict(ckpt['task_balancer'])
            elif balancer.method != 'fixed':
                raise ValueError('Adaptive balancing state is missing; use a new save_dir')
        best, start = ckpt['best'], ckpt['epoch']+1
        restore_rng(ckpt['rng'], generator)
        del ckpt
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(folder/'runs'))
    history_path = folder/'history.json'
    history = json.loads(history_path.read_text(encoding='utf-8')) if start and history_path.exists() else []
    history = [h for h in history if h['epoch'] <= start]
    print(f'Fold {number}: '+', '.join(f'{p}={len(fold[p])}' for p in ('train','val','test')))
    try:
        for epoch in range(start, args.epochs):
            if device.type == 'cuda': torch.cuda.synchronize(device)
            train_started = time.perf_counter()
            losses = train_one_epoch(model, loaders['train'], optimizer, device, ema,
                                     reconstruction_domain=args.reconstruction_domain, balancer=balancer)
            if device.type == 'cuda': torch.cuda.synchronize(device)
            train_seconds = time.perf_counter() - train_started
            scheduler.step()
            for k,v in losses.items(): writer.add_scalar(f'train/{k}', v, epoch+1)
            entry = dict(epoch=epoch+1, train=losses, train_seconds=train_seconds)
            eval_model = ema.ema_model if ema else model
            if epoch == 0 or (epoch+1) % args.eval_every == 0 or epoch+1 == args.epochs:
                visualize_epoch(eval_model,datasets['val'],device,args,stats,epoch+1,folder/'visual')
                metrics = evaluate(eval_model, loaders['val'], device, args, stats,
                    folder/'val_results'/f'epoch_{epoch+1:04d}',partition='val')
                entry['validation'] = metrics
                for k,v in metrics.items():
                    if v is not None: writer.add_scalar(f'validation/{k}', v, epoch+1)
                improved = [k for k in best if metrics[k] < best[k]]
                for k in improved: best[k] = metrics[k]
                if args.save_checkpoints:
                    payload = checkpoint_payload(args, fold, stats, model, ema, optimizer, scheduler, epoch, best, generator, balancer)
                    for k in improved:
                        torch_save_atomic(payload, folder/('best_image.pth' if k=='MAE' else 'best_sia.pth'))
                    del payload
                elif args.best_metric in improved:
                    memory_best = (cpu_weight_snapshot(eval_model), epoch+1)
                print(f'Fold {number} epoch {epoch+1}: {losses}; validation={metrics}')
            history.append(entry); json_save(history_path, history)
            if args.save_checkpoints and ((epoch+1) % args.checkpoint_every == 0 or epoch+1 == args.epochs or 'validation' in entry):
                torch_save_atomic(checkpoint_payload(args, fold, stats, model, ema, optimizer, scheduler, epoch, best, generator, balancer), last)
    finally:
        writer.close()
    if args.save_checkpoints:
        checkpoint_link(last, folder/'final.pth')
        checkpoint_link(folder/('best_image.pth' if args.best_metric=='MAE' else 'best_sia.pth'), folder/'best.pth')
    else:
        memory_final = (cpu_weight_snapshot(ema.ema_model if ema else model), args.epochs)
    # Evaluate only a predeclared primary checkpoint plus final, never select on test data.
    results = {}
    del optimizer, scheduler, ema
    for name in ('best','final'):
        if args.save_checkpoints:
            state = torch.load(folder/f'{name}.pth', map_location='cpu', weights_only=False)
            weights = state['ema'] if args.use_ema else state['model']
            model.load_state_dict(weights); selected_epoch = state['epoch']+1
            del state, weights
        else:
            weights, selected_epoch = memory_best if name == 'best' else memory_final
            model.load_state_dict(weights)
        json_save(folder/f'{name}_selection.json', dict(epoch=selected_epoch,
            metric=args.best_metric if name == 'best' else None, persisted=args.save_checkpoints))
        partition = 'val' if args.validation_only else 'test'
        result_dir = folder/('validation_results' if args.validation_only else 'test_results')
        metrics = evaluate(model, loaders[partition], device, args, stats, result_dir/name,partition=partition)
        results[name] = metrics
        json_save(result_dir/f'{name}.json', metrics)
        print(f'Fold {number} {name} epoch {selected_epoch}: {metrics}')
    timing = dict(train_seconds=sum(h.get('train_seconds',0) for h in history),
        fold_wall_seconds=time.perf_counter()-fold_started, timed_epochs=sum('train_seconds' in h for h in history),
        definition='train_seconds includes data loading, forward/backward, optimizer, EMA; excludes validation, plots, checkpoints and test; fold wall includes all')
    json_save(folder/'timing.json',timing)
    return dict(fold=number, sizes={p:len(fold[p]) for p in ('train','val','test')}, timing=timing, **results)


def write_summary(results, path, args, split_file):
    aggregate = {}
    for name in ('best','final'):
        aggregate[name] = {}
        for metric in results[0][name]:
            values = [r[name][metric] for r in results if r[name][metric] is not None]
            aggregate[name][metric] = dict(mean=float(np.mean(values)) if values else None,
                std=float(np.std(values)) if values else None, n=len(values))
    summary = dict(pipeline_version=PIPELINE_VERSION, model=args.model, split_file=str(split_file),
        reporting_partition='val' if args.validation_only else 'test',
        sia_mode=getattr(args, 'sia_mode', 'vector'),
        sia_prediction=getattr(args, 'sia_prediction', 'flow'),
        reconstruction_domain=saved_config_value(vars(args), 'reconstruction_domain'),
        best_metric=args.best_metric, fusion_method=args.fusion_method, clinical_features=args.conditions,
        clinical_embed_dim=args.clinical_embed_dim, sample_steps=args.sample_steps, sampler=args.sampler,
        time_steps=args.time_steps, ddim_eta=args.ddim_eta, config=vars(args),
        requested_folds=args.k_folds, completed_folds=len(results), fold_results=results, aggregate=aggregate)
    json_save(path, summary)
    return summary


def main():
    args = get_args()
    seed_everything(args.seed)
    device = resolve_device(args.device, args.cpu_threads)
    df = load_dataframe(args.excel_path, args.conditions)
    report = validate_table(df, args)
    validate_maps(df, args)
    # Byte fingerprint prevents silent resume after changing labels/clinical values.
    args.excel_sha256 = hashlib.sha256(Path(args.excel_path).read_bytes()).hexdigest()
    split_file = Path(args.split_file) if args.split_file else Path(args.data_root)/'patient_10fold_splits.json'
    folds = load_or_create_cv_splits(df, split_file, seed=args.seed)
    folder = Path(args.save_dir); folder.mkdir(parents=True, exist_ok=True)
    if not args.preflight_only and not args.resume and any(folder.glob('fold_*/history.json')):
        raise FileExistsError('Experiment history already exists; choose a new save_dir')
    if not args.preflight_only and any(folder.glob('fold_*/*.pth')):
        if not args.resume:
            raise FileExistsError(f'{folder} already contains checkpoints; use --resume or a new save_dir')
        if (folder/'config.json').exists():
            old_config = json.loads((folder/'config.json').read_text(encoding='utf-8'))
            allowed = {'resume','device','num_workers','cpu_threads','save_dir','visualize_n','visual_error_max',
                       'save_predictions','k_folds','preflight_only','skip_sia_checks'}
            changed = [k for k,v in vars(args).items() if k not in allowed and saved_config_value(old_config, k) != v]
            if changed:
                raise ValueError(f'Resume configuration changed: {changed}; original results preserved')
    report.update(pipeline_version=PIPELINE_VERSION, input_maps=args.input_maps, output_maps=args.output_maps,
                  reconstruction_domain=args.reconstruction_domain,
                  augmentation=args.augmentation, excel_sha256=args.excel_sha256)
    json_save(folder/'preflight.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.preflight_only:
        print('Preflight passed; no model training performed'); return
    json_save(folder/'config.json', vars(args))
    results = []
    for fold in folds[:args.k_folds]:
        results.append(run_fold(args, df, fold, device))
        write_summary(results, folder/'cv_summary.json', args, split_file)
    print(f'Completed {len(results)} folds: {folder / "cv_summary.json"}')


if __name__ == '__main__':
    main()



