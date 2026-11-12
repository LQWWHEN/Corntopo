"""CPTP/mCPTP: fixed patient-level 10-fold splits, first K folds, FM_SIA metrics."""
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
                        validate_table, validate_maps)
from transforms import DataTransform
from patient_cv import load_cv_splits
from metrics import MetricAccumulator
from sia_metrics import SIAMetricAccumulator
from visualization import save_visuals, visualization_spec
from models import create_model
from ema import EMA
from options.train_options import TrainOptions

PIPELINE_VERSION = 'cgan_fm_sia_comparison_v2'


def load_dataframe(path):
    return pd.read_csv(path) if Path(path).suffix.lower() == '.csv' else pd.read_excel(path)


def get_args(argv=None):
    return TrainOptions().parse(argv)


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
    return create_model(args, stats)


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


def train_one_epoch(model, loader, optimizer, device, ema=None):
    model.train(); totals = {}; n = 0
    for batch in tqdm(loader, desc='Train'):
        pre, post, clinical = [batch[k].to(device) for k in ('input','target','condition')]
        mask = (batch['mask_pre'] * batch['mask_post']).to(device)
        output = model(pre, clinical)
        model.netD.requires_grad_(True)
        optimizer['D'].zero_grad(set_to_none=True)
        loss_d = model.discriminator_loss(pre, post, output, mask)
        if not torch.isfinite(loss_d):
            raise FloatingPointError('Non-finite discriminator loss')
        loss_d.backward()
        torch.nn.utils.clip_grad_norm_(model.netD.parameters(), 1.0, error_if_nonfinite=True)
        optimizer['D'].step()
        model.netD.requires_grad_(False)
        # Avoid a third BatchNorm running-stat update while optimizing G.
        model.netD.eval()
        optimizer['G'].zero_grad(set_to_none=True)
        losses = model.generator_loss(pre, post, output, mask, batch['sia_vector'].to(device),
                                      batch['sia_magnitude'].to(device) if 'sia_magnitude' in batch else None)
        if not torch.isfinite(losses['loss']):
            raise FloatingPointError('Non-finite generator loss')
        losses['loss'].backward()
        torch.nn.utils.clip_grad_norm_(list(model.generator_parameters()), 1.0, error_if_nonfinite=True)
        optimizer['G'].step()
        model.netD.train()
        losses['loss_D'] = loss_d
        if ema is not None:
            ema.update(model)
        count = pre.shape[0]; n += count
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + value.detach().item()*count
    return {k: v/n for k, v in totals.items()}


@torch.no_grad()
def sample_batch(model, batch, args, device):
    out = model.sample(batch['input'].to(device), batch['condition'].to(device))
    if 'sia' in out:
        out.update(sia_samples=out['sia'].unsqueeze(0), sia_sample_std=torch.zeros_like(out['sia']))
        if 'sia_vector' in out:
            out.update(sia_vector_samples=out['sia_vector'].unsqueeze(0),sia_from_mean_vector=out['sia'])
    return out


def labels_list(labels):
    return labels.detach().cpu().tolist() if torch.is_tensor(labels) else list(labels)


def sample_image_stats(batch, j, stats):
    """Per-case physical decoding; old callers without metadata use global stats."""
    return dict(minimum=float(batch['image_minimum'][j]) if 'image_minimum' in batch else stats['image']['minimum'],
                scale=float(batch['image_scale'][j]) if 'image_scale' in batch else stats['image']['scale'])


def metric_reference_images(pred, target, batch, stats):
    """Evaluate in the SAME train-fold reference space for both model modes.

    Decode per-case outputs before re-encoding in the global reference. Invalid
    GT remains zero in this reference (the same full-frame convention as before).
    Raw MAE/RMSE from MetricAccumulator therefore remain physically meaningful.
    """
    if 'image_scale' not in batch:
        return pred,target
    minimum=torch.as_tensor(batch['image_minimum'],device=pred.device,dtype=torch.float64).reshape(-1,1,1,1)
    scale=torch.as_tensor(batch['image_scale'],device=pred.device,dtype=torch.float64).reshape(-1,1,1,1)
    ref=stats['image']
    if torch.all(minimum==ref['minimum']) and torch.all(scale==ref['scale']):
        return pred,target
    p=(pred.double()*scale+minimum-ref['minimum'])/ref['scale']
    y=(target.double()*scale+minimum-ref['minimum'])/ref['scale']
    y=torch.where(batch['mask_post'].to(device=pred.device,dtype=torch.bool),y,torch.zeros_like(y))
    return p,y


def save_batch_visuals(output_dir, stem, batch, pred, j, args, stats, *, save_metadata=True):
    norm=sample_image_stats(batch,j,stats)
    case_stats=(dict(minimum=float(batch['case_minimum'][j]),scale=float(batch['case_scale'][j]))
                if 'case_minimum' in batch else None)
    spec=save_visuals(output_dir, stem, batch['input'][j,0].detach().cpu().numpy(),
        batch['target'][j,0].detach().cpu().numpy(), pred[j,0].detach().cpu().numpy(),
        batch['mask_pre'][j,0].detach().cpu().numpy(), batch['mask_post'][j,0].detach().cpu().numpy(),
        stats['image'], error_max=getattr(args,'visual_error_max',1.0),normalization_stats=norm,
        visual_scale=getattr(args,'visual_scale','global'),case_stats=case_stats)
    spec.update(label_index=labels_list(batch['label'])[j],
                image_normalization=getattr(args,'image_normalization','global'))
    if save_metadata:
        json_save(Path(output_dir)/f'{stem}_visualization.json',spec)
    return spec


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
            save_batch_visuals(output_dir,stem,batch,out['pred'],0,args,stats,save_metadata=False)
    finally:
        model.train(was_training)
        np.random.set_state(numpy_rng); random.setstate(python_rng)


@torch.no_grad()
def evaluate(model, loader, device, args, stats, output_dir=None, *, partition='test'):
    if partition not in ('val','test'):
        raise ValueError('Evaluation partition must be val or test')
    model.eval()
    acc = SIAMetricAccumulator(stats['image']['scale'])
    rows, count, visual_count = [], 0, 0
    output_dir = Path(output_dir) if output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    if output_dir and partition == 'test':
        json_save(output_dir/'visualization.json',dict(
            **visualization_spec(stats['image'],getattr(args,'visual_error_max',1.0),getattr(args,'visual_scale','global')),
            partition=partition,sample_selection='first 5 samples' if partition=='val' else 'all samples'))
    # Isolate evaluation RNG from training. FM-SIA additionally uses per-label noise.
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(args.eval_seed)
        for batch in tqdm(loader, desc=partition.capitalize()):
            out = sample_batch(model, batch, args, device)
            target = batch['target'].to(device)
            mask = (batch['mask_pre']*batch['mask_post']).to(device)
            q_true = batch['sia_vector'].to(device) if 'sia' in out else None
            metric_pred,metric_target=metric_reference_images(out['pred'],target,batch,stats)
            s_true = batch['sia_magnitude'].to(device) if 'sia' in out else None
            acc.update(metric_pred, metric_target, mask, out.get('sia_vector'), q_true, out.get('sia'), s_true)
            for j, label in enumerate(labels_list(batch['label'])):
                # Validation returns metrics for selection, but exports only PNGs.
                if partition == 'val':
                    if output_dir and count < 5:
                        save_batch_visuals(output_dir,f'{count:06d}',batch,out['pred'],j,args,stats,
                                           save_metadata=False)
                        visual_count += 1
                    count += 1
                    continue
                row = dict(label_index=label)
                single = MetricAccumulator(stats['image']['scale'])
                single.update(metric_pred[j:j+1], metric_target[j:j+1], mask[j:j+1])
                row.update(single.compute())
                norm=sample_image_stats(batch,j,stats)
                row.update(image_minimum=norm['minimum'],image_scale=norm['scale'])
                if 'sia' in out:
                    st = (float(np.linalg.norm(q_true[j].cpu().numpy().astype(np.float64)))
                          if q_true.shape[-1] == 2 else s_true[j].item())
                    sp = out['sia'][j].item()
                    row.update(true_SIA_D=st, pred_SIA_D=sp, SIA_error_D=sp-st,
                        SIA_absolute_error_D=abs(sp-st), SIA_sample_std_D=0.0, samples=1,
                        sia_mode=args.sia_mode)
                    if 'sia_vector' in out:
                        qt, qp = q_true[j].cpu().numpy(), out['sia_vector'][j].cpu().numpy()
                        row.update(true_delta_X=float(qt[0]), true_delta_Y=float(qt[1]),
                            pred_delta_X=float(qp[0]), pred_delta_Y=float(qp[1]),
                            SIA_from_mean_vector_D=sp)
                stem = f'{count:06d}'
                row.update(sample_index=count,png_stem='')
                if output_dir:
                    if args.save_predictions:
                        raw = out['pred'][j].cpu().numpy()*norm['scale']+norm['minimum']
                        # Inference uses pre validity only; never use target validity to generate predictions.
                        raw = np.where(batch['mask_pre'][j].numpy().astype(bool), raw, np.nan)
                        np.save(output_dir/f'{stem}_pred_raw.npy', raw[0])
                        if 'sia' in out:
                            samples = dict(s=out['sia_samples'][:, j].cpu().numpy())
                            if 'sia_vector_samples' in out:
                                samples['q'] = out['sia_vector_samples'][:, j].cpu().numpy()
                            np.savez(output_dir/f'{stem}_sia_samples.npz', **samples)
                    if partition=='test' or count < 5:
                        save_batch_visuals(output_dir,stem,batch,out['pred'],j,args,stats)
                        row['png_stem'] = stem
                        visual_count += 1
                rows.append(row)
                count += 1
    metrics = acc.compute()
    if output_dir and partition == 'test':
        pd.DataFrame(rows).to_csv(output_dir/'predictions.csv', index=False, encoding='utf-8-sig')
        json_save(output_dir/'metrics.json', metrics)
        json_save(output_dir/'evaluation.json', dict(samples=count, ssim_valid_images=acc.ssim_count,
            partition=partition,visualized_samples=visual_count,visualization_files=4*visual_count,
            ssim_full_images=acc.full_ssim_count,
            valid_pixels=acc.pixels, full_pixels=acc.full_pixels,
            metric_reporting_version='full_and_valid_global_reference_v2',
            image_normalization=getattr(args,'image_normalization','global'),
            metric_reference='physical predictions re-encoded with training-fold global min/max for all model modes',
            pixel_domains=dict(valid='common finite pre/post region',
                full='all pixels in the supplied image tensor, including normalized zero-filled invalid targets'),
            metric_aliases='unsuffixed image metrics are valid-region aliases; raw denotes original units',
            psnr_data_range=1.0, psnr_aggregation='pooled pixel MSE per domain; floor 1e-12',
            ssim_aggregation='equal mean over image-channel scores with eligible windows',
            eval_seed=args.eval_seed, eval_samples=1, steps=0,
            solver='deterministic_cgan', pixel_domain='common finite pre/post region',
            image_scale=stats['image'], sia_mode=args.sia_mode,
            SIA_point_estimator={'scalar':'direct deterministic magnitude regression',
                'vector':'norm of deterministic physical X/Y prediction','off':'disabled'}[args.sia_mode],
            sample_std_is_calibrated_uncertainty=False))
    return metrics


def checkpoint_payload(args, fold, stats, model, ema, optimizer, scheduler, epoch, best, generator):
    return dict(pipeline_version=PIPELINE_VERSION, config=vars(args), model_type=args.model,
        fold=fold, stats=stats, epoch=epoch, model=model.state_dict(),
        ema=ema.ema_model.state_dict() if ema else None, optimizer={k: v.state_dict() for k,v in optimizer.items()},
        scheduler={k: v.state_dict() for k,v in scheduler.items()}, best=best.copy(),
        rng=dict(torch=torch.get_rng_state(), numpy=np.random.get_state(), python=random.getstate(),
                 cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                 loader=generator.get_state()))


def torch_save_atomic(payload, path):
    path = Path(path); temporary = path.with_name(path.name+'.tmp')
    torch.save(payload, temporary); os.replace(temporary, path)


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


def run_fold(args, df, fold, device):
    fold_started = time.perf_counter()
    number = fold['fold']; folder = Path(args.save_dir)/f'fold_{number:02d}'
    folder.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed+number)
    generator = torch.Generator().manual_seed(args.seed+number)
    last = folder/'last.pth'
    ckpt = None
    if args.resume and last.exists():
        ckpt = torch.load(last, map_location='cpu', weights_only=False)
        if ckpt['pipeline_version'] != PIPELINE_VERSION or ckpt['fold'] != fold:
            raise ValueError('Checkpoint pipeline/fold does not match')
        allowed = {'resume','device','num_workers','cpu_threads','save_dir','visualize_n','visual_error_max','visual_scale','save_predictions','k_folds','preflight_only','skip_sia_checks'}
        changed = [k for k,v in vars(args).items() if k not in allowed and ckpt['config'].get(k,'global' if k=='image_normalization' else None) != v]
        if changed:
            raise ValueError(f'Resume configuration changed: {changed}; use a new save_dir')
        stats = ckpt['stats']
    else:
        if any(folder.glob('*.pth')):
            raise FileExistsError(f'{folder} already contains checkpoints; use --resume or a new save_dir')
        stats = dict(image=fit_image_stats(args, fold['train']),
                     clinical=feature_stats(df, fold['train'], args.conditions),
                     sia=feature_stats(df, fold['train'], args.sia_columns) if args.sia_mode == 'vector' else None)
    json_save(folder/'normalization.json', stats)
    datasets = {part: make_dataset(args, df, fold[part], stats, training=part=='train') for part in ('train','val','test')}
    loaders = {part: make_loader(ds, args, device, shuffle=part=='train', generator=generator if part=='train' else None)
               for part,ds in datasets.items()}
    model = build_model(args, stats).to(device)
    ema = EMA(model, args.ema_decay) if args.use_ema else None
    optimizer = {
        'G': torch.optim.AdamW(model.generator_parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(args.beta1,args.beta2)),
        'D': torch.optim.AdamW(model.netD.parameters(), lr=args.lr_D, weight_decay=args.weight_decay, betas=(args.beta1,args.beta2))}
    scheduler = {k: torch.optim.lr_scheduler.CosineAnnealingLR(v, T_max=args.epochs) for k,v in optimizer.items()}
    best = dict(MAE=float('inf'))
    if args.sia_mode != 'off': best['SIA_MAE'] = float('inf')
    start = 0
    if ckpt:
        model.load_state_dict(ckpt['model'])
        if ema: ema.ema_model.load_state_dict(ckpt['ema'])
        for k in optimizer:
            optimizer[k].load_state_dict(ckpt['optimizer'][k]); scheduler[k].load_state_dict(ckpt['scheduler'][k])
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
            losses = train_one_epoch(model, loaders['train'], optimizer, device, ema)
            if device.type == 'cuda': torch.cuda.synchronize(device)
            train_seconds = time.perf_counter() - train_started
            for sched in scheduler.values(): sched.step()
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
                payload = checkpoint_payload(args, fold, stats, model, ema, optimizer, scheduler, epoch, best, generator)
                for k in improved:
                    torch_save_atomic(payload, folder/('best_image.pth' if k=='MAE' else 'best_sia.pth'))
                del payload
                print(f'Fold {number} epoch {epoch+1}: {losses}; validation={metrics}')
            history.append(entry); json_save(history_path, history)
            if (epoch+1) % args.checkpoint_every == 0 or epoch+1 == args.epochs or 'validation' in entry:
                torch_save_atomic(checkpoint_payload(args, fold, stats, model, ema, optimizer, scheduler, epoch, best, generator), last)
    finally:
        writer.close()
    checkpoint_link(last, folder/'final.pth')
    checkpoint_link(folder/('best_image.pth' if args.best_metric=='MAE' else 'best_sia.pth'), folder/'best.pth')
    # Evaluate only a predeclared primary checkpoint plus final, never select on test data.
    results = {}
    del optimizer, scheduler, ema
    for name in ('best','final'):
        state = torch.load(folder/f'{name}.pth', map_location='cpu', weights_only=False)
        weights = state['ema'] if args.use_ema else state['model']
        model.load_state_dict(weights); selected_epoch = state['epoch']+1
        del state, weights
        metrics = evaluate(model, loaders['test'], device, args, stats, folder/'test_results'/name,partition='test')
        json_save(folder/'test_results'/name/'checkpoint.json',dict(epoch=selected_epoch,
            checkpoint=name,used_ema=args.use_ema,fold=number))
        results[name] = metrics
        json_save(folder/'test_results'/f'{name}.json', metrics)
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
        image_normalization=getattr(args,'image_normalization','global'),
        metric_reference='training-fold global min/max after physical decoding',
        best_metric=args.best_metric, fusion_method=args.fusion_method, clinical_features=args.conditions,
        clinical_embed_dim=args.clinical_embed_dim, sia_mode=args.sia_mode, config=vars(args),
        requested_folds=args.k_folds, completed_folds=len(results), fold_results=results, aggregate=aggregate)
    json_save(path, summary)
    return summary


def main():
    args = get_args()
    seed_everything(args.seed)
    device = resolve_device(args.device, args.cpu_threads)
    df = load_dataframe(args.excel_path)
    report = validate_table(df, args)
    validate_maps(df, args)
    # Byte fingerprint prevents silent resume after changing labels/clinical values.
    args.excel_sha256 = hashlib.sha256(Path(args.excel_path).read_bytes()).hexdigest()
    split_file = Path(args.split_file)
    folds = load_cv_splits(df, split_file, seed=args.seed)
    args.split_sha256 = hashlib.sha256(split_file.read_bytes()).hexdigest()
    folder = Path(args.save_dir); folder.mkdir(parents=True, exist_ok=True)
    if not args.preflight_only and any(folder.glob('fold_*/*.pth')):
        if not args.resume:
            raise FileExistsError(f'{folder} already contains checkpoints; use --resume or a new save_dir')
        if (folder/'config.json').exists():
            old_config = json.loads((folder/'config.json').read_text(encoding='utf-8'))
            allowed = {'resume','device','num_workers','cpu_threads','save_dir','visualize_n','visual_error_max','visual_scale',
                       'save_predictions','k_folds','preflight_only','skip_sia_checks'}
            changed = [k for k,v in vars(args).items() if k not in allowed and old_config.get(k,'global' if k=='image_normalization' else None) != v]
            if changed:
                raise ValueError(f'Resume configuration changed: {changed}; original results preserved')
    report.update(pipeline_version=PIPELINE_VERSION, input_maps=args.input_maps, output_maps=args.output_maps,
                  augmentation=args.augmentation, excel_sha256=args.excel_sha256)
    json_save(folder/'preflight.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.preflight_only:
        print('Preflight passed; no model training performed'); return
    json_save(folder/'config.json', vars(args))
    if split_file.resolve() != (folder/'split_used.json').resolve():
        shutil.copyfile(split_file, folder/'split_used.json')
    results = []
    for fold in folds[:args.k_folds]:
        results.append(run_fold(args, df, fold, device))
        write_summary(results, folder/'cv_summary.json', args, split_file)
    print(f'Completed {len(results)} folds: {folder / "cv_summary.json"}')


if __name__ == '__main__':
    main()
