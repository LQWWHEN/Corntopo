"""Verify validation-selected checkpoints, then test all folds without retraining.

Weights remain temporary until the experiment audit calls checkpoint_lifecycle.py.
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from predict import load_checkpoint
from train import (evaluate, json_save, load_dataframe, make_dataset, make_loader,
                   write_summary, PIPELINE_VERSION)
from dataloader import validate_table, validate_maps


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def compare_metrics(expected, actual):
    if expected.keys() != actual.keys():
        raise ValueError('Saved validation metric schema differs')
    for key, value in expected.items():
        other = actual[key]
        if value is None or other is None:
            equal = value is other
        else:
            equal = math.isclose(value, other, rel_tol=1e-6, abs_tol=1e-7)
        if not equal:
            raise ValueError(f'Saved validation differs: {key}')


def test_run(run_dir, output_dir, device='cuda'):
    run, out = Path(run_dir).resolve(), Path(output_dir).resolve()
    if out == run or out.exists():
        raise FileExistsError('Choose a new output directory; training records are preserved')
    summary = read(run/'cv_summary.json'); config = read(run/'config.json')
    if summary['reporting_partition'] != 'val' or not config['save_checkpoints']:
        raise ValueError('Expected a validation-only run with temporary checkpoints')
    if summary['completed_folds'] != config['k_folds']:
        raise ValueError('Training folds are incomplete')
    if hashlib.sha256(Path(config['excel_path']).read_bytes()).hexdigest() != config['excel_sha256']:
        raise ValueError('Dataset differs from training')
    frozen_folds = read(config['split_file'])['folds'][:config['k_folds']]
    if len(frozen_folds) != config['k_folds']:
        raise ValueError('Frozen folds are incomplete')
    df = load_dataframe(config['excel_path'], config['conditions'])
    validate_table(df, argparse.Namespace(**config))
    validate_maps(df, argparse.Namespace(**config))
    out.mkdir(parents=True)
    started = time.perf_counter(); checkpoints = []; results = []
    # All selected/final validation outputs must match before any test loader exists.
    for fold in frozen_folds:
        number = fold['fold']; folder = run/f'fold_{number:02d}'
        history = read(folder/'history.json')
        best_epoch = min((h for h in history if 'validation' in h),
                         key=lambda h:h['validation'][config['best_metric']])['epoch']
        for name in ('best', 'final'):
            path = folder/f'{name}.pth'
            state = torch.load(path, map_location='cpu', weights_only=False)
            epoch = best_epoch if name == 'best' else config['epochs']
            if (state['pipeline_version'] != PIPELINE_VERSION or state['config'] != config
                    or state['fold'] != fold or state['epoch']+1 != epoch
                    or state['stats'] != read(folder/'normalization.json')):
                raise ValueError('Checkpoint configuration, fold, epoch or normalization differs')
            if read(folder/f'{name}_selection.json')['epoch'] != epoch:
                raise ValueError('Checkpoint selection metadata differs')
            del state
            model, args, stats, target = load_checkpoint(path, device, config['cpu_threads'])
            args.save_predictions = False
            dataset = make_dataset(args, df, fold['val'], stats)
            actual = evaluate(model, make_loader(dataset,args,target), target,args,stats,
                              partition='val')
            compare_metrics(read(folder/'validation_results'/f'{name}.json'), actual)
            checkpoints.append(dict(fold=number, name=name, epoch=epoch,
                                    validation_metrics_compared=len(actual)))
            del model
    json_save(out/'saved_validation_verification.json', dict(status='PASS', checkpoints=checkpoints))
    validation_seconds = time.perf_counter()-started
    for fold in frozen_folds:
        number = fold['fold']; folder = run/f'fold_{number:02d}'; dest = out/folder.name
        fold_started = time.perf_counter(); measured = {}
        for name in ('best','final'):
            model, args, stats, target = load_checkpoint(folder/f'{name}.pth',device,config['cpu_threads'])
            args.save_predictions = True
            dataset = make_dataset(args,df,fold['test'],stats)
            measured[name] = evaluate(model,make_loader(dataset,args,target),target,args,stats,
                                      dest/'test_results'/name,partition='test')
            json_save(dest/'test_results'/f'{name}.json',measured[name])
            json_save(dest/f'{name}_selection.json',read(folder/f'{name}_selection.json'))
            del model
        timing = read(folder/'timing.json')
        timing['test_only_seconds'] = time.perf_counter()-fold_started
        json_save(dest/'timing.json',timing)
        json_save(dest/'normalization.json',stats)
        results.append(dict(fold=number,sizes={p:len(fold[p]) for p in ('train','val','test')},
                            timing=timing,**measured))
        args.validation_only = False; args.save_dir = str(out)
        write_summary(results,out/'cv_summary.json',args,config['split_file'])
    json_save(out/'phase_timing.json',dict(validation_verification_seconds=validation_seconds,
        total_seconds=time.perf_counter()-started, training_seconds=0,
        note='Training costs in fold timing refer to the original run; no retraining occurred'))
    print(f'TEST_COMPLETE: {len(results)} folds; audit results before checkpoint cleanup')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run_dir',required=True);p.add_argument('--output_dir',required=True)
    p.add_argument('--device',default='cuda');a=p.parse_args()
    test_run(a.run_dir,a.output_dir,a.device)


# This is a command entry point, not a pytest test function.
test_run.__test__ = False
if __name__ == '__main__': main()
