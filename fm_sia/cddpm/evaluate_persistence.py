"""Evaluate the unchanged preoperative image using a frozen run's folds and scales.

This image-only diagnostic has no fitted parameters, SIA head or checkpoint.
Keep outputs private, as they include individual images and dataset provenance.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

from train import load_dataframe, make_dataset, json_save, save_batch_visuals
from metrics import MetricAccumulator
from patient_cv import _labels_and_groups, _validate_splits
from visualization import visualization_spec


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def evaluate_persistence(reference_run, output_dir, partitions=('val','test')):
    reference, out = Path(reference_run).resolve(), Path(output_dir).resolve()
    if out.exists():
        raise FileExistsError('Choose a new control output directory')
    config = read(reference/'config.json'); args = argparse.Namespace(**config)
    if config['input_maps'] != ['FRONT'] or config['output_maps'] != ['FRONT']:
        raise ValueError('The persistence control requires matching FRONT map semantics')
    if hashlib.sha256(Path(config['excel_path']).read_bytes()).hexdigest() != config['excel_sha256']:
        raise ValueError('Dataset differs from the frozen reference')
    frame = load_dataframe(config['excel_path'],config['conditions'])
    splits = read(config['split_file'])
    labels, groups, fingerprint = _labels_and_groups(frame)
    _validate_splits(splits,labels,groups,fingerprint,config['seed'],10)
    folds = splits['folds'][:config['k_folds']]
    if len(folds) != config['k_folds'] or set(partitions)-{'val','test'}:
        raise ValueError('Invalid frozen folds or partitions')
    out.mkdir(parents=True)
    started = time.perf_counter(); results = {}
    for partition in partitions:
        records = []
        for fold in folds:
            name = f"fold_{fold['fold']:02d}"
            stats = read(reference/name/'normalization.json')
            dataset = make_dataset(args,frame,fold[partition],stats,training=False)
            accumulator = MetricAccumulator(stats['image']['scale'])
            destination = out/partition/name; destination.mkdir(parents=True)
            json_save(destination/'visualization.json',dict(
                **visualization_spec(stats['image'],getattr(args,'visual_error_max',1.0)),
                partition=partition,sample_selection='first 5 samples' if partition=='val' else 'all samples'))
            independent_absolute = independent_squared = 0.; pixels = 0
            for i in range(len(dataset)):
                sample = dataset[i]; batch = default_collate([sample])
                prediction = batch['input']  # No outcome or fitted state enters this prediction.
                mask = batch['mask_pre']*batch['mask_post']
                accumulator.update(prediction,batch['target'],mask)
                valid = mask.numpy().astype(bool)
                error = prediction.numpy().astype(np.float64)[valid]-batch['target'].numpy().astype(np.float64)[valid]
                independent_absolute += np.abs(error).sum()
                independent_squared += np.square(error).sum(); pixels += error.size
                if partition == 'test' or i < 5:
                    stem = f'{i:06d}'
                    save_batch_visuals(destination,stem,batch,prediction,0,args,stats)
                    raw = prediction[0,0].numpy()*stats['image']['scale']+stats['image']['minimum']
                    raw = np.where(batch['mask_pre'][0,0].numpy().astype(bool),raw,np.nan)
                    np.save(destination/f'{stem}_pred_raw.npy',raw)
            metrics = accumulator.compute()
            np.testing.assert_allclose(metrics['MAE'],independent_absolute/pixels,rtol=0,atol=1e-12)
            np.testing.assert_allclose(metrics['RMSE'],np.sqrt(independent_squared/pixels),rtol=0,atol=1e-12)
            np.testing.assert_allclose(metrics['PSNR'],-10*np.log10(max(independent_squared/pixels,1e-12)),rtol=0,atol=1e-12)
            json_save(destination/'metrics.json',metrics)
            json_save(destination/'normalization.json',stats)
            json_save(destination/'sample_labels.json',fold[partition])
            records.append(dict(fold=fold['fold'],samples=len(dataset),metrics=metrics))
        aggregate = {key:dict(mean=float(np.mean([r['metrics'][key] for r in records])),
                              std=float(np.std([r['metrics'][key] for r in records])),n=len(records))
                     for key in records[0]['metrics'] if all(r['metrics'][key] is not None for r in records)}
        results[partition] = dict(folds=records,aggregate=aggregate)
    result = dict(audit='PASS',predictor='preoperative_FRONT_unchanged',SIA_output=None,
                  reference_run=str(reference),reference_excel_sha256=config['excel_sha256'],
                  reference_config_sha256=hashlib.sha256((reference/'config.json').read_bytes()).hexdigest(),
                  evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  reference_split_file=config['split_file'],image_size=config['image_size'],
                  training_seconds=0.,evaluation_seconds=time.perf_counter()-started,
                  evaluation_domain='common pre/post valid region; unchanged train-fold normalization',
                  results=results)
    json_save(out/'results.json',result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference_run',required=True)
    parser.add_argument('--output_dir',required=True)
    parser.add_argument('--partition',choices=['val','test','both'],default='both')
    args = parser.parse_args(); torch.set_num_threads(1)
    partitions = ('val','test') if args.partition == 'both' else (args.partition,)
    result = evaluate_persistence(args.reference_run,args.output_dir,partitions)
    print(json.dumps({p:r['aggregate'] for p,r in result['results'].items()},indent=2))


if __name__ == '__main__':
    main()
