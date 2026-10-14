"""Independent test entry point: checkpoint owns preprocessing and model settings."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from options.test_options import TestOptions
from patient_cv import load_cv_splits
from dataloader import validate_table, validate_maps
from train import (PIPELINE_VERSION, build_model, evaluate, json_save, load_dataframe,
                   make_dataset, make_loader, resolve_device, write_summary)


def main(argv=None):
    cli = TestOptions().parse(argv)
    root = Path(cli.save_dir)
    args = argparse.Namespace(**json.loads((root/'config.json').read_text(encoding='utf-8')))
    for name in ('excel_path','data_root','split_file'):
        if getattr(cli, name) is not None:
            setattr(args, name, getattr(cli, name))
    args.device, args.num_workers, args.cpu_threads = cli.device, cli.num_workers, cli.cpu_threads
    if cli.batch_size is not None:
        args.batch_size = cli.batch_size
    if cli.k_folds is not None:
        if not 1 <= cli.k_folds <= args.k_folds:
            raise ValueError('Test k_folds must be within the originally requested folds')
        args.k_folds = cli.k_folds
    if args.batch_size < 1 or args.num_workers < 0 or args.cpu_threads < 1:
        raise ValueError('Invalid runtime counts')
    device = resolve_device(args.device, args.cpu_threads)
    for name, expected in [('excel_path',args.excel_sha256), ('split_file',args.split_sha256)]:
        if hashlib.sha256(Path(getattr(args,name)).read_bytes()).hexdigest() != expected:
            raise ValueError(f'{name} differs from training; refusing an incomparable evaluation')
    df = load_dataframe(args.excel_path)
    validate_table(df,args); validate_maps(df,args)
    folds = load_cv_splits(df,args.split_file,seed=args.seed)
    out_root = Path(cli.results_dir) if cli.results_dir else root/'reevaluation'
    names = ('best','final') if cli.checkpoint == 'both' else (cli.checkpoint,)
    results = []
    for fold in folds[:args.k_folds]:
        number = fold['fold']; results_fold = dict(fold=number, sizes={p:len(fold[p]) for p in ('train','val','test')})
        for name in names:
            path = root/f'fold_{number:02d}'/f'{name}.pth'
            state = torch.load(path, map_location='cpu', weights_only=False)
            if state['pipeline_version'] != PIPELINE_VERSION or state['fold'] != fold:
                raise ValueError(f'Checkpoint pipeline/fold mismatch: {path}')
            runtime = {'excel_path','data_root','split_file','save_dir','device','num_workers','cpu_threads',
                       'batch_size','k_folds','resume','preflight_only'}
            changed = [k for k,v in vars(args).items() if k not in runtime and state['config'].get(k) != v]
            if changed:
                raise ValueError(f'Checkpoint config mismatch: {changed}')
            # Never fit statistics on test data or overwrite saved training statistics.
            stats = state['stats']
            model = build_model(args,stats).to(device)
            model.load_state_dict(state['ema'] if args.use_ema else state['model'])
            dataset = make_dataset(args,df,fold['test'],stats)
            loader = make_loader(dataset,args,device)
            output = out_root/f'fold_{number:02d}'/name
            results_fold[name] = evaluate(model,loader,device,args,stats,output,partition='test')
            json_save(output/'checkpoint.json',dict(path=str(path),epoch=state['epoch']+1,
                                                   used_ema=args.use_ema,fold=number))
        results.append(results_fold)
    if len(names) == 2:
        write_summary(results,out_root/'cv_summary.json',args,args.split_file)
    else:
        json_save(out_root/f'{names[0]}_fold_results.json', results)
    print(f'Evaluated {len(results)} folds: {out_root}')


if __name__ == '__main__':
    main()
