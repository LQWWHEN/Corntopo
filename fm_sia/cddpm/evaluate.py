"""Re-evaluate a saved checkpoint using its own normalization and fold labels."""
import argparse
import hashlib
from pathlib import Path
import pandas as pd
from predict import load_checkpoint
from train import make_dataset, make_loader, evaluate, load_dataframe
from dataloader import validate_table, validate_maps
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--data_root')
    p.add_argument('--excel_path')
    p.add_argument('--partition',choices=['val','test'],default='test')
    p.add_argument('--output_dir',required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--batch_size',type=int,default=None)
    p.add_argument('--num_workers',type=int,default=0)
    p.add_argument('--save_predictions',action='store_true')
    p.add_argument('--visual_error_max',type=float,default=None,
                   help='Override fixed error PNG color maximum in original map units')
    a=p.parse_args()
    model,args,stats,device=load_checkpoint(a.checkpoint,a.device)
    if a.data_root: args.data_root=a.data_root
    if a.excel_path: args.excel_path=a.excel_path
    if a.batch_size is not None: args.batch_size=a.batch_size
    if args.batch_size<1 or a.num_workers<0: raise ValueError('Invalid loader counts')
    args.num_workers=a.num_workers; args.save_predictions=a.save_predictions
    if a.visual_error_max is not None: args.visual_error_max=a.visual_error_max
    if hashlib.sha256(Path(args.excel_path).read_bytes()).hexdigest()!=args.excel_sha256:
        raise ValueError('Excel differs from the checkpoint dataset; use predict.py for new cases')
    # Read labels from the trusted checkpoint. Never regenerate test splits here.
    ckpt=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    labels=ckpt['fold'][a.partition]; del ckpt
    df=load_dataframe(args.excel_path)
    validate_table(df,args)
    validate_maps(df[df.label_index.isin(labels)],args)
    dataset=make_dataset(args,df,labels,stats)
    metrics=evaluate(model,make_loader(dataset,args,device),device,args,stats,a.output_dir,partition=a.partition)
    print(metrics)


if __name__=='__main__': main()
