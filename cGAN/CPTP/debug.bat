
@echo off
setlocal enabledelayedexpansion
  python train.py ^
    --model mCPTP ^
    --name debug ^
    --data_root ../../data/img_batch2 ^
    --excel_path ../../data/total_2.xlsx ^
    --checkpoints_dir ./checkpoints/batch2 ^
    --input_nc 1 ^
    --output_nc 1 ^
    --ngf 16 ^
    --ndf 16 ^
    --batch_size 8 ^
    --norm batch ^
    --netG unet_128 ^
    --lambda_L1 25 ^
    --n_epochs 1 ^
    --n_epochs_decay 1 ^
    --patience 100 ^
    --lr_policy linear ^
    --lr 0.0002 ^
    --outer_activation sigmoid ^
    --lambda_grad 10 ^
    --conditions pre_axis pre_astig pre_chamberangle age pre_kmax ^
    --targets SIA