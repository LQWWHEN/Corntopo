@echo off
setlocal enabledelayedexpansion
 


python train.py ^
    --model mCPTP ^
    --name no_cond ^
    --data_root ../../data/img_batch2 ^
    --excel_path ../../data/total_2.xlsx ^
    --checkpoints_dir ./checkpoints/batch2/mCPTP ^
    --input_nc 1 ^
    --output_nc 1 ^
    --ngf 64 ^
    --ndf 64 ^
    --batch_size 8 ^
    --norm batch ^
    --netG unet_128 ^
    --use_wandb ^
    --lambda_L1 25 ^
    --n_epochs 40 ^
    --n_epochs_decay 160 ^
    --patience 100 ^
    --lr_policy linear ^
    --lr 0.0002 ^
    --outer_activation sigmoid ^
    --lambda_grad 10 ^
    --targets SIA 



    




