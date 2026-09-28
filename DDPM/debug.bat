@echo off
setlocal enabledelayedexpansion
python main.py --data_root ../data/img_batch2  --excel_path ../data/total_2.xlsx  --image_size 128 --batch_size 8   --n_epochs 2    --lr 2e-4   --timesteps 1000   --sampling_steps 250  --checkpoints_dir results/ddpm --save_epoch_freq 2