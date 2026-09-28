@echo off
setlocal enabledelayedexpansion
python cddpm/train.py --data_root data/dataset_3  --excel_path data/dataset_3/paired_dataset.xlsx  --image_size 128 --batch_size 8   --epochs 2   --lr 1e-4    --save_dir results/debug --time_steps 1000  --model fm --ode_solver rk4 --k_folds 2 --fusion_method cross_attention --conditions 

::python train.py --data_root ../data/img_batch2  --excel_path ../data/total_2.xlsx  --image_size 128 --batch_size 16    --epochs 250   --lr 5e-4    --save_dir results/uno --model uno 

::python train.py --data_root ../data/dataset_3  --excel_path ../data/dataset_3/paired_dataset.xlsx  --image_size 128 --batch_size 16   --epochs 1   --lr 5e-4    --save_dir results/debug --model ddpm --k_folds 2 --fusion_method none