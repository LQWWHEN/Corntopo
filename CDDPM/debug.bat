@echo off
setlocal enabledelayedexpansion
::python train.py --data_root ../data/img_batch2  --excel_path ../data/total_2.xlsx  --image_size 128 --batch_size 8   --epochs 1000   --lr 1e-4    --save_dir results/fm/rk3 --time_steps 1000  --model fm --ode_solver rk4

::python train.py --data_root ../data/img_batch2  --excel_path ../data/total_2.xlsx  --image_size 128 --batch_size 16    --epochs 250   --lr 5e-4    --save_dir results/uno --model uno 

python train.py --data_root ../data/dataset_3  --excel_path ../data/dataset_3/paired_dataset.xlsx  --image_size 128 --batch_size 16   --epochs 200   --lr 5e-4    --save_dir results/debug --model fno 