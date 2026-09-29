

@echo off
setlocal enabledelayedexpansion


python cddpm/train.py --data_root data/dataset_3  --excel_path data/dataset_3/paired_dataset.xlsx  --image_size 128 --batch_size 16   --epochs 100   --lr 1e-4    --save_dir cddpm/results/FRONT/fm_fusion/CA --time_steps 1000  --model fm --k_folds 3 --fusion_method cross_attention --conditions "pre_GENERAL DATA_Cornea Front Astig" "pre_GENERAL DATA_K Max (Front)" "pre_GENERAL DATA_Pachy Min" --use_ema --input_maps "FRONT" --output_maps "FRONT"

python cddpm/train.py --data_root data/dataset_3  --excel_path data/dataset_3/paired_dataset.xlsx  --image_size 128 --batch_size 16   --epochs 100   --lr 1e-4    --save_dir cddpm/results/FRONT/fm_fusion/concat --time_steps 1000  --model fm --k_folds 3 --fusion_method concat --conditions "pre_GENERAL DATA_Cornea Front Astig" "pre_GENERAL DATA_K Max (Front)" "pre_GENERAL DATA_Pachy Min" --use_ema --input_maps "FRONT" --output_maps "FRONT"

python cddpm/train.py --data_root data/dataset_3  --excel_path data/dataset_3/paired_dataset.xlsx  --image_size 128 --batch_size 16   --epochs 100   --lr 1e-4    --save_dir cddpm/results/FRONT/fm_fusion/SA --time_steps 1000  --model fm --k_folds 3 --fusion_method self_attention --conditions "pre_GENERAL DATA_Cornea Front Astig" "pre_GENERAL DATA_K Max (Front)" "pre_GENERAL DATA_Pachy Min" --use_ema --input_maps "FRONT" --output_maps "FRONT"

python cddpm/train.py --data_root data/dataset_3  --excel_path data/dataset_3/paired_dataset.xlsx  --image_size 128 --batch_size 16   --epochs 100   --lr 1e-4    --save_dir cddpm/results/FRONT/fm_fusion/none --time_steps 1000  --model fm --k_folds 3 --fusion_method none --use_ema --input_maps "FRONT" --output_maps "FRONT"
::python train.py --data_root ../data/img_batch2  --excel_path ../data/total_2.xlsx  --image_size 128 --batch_size 16    --epochs 250   --lr 5e-4    --save_dir results/uno --model uno 

::python train.py --data_root ../data/dataset_3  --excel_path ../data/dataset_3/paired_dataset.xlsx  --image_size 128 --batch_size 16   --epochs 1   --lr 5e-4    --save_dir results/debug --model ddpm --k_folds 2 --fusion_method none