@echo off
setlocal enabledelayedexpansion

set seeds=42
set i=1

for %%s in (%seeds%) do (
    echo Running experiment !i! with seed %%s...

   

    
    python test.py ^
    --model CPTP ^
    --name grad_loss_32 ^
    --data_root ../../data/img_batch2 ^
    --excel_path ../../data/total_2.xlsx ^
    --input_nc 1 ^
    --output_nc 1 ^
    --ngf 32 ^
    --ndf 32 ^
    --batch_size 8 ^
    --netD n_layers ^
    --n_layers_D 2 ^
    --norm batch ^
    --netG unet_128 ^
    --use_wandb ^
    --outer_activation tanh ^
    --epoch best

    set /a i+=1
)

pause