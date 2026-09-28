import os
import time
import argparse
from pathlib import Path
import json
import random

from sklearn.model_selection import train_test_split

from metrics import calc_metrics
import numpy as np
import torch
from torch.utils.data import DataLoader
from util.util import init_ddp, cleanup_ddp, save_results,save_image,tensor2heatmap
from denoising_diffusion_pytorch import (
    Unet,
    GaussianDiffusion
)
from options.train_options import TrainOptions
from dataloader import CornealTopoDataset


def seed_everything(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
class EarlyStopping:

    def __init__(self, patience=20):
        self.best_loss = 1e9
        self.counter = 0
        self.patience = patience

    def step(self, val_loss):

        if val_loss < self.best_loss:

            self.best_loss = val_loss
            self.counter = 0
            return True

        self.counter += 1
        return False

    @property
    def should_stop(self):
        return self.counter >= self.patience

def save_checkpoint(diffusion,
                    optimizer,
                    epoch,
                    path):

    torch.save(
        {
            "epoch": epoch,
            "model": diffusion.state_dict(),
            "optimizer": optimizer.state_dict()
        },
        path
    )


@torch.no_grad()
def sample_images(
        diffusion,
        epoch,
        save_dir,
        n=4):

    diffusion.eval()

    imgs = diffusion.sample(batch_size=n)

    imgs = (imgs + 1) / 2
    imgs = imgs.clamp(0, 1)

    import torchvision

    grid = torchvision.utils.make_grid(
        imgs,
        nrow=2
    )

    os.makedirs(save_dir, exist_ok=True)

    torchvision.utils.save_image(
        grid,
        os.path.join(
            save_dir,
            f"epoch_{epoch}.png"
        )
    )

    diffusion.train()

@torch.no_grad()
def evaluate(diffusion,
             loader,
             device):

    diffusion.eval()

    psnr_all = []
    ssim_all = []

    loss_all = []

    for batch in loader:
        gt = batch["target"].to(device)
        loss = diffusion(gt)
        loss_all.append(loss.item())
        pred = diffusion.sample(batch_size=gt.size(0))
        metrics = calc_metrics(pred, gt)
        psnr_all.extend(metrics["PSNR_all"])
        ssim_all.extend(metrics["SSIM_all"])

    diffusion.train()

    return {
        "loss": np.mean(loss_all),
        "PSNR": np.mean(psnr_all),
        "SSIM": np.mean(ssim_all)
    }

@torch.no_grad()
def visualize(
    diffusion,
    loader,
    save_dir,
    device
):

    diffusion.eval()

    save_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for i, batch in enumerate(loader):

        gt = batch["target"].to(device)

        inp = batch["input"].to(device)

        pred = diffusion.sample(batch_size=1)

        err = torch.abs(pred - gt)

        imgs = {
            "input": inp,
            "target": gt,
            "pred": pred,
            "error": err
        }

        for name, img in imgs.items():

            heat = tensor2heatmap(img)

            save_image(
                heat,
                save_dir /
                f"{i:03d}_{name}.png"
            )

def main():

    opt = TrainOptions().parse()
    opt.device = init_ddp()
    seed_everything(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    ############################################
    # Dataset
    ############################################

    dataset = CornealTopoDataset(
        opt=opt,
        excel_path=opt.excel_path,
        data_root=opt.data_root,
        input_maps=["TOTAL CORNEAL REFRACTIVE POWER"],
        output_maps=["TOTAL CORNEAL REFRACTIVE POWER"],
        image_size=128,
        normalize=True
    )

    split_file = Path(opt.data_root) / "data_split.json"

    if split_file.exists():

        print("Load existing split.")

        with open(split_file) as f:
            split = json.load(f)

        train_idx = split["train"]
        val_idx = split["val"]
        test_idx = split["test"]

    else:

        idx = np.arange(len(dataset))

        train_idx, temp_idx = train_test_split(
            idx,
            test_size=0.2,
            random_state=42,
            shuffle=True
        )

        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=0.5,
            random_state=42,
            shuffle=True
        )

        with open(split_file, "w") as f:

            json.dump(
                {
                    "train": train_idx.tolist(),
                    "val": val_idx.tolist(),
                    "test": test_idx.tolist()
                },
                f,
                indent=4
            )

    train_set = CornealTopoDataset(
        opt=opt,
        excel_path=opt.excel_path,
        data_root=opt.data_root,
        image_size=128,
        indices=train_idx
    )

    val_set = CornealTopoDataset(
        opt=opt,
        excel_path=opt.excel_path,
        data_root=opt.data_root,
        image_size=128,
        indices=val_idx
    )

    test_set = CornealTopoDataset(
        opt=opt,
        excel_path=opt.excel_path,
        data_root=opt.data_root,
        image_size=128,
        indices=test_idx
    )

    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=4,pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=opt.batch_size, shuffle=False, num_workers=4,pin_memory=True)
    test_loader = DataLoader(
        test_set,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    ############################################
    # UNet
    ############################################

    model = Unet(
        dim=64,
        channels=1,
        dim_mults=(1, 2, 4, 8),
        flash_attn=False

    )

    ############################################
    # DDPM
    ############################################

    diffusion = GaussianDiffusion(
        model,
        image_size=128,
        timesteps=opt.timesteps,
        sampling_timesteps=opt.sampling_steps

    ).to(device)

    optimizer = torch.optim.AdamW(

        diffusion.parameters(),

        lr=opt.lr

    )

    ############################################
    # folders
    ############################################

    save_dir = Path(opt.checkpoints_dir)

    ckpt_dir = save_dir / "checkpoints"

    sample_dir = save_dir / "samples"

    ckpt_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    sample_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    ############################################
    # Training
    ############################################
    best_loss = 1e9
    for epoch in range(opt.n_epochs):

        epoch_loss = 0

        tic = time.time()

        for batch in train_loader:
            img = batch["target"].to(device)
            loss = diffusion(img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)

        print(
            f"Epoch {epoch+1:04d} | "
            f"Loss {epoch_loss:.6f} | "
            f"{time.time()-tic:.1f}s"
        )

        ####################################
        # sample
        ####################################


        if (epoch + 1) % opt.save_epoch_freq == 0:
            val_result = evaluate(diffusion,val_loader,device)
            print(f"[VAL] " f"Loss={val_result['loss']:.4f} " 
                f"PSNR={val_result['PSNR']:.3f} "
                f"SSIM={val_result['SSIM']:.4f}"  )
            visualize(diffusion,val_loader,sample_dir/f"val_{epoch+1}", device)
            save_checkpoint(
                diffusion,
                optimizer,
                epoch + 1,
                ckpt_dir /
                f"epoch_{epoch+1}.pth"

            )
    ckpt = torch.load(ckpt_dir / "best.pth",map_location=device)

    diffusion.load_state_dict(
        ckpt["model"]
    )

    test_result = evaluate(
        diffusion,
        test_loader,
        device
    )

    print("=" * 50)
    print("Test")
    print(test_result)
    visualize(diffusion,  test_loader, sample_dir / "test",device )


if __name__ == "__main__":

    main()