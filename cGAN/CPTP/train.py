import time
import random
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from models import create_model
from util.visualizer import Visualizer
from util.util import init_ddp, cleanup_ddp, save_results,save_image,tensor2heatmap
from dataloader import CornealTopoDataset
from metrics import calc_metrics  
from options.train_options import TrainOptions
import json
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

from transforms import DataTransform

# -----------------------------

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# -----------------------------
# EarlyStopping
# -----------------------------
class EarlyStopping:
    def __init__(self, patience=20):
        self.patience = patience
        self.counter = 0
        self.best_loss = np.inf

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

# ------------------------
def train_one_epoch(model, loader):
    model.train()
    loss_sum = 0
    for batch in loader:
        print(batch["input"].shape)

        model.set_input(batch)
        model.optimize_parameters()
        losses = model.get_current_losses()
        loss_sum += losses["G_GAN"] + losses["G_L1"]
    return loss_sum / len(loader)
"""
@torch.no_grad()
def evaluate_old(model, loader):
   
    model.eval()
    metric_dict = {
        "L1": [],
        "PSNR": [],
        "SSIM": [],
        "MAE": [],
        "RMSE": []
    }

    for batch in loader:
        model.set_input(batch)
        model.forward()

        fake = model.fake_B.detach()
        real = model.real_B.detach()

    

        # L1
        l1 = model.criterionL1(fake, real).item()
        metric_dict["L1"].append(l1)

        # 其他指标
        metrics = calc_metrics(fake, real)
        metric_dict["PSNR"].extend(metrics["PSNR_all"])
        metric_dict["SSIM"].extend(metrics["SSIM_all"])
        metric_dict["MAE"].extend(metrics["MAE_all"])
        metric_dict["RMSE"].extend(metrics["RMSE_all"])

        results={k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in metric_dict.items()}
    return results

"""
@torch.no_grad()
def evaluate(model, loader):
    """Evaluate model on a dataset (image + regression metrics)."""

    model.eval()

    # =========================
    # image metrics
    # =========================
    metric_dict = {
        "L1": [],
        "PSNR": [],
        "SSIM": [],
        "MAE": [],
        "RMSE": []
    }

    # =========================
    # regression storage
    # =========================
    reg_mae_all = []
    reg_mse_all = []

    for batch in loader:

        model.set_input(batch)
        model.forward()

        fake = model.fake_B.detach()
        real = model.real_B.detach()

        # -------------------------
        # image metrics
        # -------------------------
        l1 = model.criterionL1(fake, real).item()
        metric_dict["L1"].append(l1)

        metrics = calc_metrics(fake, real)
        metric_dict["PSNR"].extend(metrics["PSNR_all"])
        metric_dict["SSIM"].extend(metrics["SSIM_all"])
        metric_dict["MAE"].extend(metrics["MAE_all"])
        metric_dict["RMSE"].extend(metrics["RMSE_all"])

        # -------------------------
        # regression metrics
        # -------------------------
        if hasattr(model, "pred_out") and model.pred_out is not None:

            pred = model.pred_out.detach().float().cpu()
            gt = model.targets.detach().float().cpu()

            abs_err = torch.abs(pred - gt)        # (B, K)
            sq_err = (pred - gt) ** 2             # (B, K)

            reg_mae_all.append(abs_err.numpy())
            reg_mse_all.append(sq_err.numpy())

    # =========================
    # image results
    # =========================
    results = {
        k: {
            "mean": float(np.mean(v)),
            "std": float(np.std(v))
        }
        for k, v in metric_dict.items()
    }

    # =========================
    # regression results (per-target)
    # =========================
    if len(reg_mae_all) > 0:

        reg_mae_all = np.concatenate(reg_mae_all, axis=0)    # (N, K)
        reg_mse_all = np.concatenate(reg_mse_all, axis=0)  # (N, K)

        K = reg_mae_all.shape[1]

        results["REG"] = {}

        for i in range(K):

            results["REG"][f"Target_{i}"] = {
                "MAE": {
                    "mean": float(np.mean(reg_mae_all[:, i])),
                    "std": float(np.std(reg_mae_all[:, i]))
                },
                "MSE": {
                    "mean": float(np.mean(reg_mse_all[:, i])),
                    "std": float(np.std(reg_mse_all[:, i]))
                }
            }

        # optional: overall summary
        results["REG_OVERALL"] = {
            "MAE_mean": float(np.mean(reg_mae_all)),
            "MSE_mean": float(np.mean(reg_mse_all))
        }

    return results

@torch.no_grad()
def visualize_eval(model, loader, save_dir, max_samples=None):
    
    save_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    sample_idx = 0

    for batch in loader:

        model.set_input(batch)
        model.forward()
        model.compute_visuals()

        visuals = model.get_current_visuals()

        bs = next(iter(visuals.values())).shape[0]

        for b in range(bs):

            if max_samples is not None and sample_idx >= max_samples:
                return

           

            for label, image in visuals.items():

                image_numpy = tensor2heatmap(
                    image[b:b+1],
                    cmap="turbo"
                )

                save_path = save_dir / f"{label}.png"

                save_image(
                    image_numpy,
                    save_path
                )

            sample_idx += 1

def print_result(results):
    for k, v in results.items():

        # =========================
        # normal metrics (L1 / PSNR / SSIM ...)
        # =========================
        if isinstance(v, dict) and "mean" in v:

            print(f"{k}: {v['mean']:.4f} ± {v['std']:.4f}")

        # =========================
        # regression metrics
        # =========================
        elif k == "REG":

            print("\n[Regression Results]")

            for t_name, t_metrics in v.items():

                print(f"  {t_name}:")

                for metric_name, metric_value in t_metrics.items():

                    mean = metric_value["mean"]
                    std = metric_value["std"]

                    print(f"    {metric_name}: {mean:.4f} ± {std:.4f}")

        # =========================
        # optional overall regression
        # =========================
        elif k == "REG_OVERALL":

            print("\n[Regression Overall]")
            for m, val in v.items():
                print(f"  {m}: {val:.4f}")

# -----------------------------
# 主训练流程
# -----------------------------
if __name__ == "__main__":
    seed_everything()
    opt = TrainOptions().parse()
    opt.device = init_ddp()
    transform = DataTransform(
        hflip_prob=0.5,
        vflip_prob=0.5,
        rotation=True,
    )

    dataset = CornealTopoDataset(
        opt=opt,
        excel_path=opt.excel_path,
        data_root=opt.data_root,
        data_type="raw",  # 默认 raw
        image_size=128
    )

    n = len(dataset)

    all_indices = np.arange(n)

  
    train_idx, temp_idx = train_test_split(
        all_indices,
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

    train_set = CornealTopoDataset(opt=opt,excel_path=opt.excel_path,
        data_root=opt.data_root,
        data_type="raw", 
        image_size=128,transforms=transform,indices=train_idx)
    
    val_set   = CornealTopoDataset(opt=opt,excel_path=opt.excel_path,
        data_root=opt.data_root,
        data_type="raw",  
        image_size=128,transforms=None,indices=val_idx)
    
    test_set  = CornealTopoDataset(opt=opt,excel_path=opt.excel_path,
        data_root=opt.data_root,
        data_type="raw",  
        image_size=128,transforms=None,indices=test_idx)
    

    split_dir = Path(opt.checkpoints_dir) / opt.name
    split_dir.mkdir(parents=True, exist_ok=True)

    split_file = split_dir / "data_split.json"

    with open(split_file, "w") as f:
        json.dump(
            {
                "train": train_idx.tolist(),
                "val": val_idx.tolist(),
                "test": test_idx.tolist(),
            },
            f,
            indent=4,
        )

    print(f"Saved split to {split_file}")

    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=4,pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=opt.batch_size, shuffle=False, num_workers=4,pin_memory=True)
    test_loader = DataLoader(
        test_set,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    # -------------------------
    # 模型
    # -------------------------
    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    stopper = EarlyStopping(patience=opt.patience)
    total_iters = 0
    best_epoch = 0
    has_latest = False
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        visualizer.reset()

        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        # --- 训练 ---
        for i, data in enumerate(train_loader):
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            model.set_input(data)
            model.optimize_parameters()

            # --- 可视化 ---
            if total_iters % opt.display_freq == 0:
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(
                    model.get_current_visuals(), epoch, total_iters, save_result
                )

            # --- 打印 ---
            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                visualizer.plot_current_losses(total_iters, losses)

            # --- 保存 latest ---
            if total_iters % opt.save_latest_freq == 0:
                save_suffix = f"iter_{total_iters}" if opt.save_by_iter else "latest"
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        # --- 验证集 ---
        val_results = evaluate(model, val_loader)
        val_vis_dir = split_dir / "inter_visual" / f"val_{epoch}"

        visualize_eval(model, val_loader,  val_vis_dir, max_samples=4)

        val_loss = val_results["L1"]["mean"]
        print(
            f"[Val] "
            f"L1={val_results['L1']['mean']:.4f} "
            f"PSNR={val_results['PSNR']['mean']:.3f} "
            f"SSIM={val_results['SSIM']['mean']:.4f}"
            
        )
        print(f"Reg:{val_results['REG']}")
        if stopper.step(val_loss):
            print("[Val improved] Saving best model")
            best_epoch =epoch
            model.save_networks("best")

    

        model.update_learning_rate()

        if epoch % opt.save_epoch_freq == 0:
            has_latest = True
            model.save_networks("latest")
            model.save_networks(epoch)

        print(f"End of epoch {epoch} | Time: {time.time() - epoch_start_time:.0f}s")
    print("\n====================")
    print("Testing Best Model")
    print(f"Epoch:{best_epoch}")
    print("====================")
    model.load_networks("best")
    result_vis_dir = split_dir / "visuals"

    val_results = evaluate(
        model,
        val_loader,
    )

    test_results = evaluate(
        model,
        test_loader,
    )
    print("\nFinal Validation")

    print_result(val_results)
    val_vis_dir = split_dir / "visuals" / "val_best"

    visualize_eval(model, val_loader,val_vis_dir, max_samples=None)

    print("\nFinal Test")

    print_result(test_results)

    result_file = split_dir / "final_metrics.json"
    test_vis_dir = split_dir / "visuals" / "test"
    visualize_eval(model, test_loader, test_vis_dir, max_samples=None)
    with open(result_file, "w") as f:

        json.dump(
            {
                "best_epoch": best_epoch,
                "val": val_results,
                "test": test_results,
            },
            f,
            indent=4
        )

    if has_latest:
        model.load_networks("latest")
        result_vis_dir = split_dir / "visuals"

        val_results = evaluate(
            model,
            val_loader,
        )

        test_results = evaluate(
            model,
            test_loader,
        )
        print("\nFinal Validation")

        print_result(val_results)


        val_vis_dir = split_dir / "visuals" / "val_latest"

        visualize_eval(model, val_loader,val_vis_dir, max_samples=None)

        print("\nFinal Test")

        print_result(test_results)

        result_file = split_dir / "latest_metrics.json"
        test_vis_dir = split_dir / "visuals" / "test_latest"
        visualize_eval(model, test_loader, test_vis_dir, max_samples=None)
        with open(result_file, "w") as f:

            json.dump(
                {

                    "val": val_results,
                    "test": test_results,
                },
                f,
                indent=4
            )
    cleanup_ddp() 