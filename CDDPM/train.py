import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from torch.utils.tensorboard import SummaryWriter
import json
from dataloader import CornealTopoDataset
from transforms import DataTransform
from model.encoder import ConditionEncoder
from model.unet import ConditionalUNet
from model.cddpm import CDDPM
from model.ema import EMA
from model.bbdm import BBDM
from model.fm import FlowMatching
from model.no import FourierNeuralOperator,UnetNeuralOperator
from pathlib import Path
import pandas as pd
from patient_cv import load_or_create_cv_splits
from metrics import calculate_metrics
from util.util import save_heatmap
import time
import numpy as np
# ==========================
# 参数
# ==========================

def get_args():

    parser = argparse.ArgumentParser()


    # data
    parser.add_argument("--excel_path",type=str,required=True)
    parser.add_argument("--data_root",type=str,required=True)
    parser.add_argument("--image_size",type=int,default=128)

    # training
    parser.add_argument("--model",type=str,default="cddpm",choices=["cddpm", "bbdm", "fm", "fno", "uno"])
    parser.add_argument("--epochs", type=int,default=200)
    parser.add_argument("--batch_size",type=int,default=8)
    parser.add_argument("--lr", type=float,default=2e-4)
    parser.add_argument("--num_workers",type=int,default=4)
    parser.add_argument("--time_steps", type=int,default=50000)
    parser.add_argument("--save_dir",type=str,default="./checkpoints")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sampler", type=str, default="ddpm")
    parser.add_argument("--ode_solver", type=str, default="euler")

    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--k_folds", type=int, default=10,
                        help="仅运行前 K 折，取值为 1 到 10")
    parser.add_argument("--split_file", type=str, default=None,
                        help="10 折 JSON 路径，默认为 data_root/patient_10fold_splits.json")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--use_ema",action="store_true")
    parser.add_argument("--ema_decay",type=float,default=0.995)
    parser.add_argument("--objective",type=str , default="grad")
    parser.add_argument("--conditions", nargs="+", default=[])
    parser.add_argument("--targets", nargs="+", default=[])
    parser.add_argument(
        "--fusion_method", default="none",
        choices=["none", "concat", "add", "self_attention", "cross_attention", "film"],
        help="临床参数与图像特征的融合方式",
    )
    parser.add_argument("--clinical_embed_dim", type=int, default=128)


    parser.add_argument(
        "--fno_modes",
        type=int,
        nargs=2,
        default=[32,32],
        help="Fourier modes in each spatial dimension"
    )
    parser.add_argument(
        "--fno_hidden_channels",
        type=int,
        default=64,
        help="FNO hidden feature dimension"
    )
    parser.add_argument(
        "--fno_layers",
        type=int,
        default=4,
        help="Number of FNO layers"
    )
    parser.add_argument(
        "--fno_in_channels",
        type=int,
        default=1
    )
    parser.add_argument(
        "--fno_out_channels",
        type=int,
        default=1
    )
    parser.add_argument(
        "--fno_domain_padding",
        type=float,
        default=0.05
    )

    return parser.parse_args()


def forward_batch(model, pre, post, clinical):
    if hasattr(model, "sample"):
        return model(pre, post, clinical=clinical)
    return model(pre, post)


def sample_batch(model, pre, clinical):
    if hasattr(model, "sample"):
        return model.sample(pre, clinical=clinical)
    return model(pre)["pred"]


def train_one_epoch(model,loader,optimizer,device,ema=None):
    model.train()
    total_loss = 0
    pbar = tqdm(loader)
    for batch in pbar:
        pre = batch["input"].to(device)
        post = batch["target"].to(device)
        clinical = batch["condition"].to(device)
        output = forward_batch(model,pre,post,clinical)
        loss = output["loss"]
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        if ema is not None:
            ema.update(model)
        total_loss += loss.item()
        pbar.set_description("Train")

    return total_loss / len(loader)

# ==========================
# validation
# =========================


@torch.no_grad()
def validate(model,loader,device):
    model.eval()
    total_loss = 0
    metrics = {
        "MAE":0,
        "RMSE":0,
        "PSNR":0,
        "SSIM":0
    }

    for batch in tqdm(loader,desc="Val"):
        pre=batch["input"].to(device)
        post=batch["target"].to(device)
        clinical=batch["condition"].to(device)
        output=forward_batch(model,pre,post,clinical)
        total_loss += output["loss"].item()

        pred=sample_batch(model,pre,clinical)
        # =====================
        # Sampling prediction
        # =====================
        result=calculate_metrics(pred,post)

        for k in metrics:
            metrics[k]+=result[k]

    n=len(loader)
    total_loss/=n
    for k in metrics:
        metrics[k]/=n
    return (
        total_loss,
        metrics
    )
@torch.no_grad()
def test(model,loader,device,save_dir):
    model.eval()
    total_metrics = {
        "MAE": 0,
        "RMSE": 0,
        "PSNR": 0,
        "SSIM": 0
    }
    os.makedirs(
        save_dir,
        exist_ok=True
    )


    num_batches = len(loader)


    for idx, batch in tqdm(enumerate(loader),desc="Test"):
        pre = batch["input"].to(device)
        target = batch["target"].to(device)
        clinical = batch["condition"].to(device)

        pred=sample_batch(model,pre,clinical)

        metrics = calculate_metrics(pred,target)
        for k in total_metrics:
            total_metrics[k] += metrics[k]

        if idx < 20:
            error = torch.abs(
                pred-target
            )

            save_heatmap(pre[0],os.path.join(save_dir,f"{idx}_pre.png"))
            save_heatmap(target[0],os.path.join(save_dir,f"{idx}_gt.png"))
            save_heatmap(pred[0],os.path.join(save_dir,f"{idx}_pred.png"))
            save_heatmap(error[0],os.path.join(save_dir,f"{idx}_error.png"))
    for k in total_metrics:
        total_metrics[k] /= num_batches
    return total_metrics


@torch.no_grad()
def visualize_result(
        model,
        loader,
        device,
        epoch,
        save_dir
):
    model.eval()
    batch = next(iter(loader))
    pre = batch["input"].to(device)
    post = batch["target"].to(device)
    clinical = batch["condition"].to(device)

    generated=sample_batch(model,pre[:1],clinical[:1])
    # ==========================
    # remove batch dimension
    # ==========================
    pred = generated[0]
    gt = post[0]
    pre_img = pre[0]
    # ==========================
    # Error map
    # ==========================

    error = torch.abs(pred - gt)
    # 归一化到[0,1]用于heatmap显示
    #error = error / (error.max() + 1e-8)
    
    # ==========================
    # Save images
    # ==========================
    os.makedirs(
        save_dir,
        exist_ok=True
    )
    save_heatmap(
        pre_img,
        os.path.join(
            save_dir,
            f"epoch_{epoch}_pre.png"
        )
    )
    save_heatmap(
        gt,
        os.path.join(
            save_dir,
            f"epoch_{epoch}_gt.png"
        )
    )
    save_heatmap(
        pred,
        os.path.join(
            save_dir,
            f"epoch_{epoch}_pred.png"
        )
    )
    save_heatmap(
        error,
        os.path.join(
            save_dir,
            f"epoch_{epoch}_error.png"
        )
    )
# ==========================
# main
# ==========================
def seed_everything(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_model(args):
    if args.model == "fno":
        return FourierNeuralOperator(
            in_channels=args.fno_in_channels,
            out_channels=args.fno_out_channels,
            n_layers=args.fno_layers,
            hidden_channels=args.fno_hidden_channels,
            modes=args.fno_modes,
        )
    if args.model == "uno":
        return UnetNeuralOperator(
            in_channels=args.fno_in_channels,
            out_channels=args.fno_out_channels,
            hidden_channels=args.fno_hidden_channels,
        )

    encoder = ConditionEncoder()
    unet = ConditionalUNet(
        discrete=args.model != "fm",
        clinical_dim=len(args.conditions),
        clinical_embed_dim=args.clinical_embed_dim,
        fusion_method=args.fusion_method,
    )
    if args.model == "bbdm":
        return BBDM(
            encoder, unet, timesteps=args.time_steps,
            objective=args.objective, sample_steps=args.sample_steps,
        )
    if args.model == "fm":
        return FlowMatching(encoder, unet, method=args.ode_solver)
    return CDDPM(
        encoder, unet, timesteps=args.time_steps,
        sampler=args.sampler, sample_steps=args.sample_steps,
    )


def get_clinical_stats(dataframe, train_labels, columns):
    if not columns:
        return None
    missing = set(columns) - set(dataframe.columns)
    if missing:
        raise ValueError(f"Excel 缺少临床参数列: {sorted(missing)}")
    numeric = dataframe.loc[:, columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise ValueError("临床参数列包含缺失或无穷值，请先处理再训练")
    train_mask = dataframe["label_index"].isin(train_labels)
    values = numeric.loc[train_mask].to_numpy(dtype=np.float64)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std < 1e-8] = 1.0
    return {"columns": list(columns), "mean": mean.tolist(), "std": std.tolist()}


def run_fold(args, dataframe, fold, device):
    fold_number = fold["fold"]
    fold_dir = Path(args.save_dir) / f"fold_{fold_number:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Fold {fold_number}/10: "
        f"train={len(fold['train'])}, val={len(fold['val'])}, test={len(fold['test'])}"
    )

    clinical_stats = get_clinical_stats(
        dataframe, fold["train"], args.conditions if args.fusion_method != "none" else [],
    )
    if clinical_stats is not None:
        with (fold_dir / "clinical_stats.json").open("w", encoding="utf-8") as f:
            json.dump(clinical_stats, f, ensure_ascii=False, indent=2)

    augmentation = DataTransform(hflip_prob=0.5, vflip_prob=0.5, rotation=True)
    dataset_args = dict(
        opt=args, excel_path=args.excel_path, data_root=args.data_root,
        image_size=args.image_size, dataframe=dataframe,
    )
    if clinical_stats is not None:
        dataset_args.update(
            clinical_mean=clinical_stats["mean"], clinical_std=clinical_stats["std"],
        )
    train_set = CornealTopoDataset(
        **dataset_args, labels=fold["train"], transforms=augmentation,
    )
    val_set = CornealTopoDataset(**dataset_args, labels=fold["val"])
    test_set = CornealTopoDataset(**dataset_args, labels=fold["test"])
    loader_args = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)
    test_loader = DataLoader(test_set, shuffle=False, **loader_args)

    # Every fold starts with new weights, optimizer, scheduler and EMA.
    seed_everything(args.seed + fold_number)
    model = build_model(args).to(device)
    ema = EMA(model, decay=args.ema_decay) if args.use_ema else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    writer = SummaryWriter(log_dir=str(fold_dir / "runs"))
    best_loss = float("inf")
    start = time.time()

    try:
        for epoch in range(args.epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, device, ema=ema)
            scheduler.step()
            writer.add_scalar("loss/train", train_loss, epoch)
            print(
                f"Fold {fold_number} Epoch {epoch + 1}/{args.epochs} "
                f"train_loss={train_loss:.6f}, lr={optimizer.param_groups[0]['lr']:.6g}, "
                f"elapsed={(time.time() - start) / 3600:.2f} h"
            )
            if epoch % 10 != 0 and epoch != args.epochs - 1:
                continue

            eval_model = ema.ema_model if ema is not None else model
            val_loss, metrics = validate(eval_model, val_loader, device)
            writer.add_scalar("loss/val", val_loss, epoch)
            for name, value in metrics.items():
                writer.add_scalar(f"val/{name}", value, epoch)
            print(f"Fold {fold_number} Epoch {epoch + 1}: val_loss={val_loss:.6f}, metrics={metrics}")
            visualize_result(eval_model, val_loader, device, epoch, str(fold_dir / "visual"))

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({
                    "epoch": epoch, "model": model.state_dict(),
                    "ema": ema.ema_model.state_dict() if ema is not None else None,
                    "optimizer": optimizer.state_dict(), "loss": best_loss,
                    "model_type": args.model, "fusion_method": args.fusion_method,
                    "clinical_embed_dim": args.clinical_embed_dim,
                    "clinical_stats": clinical_stats,
                }, fold_dir / "best.pth")

        torch.save({
            "epoch": args.epochs - 1, "model": model.state_dict(),
            "ema": ema.ema_model.state_dict() if ema is not None else None,
            "optimizer": optimizer.state_dict(), "loss": best_loss,
            "model_type": args.model, "fusion_method": args.fusion_method,
            "clinical_embed_dim": args.clinical_embed_dim,
            "clinical_stats": clinical_stats,
        }, fold_dir / "final.pth")
    finally:
        writer.close()

    results = {}
    for checkpoint_name in ("best", "final"):
        # These checkpoints were just written by this training run. Neural operator
        # state dictionaries can carry callable metadata (such as GELU), which
        # PyTorch's restricted weights-only unpickler cannot deserialize.
        checkpoint = torch.load(
            fold_dir / f"{checkpoint_name}.pth", map_location=device, weights_only=False,
        )
        weights = checkpoint["ema"] if args.use_ema else checkpoint["model"]
        # Some neuralop versions also store non-parameter metadata as a key.
        weights.pop("_metadata", None)
        model.load_state_dict(weights)
        metrics = test(model, test_loader, device, str(fold_dir / "test_results" / checkpoint_name))
        results[checkpoint_name] = metrics
        with (fold_dir / "test_results" / f"{checkpoint_name}.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"Fold {fold_number} {checkpoint_name} test: {metrics}")
    return {"fold": fold_number, "sizes": {
        name: len(fold[name]) for name in ("train", "val", "test")
    }, "best": results["best"], "final": results["final"]}


def write_summary(results, path, args, split_file):
    aggregate = {}
    for checkpoint_name in ("best", "final"):
        if not results:
            continue
        aggregate[checkpoint_name] = {
            metric: {
                "mean": float(np.mean([fold[checkpoint_name][metric] for fold in results])),
                "std": float(np.std([fold[checkpoint_name][metric] for fold in results])),
            }
            for metric in results[0][checkpoint_name]
        }
    summary = {
        "model": args.model, "split_file": str(split_file),
        "fusion_method": args.fusion_method, "clinical_features": list(args.conditions),
        "clinical_embed_dim": args.clinical_embed_dim,
        "requested_folds": args.k_folds, "completed_folds": len(results),
        "fold_results": results, "aggregate": aggregate,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main():
    args = get_args()
    if args.fusion_method != "none":
        if args.model in {"fno", "uno"}:
            raise ValueError("临床参数融合仅支持 cddpm、bbdm 和 fm")
        if not args.conditions:
            raise ValueError("启用临床参数融合时必须通过 --conditions 指定术前临床参数列")
        if any(name == "label_index" or name.startswith("post_") for name in args.conditions):
            raise ValueError("--conditions 只能使用术前可获得的参数，不能包含 label_index 或 post_ 列")
        if len(set(args.conditions)) != len(args.conditions):
            raise ValueError("--conditions 不应包含重复列")
        if args.clinical_embed_dim < 1:
            raise ValueError("--clinical_embed_dim 必须大于零")
    if not 1 <= args.k_folds <= 10:
        raise ValueError("--k_folds 必须介于 1 和 10 之间")
    if args.epochs < 1:
        raise ValueError("--epochs 必须大于零")
    seed_everything(args.seed)
    device = torch.device(
        "cpu" if args.device.startswith("cuda") and not torch.cuda.is_available() else args.device
    )
    output_dir = Path(args.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_file = Path(args.split_file) if args.split_file else Path(args.data_root) / "patient_10fold_splits.json"
    dataframe = pd.read_excel(args.excel_path)
    folds = load_or_create_cv_splits(dataframe, split_file, seed=args.seed)
    print(f"10-fold patient-level splits: {split_file}")

    results = []
    for fold in folds[:args.k_folds]:
        results.append(run_fold(args, dataframe, fold, device))
        write_summary(results, output_dir / "cv_summary.json", args, split_file)
    print(f"Completed {len(results)} folds; summary: {output_dir / 'cv_summary.json'}")


if __name__ == "__main__":
    main()
