import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from sklearn.model_selection import train_test_split

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
from metrics import calculate_metrics
from util.util import tensor2heatmap,save_heatmap
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
    parser.add_argument("--model",type=str , default="cddpm")
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

    parser.add_argument("--use_ema",action="store_true")
    parser.add_argument("--ema_decay",type=float,default=0.995)
    parser.add_argument("--objective",type=str , default="grad")
    parser.add_argument("--conditions", nargs="+", default=[])
    parser.add_argument("--targets", nargs="+", default=[])


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


def train_one_epoch(model,loader,optimizer,device,ema=None):
    model.train()
    total_loss = 0
    pbar = tqdm(loader)
    for batch in pbar:
        pre = batch["input"].to(device)
        post = batch["target"].to(device)
        # ------------------
        # CDDPM forward
        # --------------------
        output = model(pre,post)
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
        # diffusion loss
        output=model(pre,post)
        total_loss += output["loss"].item()

        if hasattr(model,"sample"):
            # diffusion / BBDM / FM
            pred=model.sample(pre)
        else:
            # Neural Operator / FNO
            pred=model(pre)["pred"]
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

        if hasattr(model,"sample"):
            # diffusion / BBDM / FM
            pred=model.sample(pre)
        else:
            # Neural Operator / FNO
            pred=model(pre)["pred"]

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

    if hasattr(model,"sample"):
        generated=model.sample(
            pre[:1]
        )

    else:
        generated=model(
            pre[:1]
        )["pred"]
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

def main():
    seed_everything(42)
    writer = SummaryWriter(log_dir="./runs/cddpm")

    args=get_args()
    device=torch.device(
        args.device
        if torch.cuda.is_available()
        else "cpu"
    )
    os.makedirs(
        args.save_dir,
        exist_ok=True
    )
    # --------------------------
    # Dataset
    # --------------------------
    transform = DataTransform(
        hflip_prob=0.5,
        vflip_prob=0.5,
        rotation=True,
    )
    dataset = CornealTopoDataset(
        opt=args,
        excel_path=args.excel_path,
        data_root=args.data_root,
        input_maps=["TOTAL CORNEAL REFRACTIVE POWER"],
        output_maps=["TOTAL CORNEAL REFRACTIVE POWER"],
        image_size=128,
        normalize=True
    )

    split_file = Path(args.data_root) / "data_split.json"

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
        opt=args,
        excel_path=args.excel_path,
        data_root=args.data_root,
        image_size=128,
        transforms=transform,
        indices=train_idx
    )

    val_set = CornealTopoDataset(
        opt=args,
        excel_path=args.excel_path,
        data_root=args.data_root,
        image_size=128,
        transforms=None,
        indices=val_idx
    )

    test_set = CornealTopoDataset(
        opt=args,
        excel_path=args.excel_path,
        data_root=args.data_root,
        image_size=128,
        transforms=None,
        indices=test_idx
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4,pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4,pin_memory=True)
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    # --------------------------
    # Model
    # --------------------------
    condition_encoder=ConditionEncoder()
    print(f"Using {args.model} model.")
    if args.model == "bbdm":
        unet=ConditionalUNet()

        model=BBDM(condition_encoder,unet,timesteps=args.time_steps,objective=args.objective)
    elif args.model == "fm":
        unet=ConditionalUNet(discrete=False)

        model=FlowMatching(condition_encoder,unet,method=args.ode_solver)
    elif args.model == "fno":
        model = FourierNeuralOperator(in_channels=args.fno_in_channels,out_channels=args.fno_out_channels,n_layers=args.fno_layers,hidden_channels=args.fno_hidden_channels,modes=args.fno_modes)
    elif args.model == "uno":
        model = UnetNeuralOperator(in_channels=args.fno_in_channels,out_channels=args.fno_out_channels,n_layers=5,hidden_channels=args.fno_hidden_channels)
    else:
        model=CDDPM(
            condition_encoder,
            unet,
            timesteps=args.time_steps,
            sampler = args.sampler,
            sample_steps=args.sample_steps
        )
    
    model=model.to(device)

    ema = None
    if args.use_ema:
        print("Using EMA.")
        ema = EMA(model=model,decay=args.ema_decay)
 
  
    # --------------------------
    # Optimizer
    # --------------------------
    optimizer=torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )


    # --------------------------
    # Training
    # --------------------------


    best_loss=999999
    val_loss=0
    print(f"Start Training for {args.epochs} Epochs")
    start_time=time.time()
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )

    for epoch in range(args.epochs):
        epoch_start=time.time()
        train_loss=train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            ema=ema
        )

        writer.add_scalar(
            "loss/train",
            train_loss,
            epoch
        )


        scheduler.step()
        epoch_time=time.time()-epoch_start
        total_time=time.time()-start_time
       
        print(f"""Train:Epoch [{epoch+1}/{args.epochs}]\tLR:{optimizer.param_groups[0]["lr"]}\tTrain Loss:{train_loss:.6f}\tEpoch Time:{epoch_time:.2f}s\tTotal Time:{total_time/3600:.2f} h""")
        if epoch % 10 == 0:

            eval_model = ema.ema_model if ema is not None else model
   
            val_loss,metrics=validate(eval_model,val_loader,device)
      


            print(
            f"""Val:Epoch [{epoch+1}/{args.epochs}]\tLR:{optimizer.param_groups[0]["lr"]:.6}\tTrain Loss:{train_loss:.6f}\tVal Loss:{val_loss:.6f}\tMAE:{metrics['MAE']:.6f}\tRMSE:{metrics['RMSE']:.6f}\tPSNR:{metrics['PSNR']:.3f}dB\tSSIM:{metrics['SSIM']:.4f}\tEpoch Time:{epoch_time:.2f}s\tTotal Time:{total_time/3600:.2f} h"""
            )
            writer.add_scalar("loss/val", val_loss,epoch)
            visualize_result( eval_model,val_loader,device,epoch,args.save_dir+"/visual")


            # save best
            if val_loss < best_loss:
                best_loss=val_loss
                torch.save(
                    {
                        "epoch":epoch,
                        "model":
                        model.state_dict(),
                        "ema":ema.ema_model.state_dict() if ema is not None else None,
                        "optimizer":
                        optimizer.state_dict(),
                        "loss":
                        best_loss
                    },os.path.join(args.save_dir,"best_cddpm.pth"))


    torch.save(
                        {
                            "epoch":epoch,
                            "model":
                            model.state_dict(),
                            "ema":ema.ema_model.state_dict() if ema is not None else None,
                            "optimizer":
                            optimizer.state_dict(),
                            "loss":
                            best_loss
                        },
                        os.path.join(
                            args.save_dir,
                            "final_cddpm.pth"
                        )

                    )
    print("Loading best checkpoint for testing...")

    checkpoint=torch.load(
        os.path.join(
            args.save_dir,
            "best_cddpm.pth"

        ),
        map_location=device,
            weights_only=False
    )
    if args.use_ema and checkpoint["ema"] is not None:
        print("Load EMA weights.")
        model.load_state_dict(checkpoint["ema"])
    else:
        print("Load model weights.")
        #model.load_state_dict(checkpoint["model"])
        state_dict = checkpoint["model"]
        if "_metadata" in state_dict:
            del state_dict["_metadata"]

        model.load_state_dict(
            state_dict,
            strict=True
        )

    test_metrics=test(
        model,
        test_loader,
        device,
        args.save_dir+"/test_results/best"
    )


    print(
    """
    ==========================
    Best Test Results
    ==========================
    """
    )


    for k,v in test_metrics.items():

        print(
            f"{k}: {v:.6f}"
        )

    with open(args.save_dir+"/test_results/best.json","w") as f:
        json.dump(
            test_metrics,
            f,
            indent=4
        )
    checkpoint=torch.load(
        os.path.join(
            args.save_dir,
            "final_cddpm.pth"

        ),
        map_location=device,
        weights_only=False
    )
    if args.use_ema and checkpoint["ema"] is not None:
        print("Load EMA weights.")
        model.load_state_dict(checkpoint["ema"])
    else:
        print("Load model weights.")
        state_dict = checkpoint["model"]
        if "_metadata" in state_dict:
            del state_dict["_metadata"]

        model.load_state_dict(
            state_dict,
            strict=True
        )
    test_metrics=test(
        model,
        test_loader,
        device,
        args.save_dir+"/test_results/final"
    )


    print(
    """
    ==========================
    Final Test Results
    ==========================
    """
    )


    for k,v in test_metrics.items():

        print(
            f"{k}: {v:.6f}"
        )

    with open( args.save_dir+"/test_results/final.json","w") as f:
        json.dump(
            test_metrics,
            f,
            indent=4
        )



if __name__=="__main__":

    main()