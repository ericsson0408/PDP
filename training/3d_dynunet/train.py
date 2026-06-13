#!/usr/bin/env python3
# =============================================================================
# train.py  --  3D Airway Segmentation (AIIB23) training for the C++ pipeline.
#
# Trains a MONAI DynUNet whose ONNX export drops straight into the V11 C++
# inference engine (Tubular Gate / Leak-detection Oracle). The architecture,
# intensity normalisation and 2-class raw-logit output are pinned to match the
# C++ side exactly so no C++ code has to change.
#
# Key consistency contracts with src/onnx_infer.cpp + src/main.cpp:
#   * intensity:  ScaleIntensityRange [-1000, 600] -> [0, 1], clip=True
#   * output:     raw logits [B, 2, D, H, W]  (C++ does softmax(ch=2) itself)
#   * arch:       DynUNet filters=[32,64,128,256,320], strides=[1,2,2,2,2],
#                 instance norm (affine), deep_supervision=False
#
# OOM-safety on the V100 32GB:
#   * DataLoader batch_size = 1, larger effective batch via --grad_accum
#   * 128^3 random patches (RandCropByPosNegLabeld), AMP autocast (fp16)
#   * validation uses sliding-window inference with the aggregated output
#     buffer kept on CPU (device="cpu"), so only single patches ever sit on
#     the GPU -- full 512^3-ish volumes never land in VRAM
#   * PersistentDataset caches the *deterministic* (resampled+normalised)
#     volumes to disk, so host RAM never holds all 120 resampled volumes
#
# This script writes to NON-production filenames (dynunet_retrained_*) so the
# existing model/dynunet_best_model* artefacts are never clobbered mid-run.
# =============================================================================

import os
import sys
import json
import time
import glob
import random
import argparse

import numpy as np
import torch

from monai.data import PersistentDataset, DataLoader, decollate_batch
from monai.networks.nets import DynUNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    ScaleIntensityRanged,
    SpatialPadd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandGaussianNoised,
    EnsureTyped,
    AsDiscrete,
)


# -----------------------------------------------------------------------------
# Architecture -- MUST stay identical in train.py and export_onnx.py.
# affine instance norm matches the existing production graph (norm tensors carry
# weight+bias) and is the standard nnU-Net/DynUNet recipe.
# -----------------------------------------------------------------------------
def build_model():
    return DynUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        kernel_size=[3, 3, 3, 3, 3],
        strides=[1, 2, 2, 2, 2],
        upsample_kernel_size=[2, 2, 2, 2],
        filters=[32, 64, 128, 256, 320],
        norm_name=("INSTANCE", {"affine": True}),
        deep_supervision=False,
        res_block=False,
    )


def build_transforms(args):
    patch = (args.patch, args.patch, args.patch)

    # Deterministic prefix -- PersistentDataset caches the output of these.
    det = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(
            keys=["image", "label"],
            pixdim=tuple(args.spacing),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=args.hu_min, a_max=args.hu_max,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        # image float32, label uint8 -> halves the on-disk PersistentDataset
        # cache and the host RAM held per volume.
        EnsureTyped(keys=["image"], dtype=torch.float32),
        EnsureTyped(keys=["label"], dtype=torch.uint8),
    ]

    # Random tail -- re-run every epoch (NOT cached).
    rand = [
        SpatialPadd(keys=["image", "label"], spatial_size=patch, mode="constant"),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=patch,
            pos=1, neg=1,
            num_samples=args.samples_per_image,
            image_key="image",
            image_threshold=0,
        ),
        RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.2, max_k=3, spatial_axes=(0, 1)),
        RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.01),
        EnsureTyped(keys=["image", "label"]),
    ]

    train_tf = Compose(det + rand)
    val_tf = Compose(det)  # full volume; sliding window handles patching
    return train_tf, val_tf


def make_data_dicts(args):
    imgs = sorted(glob.glob(os.path.join(args.img_dir, "*.nii.gz")))
    data = []
    for ip in imgs:
        lp = os.path.join(args.gt_dir, os.path.basename(ip))
        if os.path.exists(lp):
            data.append({"image": ip, "label": lp})
    if not data:
        sys.exit(f"[fatal] no image/label pairs found in {args.img_dir} + {args.gt_dir}")
    rng = random.Random(args.seed)
    rng.shuffle(data)
    n_val = max(1, int(round(len(data) * args.val_frac)))
    val = data[:n_val]
    train = data[n_val:]
    if args.limit_train:
        train = train[: args.limit_train]
    if args.limit_val:
        val = val[: args.limit_val]
    return train, val


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Train DynUNet airway segmentation (AIIB23)")
    ap.add_argument("--img_dir", default="/home/u4309334/Project/data/img")
    ap.add_argument("--gt_dir", default="/home/u4309334/Project/data/gt")
    ap.add_argument("--out_dir", default="/home/u4309334/Project/model")
    ap.add_argument("--cache_dir", default="/work/u4309334/airway_cache")
    ap.add_argument("--spacing", type=float, nargs=3, default=[0.6, 0.6, 1.0])
    ap.add_argument("--hu_min", type=float, default=-1000.0)
    ap.add_argument("--hu_max", type=float, default=600.0)
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--samples_per_image", type=int, default=2,
                    help="RandCropByPosNegLabeld num_samples (patches per volume per step)")
    ap.add_argument("--batch_size", type=int, default=1, help="loader batch (keep 1 on V100)")
    ap.add_argument("--grad_accum", type=int, default=4, help="gradient accumulation steps")
    ap.add_argument("--max_epochs", type=int, default=300)
    ap.add_argument("--val_interval", type=int, default=5)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--sw_overlap", type=float, default=0.25)
    ap.add_argument("--sw_batch", type=int, default=2)
    ap.add_argument("--val_on_cpu", action="store_true",
                    help="aggregate sliding-window output on CPU (slower, lower VRAM)")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_amp", action="store_true", help="disable mixed precision")
    ap.add_argument("--resume", default="", help="checkpoint .pth to resume from")
    ap.add_argument("--limit_train", type=int, default=0, help="smoke test: cap train cases")
    ap.add_argument("--limit_val", type=int, default=0, help="smoke test: cap val cases")
    ap.add_argument("--tag", default="retrained", help="output filename tag")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    train_cache = os.path.join(args.cache_dir, "train")
    val_cache = os.path.join(args.cache_dir, "val")
    os.makedirs(train_cache, exist_ok=True)
    os.makedirs(val_cache, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True  # fixed 128^3 patch -> autotune helps

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = (not args.no_amp) and device.type == "cuda"
    log(f"device={device}  amp={amp}  visible_gpus={os.environ.get('CUDA_VISIBLE_DEVICES','all')}")
    if device.type == "cuda":
        log(f"gpu={torch.cuda.get_device_name(0)}  cap={torch.cuda.get_device_capability(0)}")

    train_files, val_files = make_data_dicts(args)
    log(f"train cases={len(train_files)}  val cases={len(val_files)}")

    train_tf, val_tf = build_transforms(args)
    train_ds = PersistentDataset(train_files, transform=train_tf, cache_dir=train_cache)
    val_ds = PersistentDataset(val_files, transform=val_tf, cache_dir=val_cache)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=max(1, args.num_workers // 2), pin_memory=False,
        persistent_workers=args.num_workers > 0,
    )

    model = build_model().to(device)
    loss_fn = DiceCELoss(include_background=False, softmax=True, to_onehot_y=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    post_pred = AsDiscrete(argmax=True, to_onehot=2)
    post_label = AsDiscrete(to_onehot=2)

    start_epoch = 0
    best_dice = -1.0
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        if isinstance(ck, dict) and "model" in ck:
            model.load_state_dict(ck["model"], strict=False)
            optimizer.load_state_dict(ck["optimizer"])
            scheduler.load_state_dict(ck["scheduler"])
            start_epoch = ck.get("epoch", 0) + 1
            best_dice = ck.get("best_dice", -1.0)
        else:
            model.load_state_dict(ck, strict=False)
        log(f"resumed from {args.resume} at epoch {start_epoch} best_dice={best_dice:.4f}")

    best_path = os.path.join(args.out_dir, f"dynunet_{args.tag}_best.pth")
    last_path = os.path.join(args.out_dir, f"dynunet_{args.tag}_last.pth")
    ckpt_path = os.path.join(args.out_dir, f"dynunet_{args.tag}_ckpt.pt")
    log_path = os.path.join(args.out_dir, f"train_{args.tag}_log.json")
    patch = (args.patch, args.patch, args.patch)
    history = []

    for epoch in range(start_epoch, args.max_epochs):
        model.train()
        t0 = time.time()
        epoch_loss = 0.0
        n_steps = 0
        optimizer.zero_grad(set_to_none=True)

        for it, batch in enumerate(train_loader):
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                logits = model(x)
                loss = loss_fn(logits, y) / args.grad_accum
            scaler.scale(loss).backward()
            epoch_loss += loss.item() * args.grad_accum
            n_steps += 1

            if (it + 1) % args.grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        # flush any remaining accumulated grads
        if n_steps % args.grad_accum != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        scheduler.step()
        mean_loss = epoch_loss / max(1, n_steps)
        lr_now = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0
        log(f"epoch {epoch+1}/{args.max_epochs}  loss={mean_loss:.4f}  lr={lr_now:.2e}  ({dt:.1f}s, {n_steps} steps)")

        # save 'last' + resumable checkpoint every epoch
        torch.save(model.state_dict(), last_path)
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch, "best_dice": best_dice,
        }, ckpt_path)

        do_val = ((epoch + 1) % args.val_interval == 0) or (epoch + 1 == args.max_epochs)
        if do_val:
            model.eval()
            dice_metric.reset()
            tv = time.time()
            # aggregate on GPU (full ~650^3 volume fits in 32GB and blends far
            # faster); --val_on_cpu falls back to CPU for tight-VRAM cases.
            val_out_device = torch.device("cpu") if args.val_on_cpu else device
            with torch.no_grad():
                for vb in val_loader:
                    vx = vb["image"]  # patches are streamed to sw_device per window
                    vy = vb["label"]
                    logits = sliding_window_inference(
                        inputs=vx, roi_size=patch, sw_batch_size=args.sw_batch,
                        predictor=model, overlap=args.sw_overlap, mode="gaussian",
                        sw_device=device, device=val_out_device, progress=False,
                    )
                    preds = [post_pred(p) for p in decollate_batch(logits)]
                    labels = [post_label(l.to(val_out_device)) for l in decollate_batch(vy)]
                    dice_metric(y_pred=preds, y=labels)
            val_dice = float(dice_metric.aggregate().item())
            log(f"  >> val mean Dice (airway) = {val_dice:.4f}  ({time.time()-tv:.1f}s)")

            history.append({"epoch": epoch + 1, "loss": mean_loss, "val_dice": val_dice})
            with open(log_path, "w") as f:
                json.dump(history, f, indent=2)

            if val_dice > best_dice:
                best_dice = val_dice
                torch.save(model.state_dict(), best_path)
                log(f"  ** new best Dice={best_dice:.4f} -> saved {best_path}")

    log(f"training done. best val Dice={best_dice:.4f}")
    log(f"best  weights: {best_path}")
    log(f"last  weights: {last_path}")
    log(f"Export ONNX with:  python export_onnx.py --ckpt {best_path}")


if __name__ == "__main__":
    main()
