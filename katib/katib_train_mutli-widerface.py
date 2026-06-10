#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from ultralytics import YOLO
import os
import torch
import subprocess
import shutil
import sys
import re

def normalize_yolo_model_path(model_path: str) -> str:
    """
    Normalize YOLO model yaml names:

    Examples:
        yolo11s.yaml        -> yolo11.yaml
        yolo26n.yaml        -> yolo26.yaml
        yolov3m.yaml        -> yolov3.yaml
        yolo11n-cls.yaml    -> yolo11-cls.yaml
        yolo11s-seg.yaml    -> yolo11-seg.yaml
    """

    filename = os.path.basename(model_path)

    # 1. split extension
    name, ext = os.path.splitext(filename)

    # 2. remove scale letter only if it is at the end of base part
    #    match: n/s/m/l/x just before optional "-xxx"
    name = re.sub(r"^(yolo\w+?)[nsmxl](?=-|$)", r"\1", name)

    return os.path.join(os.path.dirname(model_path), name + ext)

def gpu_system_check():
    print("\n[Katib] ===== GPU SYSTEM CHECK =====", flush=True)

    print("[Torch] cuda available:", torch.cuda.is_available(), flush=True)
    print("[Torch] cuda version:", torch.version.cuda, flush=True)

    try:
        result = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print(result.stdout, flush=True)
    except Exception as e:
        print("[nvidia-smi] FAILED:", e, flush=True)

    if not torch.cuda.is_available():
        raise RuntimeError("❌ CUDA NOT AVAILABLE - STOP TRAINING")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', type=str, default='0')

    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--patience', type=int, default=100)

    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--optimizer', type=str, required=True)

    # New parameter
    # parser.add_argument('--fl_gamma', type=float, default=1.5)
    # parser.add_argument('--fl_alpha', type=float, default=0.25)
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--cls_pw', type=float, default=1.0)
    parser.add_argument('--focal_loss', action='store_true')
    parser.add_argument('--dump_batches_dir', type=str, default=None)  # 指定 dump_batches_dir 才拷贝训练 batch；不写路径则关闭；dump_batches_per_epoch 默认 1
    
    parser.add_argument('--project', type=str, default='/katib/output/detect')
    parser.add_argument('--name', type=str, default=None)

    args = parser.parse_args()

    print("\n[Katib] ===== ENV CHECK =====", flush=True)
    print(f"[Katib] torch version: {torch.__version__}", flush=True)
    print(f"[Katib] cuda available: {torch.cuda.is_available()}", flush=True)

    if torch.cuda.is_available():
        print(f"[Katib] cuda device count: {torch.cuda.device_count()}", flush=True)
        print(f"[Katib] device name: {torch.cuda.get_device_name(0)}", flush=True)

    lr = float(args.lr)
    batch = int(args.batch)
    optimizer = args.optimizer
    model_cfg = normalize_yolo_model_path(args.model)
    
    trial_name = args.name if args.name else f"{batch}-{optimizer}-lr{lr}"
    out_dir = os.path.join(args.project, trial_name)
    os.makedirs(out_dir, exist_ok=True)

    # ============================================================
    # copy file
    # ============================================================
    print("\n[Katib] ===== COPY FILE =====", flush=True)
    try:
        copy_files = [model_cfg, args.data]

        # copy current running script
        current_script = os.path.abspath(__file__)
        shutil.copy2(current_script, os.path.join(out_dir, os.path.basename(current_script)))

         # copy file list
        for f in copy_files:
            if os.path.exists(f):
                shutil.copy2(f, os.path.join(out_dir, os.path.basename(f)))
                print(f"[Katib] copied: {f}", flush=True)
            else:
                print(f"[Katib] WARNING: file not found -> {f}", flush=True)
    
        print(f"[Katib] copied script: {current_script}", flush=True)
        print("[Katib] all files copied to output dir", flush=True)
    
    except Exception as e:
        print(f"[Katib] script copy failed: {e}", flush=True)
            

    metrics_file = "/var/log/katib/metrics.log"
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)

    print("[Katib] Training started", flush=True)

    model = YOLO(args.model)

    def on_epoch_end(trainer):
        epoch = trainer.epoch + 1

        try:
            metrics = trainer.metrics or {}
            wider_easy = float(metrics.get("metrics/wider_easy_ap", 0.0) or 0.0)
            wider_medium = float(metrics.get("metrics/wider_medium_ap", 0.0) or 0.0)
            wider_hard = float(metrics.get("metrics/wider_hard_ap", 0.0) or 0.0)
        except Exception as e:
            print(f"[Katib] metric read failed: {e}", flush=True)
            wider_easy = wider_medium = wider_hard = 0.0

        # Katib File Collector 主目标用 wider_easy_ap（与 Experiment objectiveMetricName 一致）
        line = (
            f"epoch={epoch} "
            f"wider_easy_ap={wider_easy:.6f} "
            f"wider_medium_ap={wider_medium:.6f} "
            f"wider_hard_ap={wider_hard:.6f}"
        )

        try:
            with open(metrics_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())

            print(f"[Katib] {line}", flush=True)

        except Exception as e:
            print(f"[Katib] metrics write failed: {e}", flush=True)

    model.add_callback("on_train_epoch_end", on_epoch_end)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device,

        workers=args.workers,
        cache=args.cache,
        patience=args.patience,
                
        lr0=args.lr,
        batch=batch,
        optimizer=args.optimizer,

        # fl_gamma=args.fl_gamma,
        # fl_alpha=args.fl_alpha,
        pretrained=args.pretrained,
        cls_pw=args.cls_pw,
        focal_loss=args.focal_loss,
        dump_batches_dir = args.dump_batches_dir,
        
        project=args.project,
        name=trial_name,
        exist_ok=True
    )

    # ============================================================
    # print and copy metrics.log 
    # ============================================================
    print("\n[Katib] ===== metrics.log CONTENT =====", flush=True)
    with open(metrics_file, "r", encoding="utf-8") as f:
        print(f.read(), flush=True)

    try:
        shutil.copy2(metrics_file, os.path.join(out_dir, "metrics.log"))
        print("[Katib] metrics.log copied to output dir", flush=True)
    except Exception as e:
        print(f"[Katib] metrics copy failed: {e}", flush=True)

    print("\n[Katib] ===== metrics.log FULL PATH =====", flush=True)
    print(f"[Katib] metrics.log path: {metrics_file}", flush=True)

    print(f"[Katib] Done. Results saved in {out_dir}", flush=True)


if __name__ == "__main__":
    gpu_system_check()
    main()