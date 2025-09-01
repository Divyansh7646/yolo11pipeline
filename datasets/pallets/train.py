from ultralytics import YOLO
import torch
import json
import os

def main():
    # ==========================
    # Config
    # ==========================
    DATA_YAML = "datasets/pallets/data.yaml"   # Path to your dataset yaml
    MODEL = "yolo11l.pt"      # Large model (fits 8GB GPU)
    EPOCHS = 50               # Epochs per stage
    BATCH_SIZE = 8            # Safe for 8GB GPU
    IMG_SIZE = 640
    DEVICE = 0
    WORKERS = 8
    LOG_FILE = "training_metrics.json"

    # ==========================
    # Check device
    # ==========================
    if torch.cuda.is_available():
        print(f"✅ Training will run on GPU: {torch.cuda.get_device_name(DEVICE)}")
    else:
        print("⚠️ CUDA not available. Training will run on CPU")
        DEVICE = "cpu"

    # ==========================
    # Load YOLO model
    # ==========================
    model = YOLO(MODEL)

    # ==========================
    # JSON Logger
    # ==========================
    class JSONLogger:
        def __init__(self, log_file):
            self.log_file = log_file
            self.metrics = []

        def on_fit_epoch_end(self, trainer):
            stats = trainer.metrics
            log_entry = {
                "epoch": trainer.epoch + 1,
                "train/box_loss": stats.get("train/box_loss", None),
                "train/cls_loss": stats.get("train/cls_loss", None),
                "train/dfl_loss": stats.get("train/dfl_loss", None),
                "val/precision": stats.get("metrics/precision(B)", None),
                "val/recall": stats.get("metrics/recall(B)", None),
                "val/mAP50": stats.get("metrics/mAP50(B)", None),
                "val/mAP50-95": stats.get("metrics/mAP50-95(B)", None),
            }
            self.metrics.append(log_entry)
            with open(self.log_file, "w") as f:
                json.dump(self.metrics, f, indent=4)

    json_logger = JSONLogger(LOG_FILE)
    model.add_callback("on_fit_epoch_end", json_logger.on_fit_epoch_end)

    # ==========================
    # Stage 1: Freeze backbone (train head only)
    # ==========================
    for name, param in model.model.named_parameters():
        if "head" not in name:   # keep head trainable
            param.requires_grad = False
    print("➡️ Stage 1: Backbone frozen, head trainable")

    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        device=DEVICE,
        workers=WORKERS,
        optimizer="AdamW",
        lr0=0.0005,
        lrf=0.01,
        cos_lr=True,
        patience=15,
        augment=False,
        pretrained=True,
        save=True,
        save_period=5,
        project="runs/train",
        name="igps_vs_nonigps_stage1",
        amp=True
    )

    # ==========================
    # Stage 2: Unfreeze full model
    # ==========================
    for param in model.model.parameters():
        param.requires_grad = True
    print("➡️ Stage 2: Entire model unfrozen for fine-tuning")

    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        device=DEVICE,
        workers=WORKERS,
        optimizer="AdamW",
        lr0=0.0003,
        lrf=0.01,
        cos_lr=True,
        patience=20,
        augment=False,
        pretrained=True,
        save=True,
        save_period=5,
        project="runs/train",
        name="igps_vs_nonigps_stage2",
        amp=True
    )

if __name__ == "__main__":
    main()
