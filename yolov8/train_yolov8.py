"""
使用 Ultralytics YOLOv8 對「棒球」做微調訓練。

1. 先跑 prepare_dataset.py 抽幀。
2. 用標註工具將球標成 YOLO 格式 (txt)，放在同路徑下：
   - images/train/xxx.jpg 對應 labels/train/xxx.txt
   - images/val/xxx.jpg   對應 labels/val/xxx.txt（可先全部放 train，再手動切一部分到 val）
"""

import os
from pathlib import Path

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DATA_CONFIG = ROOT / "yolov8" / "data_baseball.yaml"


def get_device():
    # 在 M1/M2 Mac 上，優先使用 mps；否則用 cpu
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "0"
    return "cpu"


def main():
    device = get_device()
    print(f"Using device: {device}")

    # 以 yolov8n 預訓練權重為基礎做微調（速度較快）
    model = YOLO("yolov8n.pt")

    model.train(
        data=str(DATA_CONFIG),
        epochs=50,
        imgsz=640,
        batch=8,
        device=device,
        project=str(ROOT / "yolov8" / "runs"),
        name="baseball_yolov8n",
        patience=10,
    )

    print(
        "\n訓練完成後，最佳權重會在：\n"
        "  yolov8/runs/baseball_yolov8n/weights/best.pt\n"
        "之後可用於推論與整合現有的軌跡/overlay pipeline。"
    )


if __name__ == "__main__":
    main()

