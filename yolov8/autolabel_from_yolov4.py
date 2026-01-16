"""
利用現有的 YOLOv4-tiny 棒球模型，自動幫 /videos 底下的影片做標註，
產生 YOLOv8 可用的訓練資料（images + labels）。

流程：
1. 從 /videos 所有影片擷取影格。
2. 用 YOLOv4-tiny 偵測球的位置。
3. 將有偵測到球的影格儲存為 jpg，並在 labels/train 產生對應的 txt（class 0: ball）。
4. 之後可以直接執行 yolov8/train_yolov8.py 進行微調。
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants

# 確保可以匯入專案根目錄底下的 src 模組
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.get_pitch_frames import detect  # 重用原本的偵測邏輯


ROOT = Path(__file__).resolve().parents[1]
VIDEOS_DIR = ROOT / "videos"

DATASET_ROOT = ROOT / "yolov8" / "datasets" / "baseball"
IMAGES_TRAIN = DATASET_ROOT / "images" / "train"
LABELS_TRAIN = DATASET_ROOT / "labels" / "train"


def ensure_dirs():
    for d in [IMAGES_TRAIN, LABELS_TRAIN]:
        d.mkdir(parents=True, exist_ok=True)


def load_yolov4_model():
    weights = ROOT / "model" / "yolov4-tiny-baseball-416"
    if not weights.exists():
        raise FileNotFoundError(
            f"找不到 YOLOv4 模型目錄：{weights}\n請確認 model/yolov4-tiny-baseball-416 是否存在。"
        )
    print(f"Loading YOLOv4-tiny model from: {weights}")
    saved_model_loaded = tf.saved_model.load(str(weights), tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures["serving_default"]
    return infer


def process_video(video_path: Path, infer, input_size: int = 416):
    print(f"\n[Video] {video_path}")
    vid = cv2.VideoCapture(str(video_path))
    if not vid.isOpened():
        print("  無法開啟影片，略過。")
        return

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        print("  讀不到影片尺寸，略過。")
        vid.release()
        return

    iou = 0.45
    score_threshold = 0.5

    frame_idx = 0
    saved_count = 0

    while True:
        ok, frame = vid.read()
        if not ok:
            break

        # 只取每 2 幀一張，避免資料量過大
        if frame_idx % 2 != 0:
            frame_idx += 1
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detected_balls = []
        detections = detect(
            infer,
            frame_rgb,
            input_size,
            iou,
            score_threshold,
            detected_balls,
        )

        if len(detections) == 0:
            frame_idx += 1
            continue

        # 目前只標第一個偵測到的球
        det = detections[0]
        x1, y1, x2, y2, score = det.tolist()

        # 轉成 YOLO 格式 (cx, cy, w, h) / 相對座標
        cx = ((x1 + x2) / 2.0) / width
        cy = ((y1 + y2) / 2.0) / height
        w = (x2 - x1) / width
        h = (y2 - y1) / height

        # 基本邊界檢查
        if w <= 0 or h <= 0:
            frame_idx += 1
            continue

        img_name = f"{video_path.stem}_{frame_idx:06d}.jpg"
        label_name = f"{video_path.stem}_{frame_idx:06d}.txt"
        img_path = IMAGES_TRAIN / img_name
        label_path = LABELS_TRAIN / label_name

        cv2.imwrite(str(img_path), frame)
        with open(label_path, "w", encoding="utf-8") as f:
            # 類別 0: ball
            f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        saved_count += 1
        frame_idx += 1

    vid.release()
    print(f"  產生 {saved_count} 筆自動標註樣本。")


def main():
    ensure_dirs()
    infer = load_yolov4_model()

    # 逐一處理 videos/ 底下所有 mp4/avi/mov/mkv
    if not VIDEOS_DIR.exists():
        print(f"找不到資料夾：{VIDEOS_DIR}")
        return

    for sub in [VIDEOS_DIR] + [p for p in VIDEOS_DIR.iterdir() if p.is_dir()]:
        for file in sub.glob("**/*"):
            if file.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
                process_video(file, infer)

    print(
        "\n自動標註完成！\n"
        f"影像輸出在：{IMAGES_TRAIN}\n"
        f"標註輸出在：{LABELS_TRAIN}\n"
        "你可以直接執行 `python3 yolov8/train_yolov8.py` 進行 YOLOv8 微調，\n"
        "之後再用 pitching_overlay_yolov8.py 來產生含有球軌跡的 overlay。"
    )


if __name__ == "__main__":
    main()

