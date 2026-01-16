"""
從現有影片中抽幀，建立 YOLOv8 可用的資料集骨架。

步驟概要：
1. 把你要用來訓練的影片放到 videos/ 底下，或直接使用 pitcher.mp4。
2. 執行本腳本，會在 yolov8/datasets/baseball/images/train 產生一堆 jpg。
3. 使用標註工具（例如 labelImg、Roboflow、CVAT 等）打開這些 jpg，
   手動畫出「球」的 bounding box，並匯出 YOLO txt 標註到同一路徑。
4. 至少準備數百張有球的標註影像，再跑 train_yolov8.py 做微調。
"""

import os
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
VIDEOS_DIR = ROOT / "videos"
SINGLE_VIDEO = ROOT / "pitcher.mp4"  # 你目前正在用的影片

DATASET_ROOT = ROOT / "yolov8" / "datasets" / "baseball"
IMAGES_TRAIN = DATASET_ROOT / "images" / "train"
IMAGES_VAL = DATASET_ROOT / "images" / "val"


def ensure_dirs():
    for d in [IMAGES_TRAIN, IMAGES_VAL]:
        d.mkdir(parents=True, exist_ok=True)


def extract_frames_from_video(video_path: Path, out_dir: Path, stride: int = 2):
    """
    從影片每隔 stride 幀擷取一張影像，輸出為 jpg。
    建議前期先多抽一點，之後用標註工具挑可用的幀。
    """
    print(f"Extracting frames from: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  無法開啟影片：{video_path}")
        return

    idx = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            out_path = out_dir / f"{video_path.stem}_{idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1
        idx += 1

    cap.release()
    print(f"  已輸出 {saved} 張影像到 {out_dir}")


def main():
    ensure_dirs()

    # 1) 先處理根目錄的 pitcher.mp4（如果存在）
    if SINGLE_VIDEO.exists():
        extract_frames_from_video(SINGLE_VIDEO, IMAGES_TRAIN, stride=1)

    # 2) 處理 videos/ 下的所有 mp4/avi/mov/mkv
    if VIDEOS_DIR.exists():
        for sub in [VIDEOS_DIR] + [p for p in VIDEOS_DIR.iterdir() if p.is_dir()]:
            for file in sub.glob("**/*"):
                if file.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
                    extract_frames_from_video(file, IMAGES_TRAIN, stride=3)

    print(
        "\n資料集已建立在 yolov8/datasets/baseball/images/train。\n"
        "請用標註工具打開這些 jpg，將『球』標成 YOLO 格式的 txt，\n"
        "之後再執行 train_yolov8.py 進行微調。"
    )


if __name__ == "__main__":
    main()

