"""
資料集需為 YOLO 格式，且路徑需符合 yolov8/data_baseball.yaml：
  yolov8/datasets/baseball/
    images/train/*.jpg
    labels/train/*.txt

用法：
  cd yolov8
  python train_yolo11.py

透過環境變數調參：
  YOLO_MODEL=yolo11s.pt EPOCHS=100 IMGSZ=960 BATCH=8 python train_yolo11.py
  # 若要避免與既有 runs 名稱衝突，可自訂 RUN_NAME
  RUN_NAME=baseball_yolo11n_v2 python train_yolo11.py
"""

from __future__ import annotations

from pathlib import Path
import os

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA_CONFIG = ROOT / "yolov8" / "data_baseball.yaml"


def get_device() -> str:
    # 在 Apple Silicon 上優先使用 mps；其次 cuda；最後 cpu
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "0"
    return "cpu"


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if not v:
        return default
    try:
        return int(v)
    except Exception:
        return default


def _env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return v.strip() if v and v.strip() else default


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip().lower()
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off", ""):
        return False
    return default


def main() -> None:
    device = get_device()
    print(f"Using device: {device}")

    model_name = _env_str("YOLO_MODEL", "yolo11n.pt")
    epochs = _env_int("EPOCHS", 80)
    imgsz = _env_int("IMGSZ", 640)
    batch = _env_int("BATCH", 8)
    exist_ok = _env_bool("EXIST_OK", default=False)
    resume = _env_bool("RESUME", default=False)

    # 以 YOLO11 預訓練權重為基礎做微調
    model = YOLO(model_name)

    run_name_default = f"baseball_{Path(model_name).stem}"
    run_name = _env_str("RUN_NAME", run_name_default)
    model.train(
        data=str(DATA_CONFIG),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(ROOT / "yolov8" / "runs"),
        name=run_name,
        patience=15,
        exist_ok=exist_ok,
        resume=resume,
    )

    print(
        "\n訓練完成後，最佳權重會在：\n"
        f"  yolov8/runs/{run_name}/weights/best.pt\n"
        "之後可用於 pitching_overlay.py / GUI 進行推論。"
    )


if __name__ == "__main__":
    main()

