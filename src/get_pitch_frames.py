"""
Legacy shim for YOLOv4/TensorFlow pipeline.

此檔案保留舊匯入路徑 `src.get_pitch_frames` 以維持相容性，但實作已移到：
`legacy/yolov4/get_pitch_frames.py`

重點：此 shim 採「延遲載入」，讓一般只用 Ultralytics YOLO（YOLO11/YOLOv8）
的使用者在未安裝 TensorFlow 時，import 專案也不會失敗。
"""

from __future__ import annotations

from typing import Any, Callable


def _legacy_import():
    try:
        from legacy.yolov4 import get_pitch_frames as impl  # type: ignore
        return impl
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "你正在使用 YOLOv4（legacy）功能，但目前環境缺少必要套件。\n"
            "請先安裝：python3 -m pip install -r requirements-yolov4.txt\n"
            "（平常使用 YOLO11/YOLOv8 不需要 TensorFlow）"
        ) from e


def get_pitch_frames(*args: Any, **kwargs: Any):
    return _legacy_import().get_pitch_frames(*args, **kwargs)


def detect(*args: Any, **kwargs: Any):
    return _legacy_import().detect(*args, **kwargs)


def add_balls_before_SORT(*args: Any, **kwargs: Any):
    return _legacy_import().add_balls_before_SORT(*args, **kwargs)


def add_lost_frames(*args: Any, **kwargs: Any):
    return _legacy_import().add_lost_frames(*args, **kwargs)
