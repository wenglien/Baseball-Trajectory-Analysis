"""
Legacy shim for YOLOv4 CoreML conversion.

實作已移到：`legacy/yolov4/convert_yolov4_tiny_to_coreml.py`
此檔案保留舊路徑 `model/convert_yolov4_tiny_to_coreml.py` 以維持相容性。
"""

import os


def main():
    from legacy.yolov4.convert_yolov4_tiny_to_coreml import main as _main

    _main()


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    main()

