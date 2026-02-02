"""
將現有的 YOLOv4-tiny 棒球模型 (TensorFlow SavedModel)
轉成 CoreML 模型，供 iOS App 使用。

- YOLOv4-tiny 的 TensorFlow 結構與自定義後處理較複雜，
  此腳本提供一個參考流程，實際上需要根據
  自己訓練時的輸入尺寸 / 輸出節點名稱做調整。
"""

import os
from pathlib import Path

try:
    import tensorflow as tf  # type: ignore
except ModuleNotFoundError as e:
    raise RuntimeError(
        "缺少 TensorFlow：此腳本為 YOLOv4 legacy 專用。\n"
        "請先安裝：python3 -m pip install -r requirements-yolov4.txt"
    ) from e

try:
    import coremltools as ct  # type: ignore
except ModuleNotFoundError as e:
    raise RuntimeError(
        "缺少 coremltools：此腳本用於輸出 .mlmodel（通常只在 macOS 需要）。\n"
        "請先安裝：python3 -m pip install coremltools"
    ) from e


ROOT = Path(__file__).resolve().parents[2]
SAVED_MODEL_DIR = ROOT / "model" / "yolov4-tiny-baseball-416"
OUTPUT_MLMODEL_PATH = ROOT / "model" / "yolov4_tiny_baseball_416.mlmodel"


def load_saved_model():
    print(f"Loading SavedModel from: {SAVED_MODEL_DIR}")
    model = tf.saved_model.load(str(SAVED_MODEL_DIR))
    infer = model.signatures.get("serving_default")
    if infer is None:
        raise RuntimeError("Cannot find 'serving_default' signature in SavedModel.")
    return infer


def convert_to_coreml(concrete_func: tf.types.experimental.ConcreteFunction):
    """
    將 TensorFlow ConcreteFunction 轉成 CoreML。
    這裡假設輸入為一個 416x416x3 的 float32 影像張量；
    若你原本訓練時使用其他尺寸，依實際情況調整。
    """
    print("Converting to CoreML...")

    input_keys = list(concrete_func.inputs)
    output_keys = list(concrete_func.outputs)
    print("TF input tensors:", input_keys)
    print("TF output tensors:", output_keys)

    mlmodel = ct.convert(
        concrete_func,
        source="tensorflow",
        inputs=[
            ct.TensorType(
                name=input_keys[0].name.split(":")[0],
                shape=concrete_func.inputs[0].shape,
                dtype=concrete_func.inputs[0].dtype.as_numpy_dtype,
            )
        ],
    )

    print(f"Saving CoreML model to: {OUTPUT_MLMODEL_PATH}")
    mlmodel.save(str(OUTPUT_MLMODEL_PATH))
    print("Done.")


def main():
    if not SAVED_MODEL_DIR.exists():
        raise FileNotFoundError(f"SavedModel directory not found: {SAVED_MODEL_DIR}")

    infer = load_saved_model()
    concrete_func = infer

    convert_to_coreml(concrete_func)


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    main()

