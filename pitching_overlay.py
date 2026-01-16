import logging
import os
import sys
import warnings
from optparse import OptionParser

import tensorflow as tf
from tensorflow.python.saved_model import tag_constants

from src.get_pitch_frames import get_pitch_frames
from src.generate_overlay import generate_overlay

# Ignore warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
tf.get_logger().setLevel(logging.ERROR)


def _enable_gpu_memory_growth() -> None:
    """
    啟用 TensorFlow GPU 記憶體漸進分配（若有 GPU）。
    這段會在 CLI 與其他呼叫端共用，避免重複程式碼。
    """
    try:
        physical_devices = tf.config.experimental.list_physical_devices("GPU")
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except Exception:
        # 在 CPU-only 或某些環境下可能失敗，屬可忽略錯誤
        pass


def run_yolov4_pipeline(
    video_paths: list[str],
    output_path: str,
    *,
    show_preview: bool = True,
    weights_dir: str | None = None,
) -> None:
    """
    將原本 pitching_overlay.py 的主流程封裝成可重複呼叫的函式。

    - video_paths: 要處理的投球影片路徑清單（1 支或多支）
    - output_path: 輸出 overlay 影片路徑
    - show_preview: 是否顯示 OpenCV 預覽視窗
    - weights_dir: YOLOv4-tiny 模型目錄（預設為 model/yolov4-tiny-baseball-416）
    """
    if not video_paths:
        raise ValueError("video_paths 不可為空，至少需要一支投球影片。")

    _enable_gpu_memory_growth()

    # Initialize variables
    size = 416
    iou = 0.45
    score = 0.5
    weights = weights_dir or os.path.join("model", "yolov4-tiny-baseball-416")

    if not os.path.isdir(weights):
        raise FileNotFoundError(
            f"找不到 YOLOv4-tiny 模型目錄：{weights}\n"
            "請確認 model/yolov4-tiny-baseball-416 是否存在。"
        )

    # Load pretrained model
    saved_model_loaded = tf.saved_model.load(weights, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures["serving_default"]

    # Store the pitch frames and ball location of each video
    pitch_frames: list = []
    width = height = fps = None

    # Iterate all videos
    for idx, video_path in enumerate(video_paths):
        print(f"Processing Video {idx + 1}: {video_path}")
        if not os.path.isfile(video_path):
            print(f"Warning: video file not found, skip: {video_path}")
            continue

        try:
            ball_frames, width, height, fps = get_pitch_frames(
                video_path, infer, size, iou, score, show_preview=show_preview
            )
            if ball_frames and len(ball_frames) > 0:
                pitch_frames.append(ball_frames)
            else:
                print(
                    f"Warning: 視訊 {os.path.basename(video_path)} "
                    "中沒有偵測到足夠的球軌跡，將略過此影片的 overlay。"
                )
        except Exception as e:
            print(
                f"Error: Sorry we could not get enough baseball detection from the "
                f"video, video {os.path.basename(video_path)} will not be overlayed"
            )
            print(e)

    if pitch_frames and width is not None and height is not None and fps is not None:
        generate_overlay(
            pitch_frames, width, height, fps, output_path, show_preview=show_preview
        )
        print(f"Overlay 影片已輸出到：{output_path}")
    else:
        print(
            "沒有任何影片偵測到足夠的球軌跡，因此不會產生 Overlay 影片，"
            "避免輸出損壞或空白檔案。"
        )


def _parse_cli_args():
    """CLI 模式下使用的參數解析。"""
    optparser = OptionParser()
    optparser.add_option(
        "-f",
        "--videos_folder",
        dest="rootDir",
        help="Root directory that contains your pitching videos",
        default=os.path.join("videos", "videos1"),
    )
    optparser.add_option(
        "-v",
        "--video_file",
        dest="videoFile",
        help="Single video file to analyze (overrides --videos_folder)",
        default=None,
    )
    return optparser.parse_args()


if __name__ == "__main__":
    options, args = _parse_cli_args()

    # Decide input videos and output path
    video_paths: list[str] = []
    if options.videoFile:
        video_file = options.videoFile
        if not os.path.isfile(video_file):
            print(f"Error: video file not found: {video_file}")
            sys.exit(1)
        video_paths = [video_file]
        output_path = os.path.join(os.path.dirname(video_file), "Overlay.mp4")
    else:
        rootDir = options.rootDir
        if not os.path.isdir(rootDir):
            print(f"Error: videos folder not found: {rootDir}")
            sys.exit(1)
        output_path = os.path.join(rootDir, "Overlay.mp4")
        for path in os.listdir(rootDir):
            if path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                video_paths.append(os.path.join(rootDir, path))

    if not video_paths:
        print("No video files found to process.")
        sys.exit(0)

    run_yolov4_pipeline(video_paths, output_path, show_preview=True)
