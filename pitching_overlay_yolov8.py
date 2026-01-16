import logging
import os
import sys
import warnings
from optparse import OptionParser

from ultralytics import YOLO

from src.get_pitch_frames_yolov8 import get_pitch_frames_yolov8
from src.generate_overlay import generate_overlay

# Ignore warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")


def run_yolov8_pipeline(
    video_paths: list[str],
    weights_path: str,
    *,
    conf: float = 0.1,
    output_path: str,
    show_preview: bool = False,
) -> None:
    """
    將原本 pitching_overlay_yolov8.py 的流程封裝成函式，方便 GUI 或其他程式呼叫。

    Args:
        video_paths: 要處理的投球影片路徑清單
        weights_path: YOLOv8 權重檔路徑 (.pt)
        conf: YOLOv8 置信度閾值（建議 0.03~0.1，數值越低越容易偵測到小球）
        output_path: 輸出 overlay 影片路徑
        show_preview: 是否顯示預覽畫面
    
    Raises:
        ValueError: 當 video_paths 為空時
        FileNotFoundError: 當找不到權重檔案時
        RuntimeError: 當處理影片時發生錯誤
    """
    if not video_paths:
        raise ValueError("video_paths 不可為空，至少需要一支投球影片。")

    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"找不到 YOLOv8 權重檔案：{weights_path}\n"
            "請先依照 yolov8/train_yolov8.py 的說明完成訓練。"
        )

    print(f"Loading YOLOv8 model from: {weights_path}")
    yolo_model = YOLO(weights_path)

    pitch_frames = []
    width = height = fps = None

    for idx, video_path in enumerate(video_paths):
        print(f"Processing Video {idx + 1}: {video_path}")
        if not os.path.isfile(video_path):
            print(f"Warning: video file not found, skip: {video_path}")
            continue

        try:
            # Use the adjustable conf_threshold, so that the trained YOLOv8 is easier to detect the ball
            ball_frames, width, height, fps = get_pitch_frames_yolov8(
                video_path,
                yolo_model,
                conf_threshold=conf,
                show_preview=show_preview,
            )
            if ball_frames and len(ball_frames) > 0:
                pitch_frames.append(ball_frames)
            else:
                print(
                    f"Warning: 視訊 {os.path.basename(video_path)} 中沒有偵測到足夠的球軌跡，將略過此影片的 overlay。"
                )
        except Exception as e:
            error_msg = (
                f"Error: Sorry we could not get enough baseball detection from the video, "
                f"video {os.path.basename(video_path)} will not be overlayed"
            )
            print(error_msg)
            print(f"詳細錯誤：{e}")
            # Log the error but continue processing other videos
            continue

    # Only generate overlay when at least one video detects enough ball trajectories
    valid_pitch_frames = [pf for pf in pitch_frames if pf and len(pf) > 0]

    if valid_pitch_frames and width is not None and height is not None and fps is not None:
        generate_overlay(
            valid_pitch_frames,
            width,
            height,
            fps,
            output_path,
            show_preview=show_preview,
        )
        print(f"Overlay 影片已輸出到：{output_path}")
    else:
        print(
            "沒有任何影片偵測到足夠的球軌跡，因此不會產生 Overlay 影片，避免輸出損壞或空白檔案。"
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
    optparser.add_option(
        "-w",
        "--weights",
        dest="weights",
        help="Path to YOLOv8 weights (.pt), e.g. yolov8/runs/baseball_yolov8n/weights/best.pt",
        default=os.path.join(
            "yolov8", "runs", "baseball_yolov8n2", "weights", "best.pt"
        ),
    )
    optparser.add_option(
        "-c",
        "--conf",
        dest="conf",
        type="float",
        help="YOLOv8 confidence threshold (default: 0.1, 建議 0.05~0.25 間調整)",
        default=0.1,
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
        output_path = os.path.join(os.path.dirname(video_file), "Overlay_yolov8.mp4")
    else:
        rootDir = options.rootDir
        if not os.path.isdir(rootDir):
            print(f"Error: videos folder not found: {rootDir}")
            sys.exit(1)
        output_path = os.path.join(rootDir, "Overlay_yolov8.mp4")
        for path in os.listdir(rootDir):
            if path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                video_paths.append(os.path.join(rootDir, path))

    if not video_paths:
        print("No video files found to process.")
        sys.exit(0)

    run_yolov8_pipeline(
        video_paths,
        weights_path=options.weights,
        conf=options.conf,
        output_path=output_path,
        show_preview=False,
    )

