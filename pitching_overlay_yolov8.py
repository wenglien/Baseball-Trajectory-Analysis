import logging
import os
import sys
import warnings
from optparse import OptionParser
from typing import Optional

from ultralytics import YOLO

from src.get_pitch_frames_yolov8 import get_pitch_frames_yolov8
from src.generate_overlay import generate_overlay
from src.ball_speed_calculator import BallSpeedCalculator
from src.release_point_detector import ReleasePointDetector

if not sys.warnoptions:
    warnings.simplefilter("ignore")


def run_yolov8_pipeline(
    video_paths: list[str],
    weights_path: str,
    *,
    conf: float = 0.1,
    output_path: str,
    show_preview: bool = False,
    enable_speed_calculation: bool = True,
    enable_field_calibration: bool = True,
    manual_distance_meters: Optional[float] = None,
) -> None:
    """
    將pitching_overlay_yolov8.py 流程封裝成函式，方便GUI呼叫。

    Args:
        video_paths: 要處理的投球影片路徑清單
        weights_path: YOLOv8 權重檔路徑 (.pt)
        conf: YOLOv8 閾值（建議 0.03~0.1，數值越低越容易偵測到小球）
        output_path: 輸出 overlay 影片路徑
        show_preview: 是否顯示預覽畫面
        enable_speed_calculation: 是否啟用球速計算
        enable_field_calibration: （已廢止/保留相容）過去用於互動式場地校正（點選參考點）
        manual_distance_meters: 手動輸入的投手到捕手距離（米）。若未指定，預設使用 18.44m
    
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

    speed_calculator = None
    if enable_speed_calculation:
        import cv2
        cap = cv2.VideoCapture(video_paths[0])
        if cap.isOpened():
            video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            ret, first_frame = cap.read()
            cap.release()
            
            if ret:
                pitch_distance_m = manual_distance_meters if manual_distance_meters is not None else 18.44
                print("\n" + "="*60)
                print("球速計算：使用手動輸入距離（不再需要點選校正）")
                print(f"投手到捕手距離：{pitch_distance_m:.2f} 公尺")
                print("="*60 + "\n")
                speed_calculator = BallSpeedCalculator(
                    fps=video_fps,
                    video_width=video_width,
                    video_height=video_height,
                    theoretical_distance=pitch_distance_m,
                )

    pitch_frames = []
    width = height = fps = None
    all_speed_info = []

    for idx, video_path in enumerate(video_paths):
        print(f"Processing Video {idx + 1}: {video_path}")
        if not os.path.isfile(video_path):
            print(f"Warning: video file not found, skip: {video_path}")
            continue

        try:
            ball_frames, width, height, fps, speed_info = get_pitch_frames_yolov8(
                video_path,
                yolo_model,
                conf_threshold=conf,
                show_preview=show_preview,
                speed_calculator=speed_calculator,
            )
            if ball_frames and len(ball_frames) > 0:
                pitch_frames.append(ball_frames)
                if speed_info:
                    all_speed_info.append(speed_info)
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
            continue

    valid_pitch_frames = [pf for pf in pitch_frames if pf and len(pf) > 0]

    if valid_pitch_frames and width is not None and height is not None and fps is not None:
        speed_info_for_overlay = all_speed_info[0] if all_speed_info else None
        
        generate_overlay(
            valid_pitch_frames,
            width,
            height,
            fps,
            output_path,
            show_preview=show_preview,
            speed_info=speed_info_for_overlay,
        )
        print(f"Overlay 影片已輸出到：{output_path}")
    else:
        print(
            "沒有任何影片偵測到足夠的球軌跡，因此不會產生 Overlay 影片，避免輸出損壞或空白檔案。"
        )


def _parse_cli_args():
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
        help="YOLOv8 confidence threshold (default: 0.15, 建議 0.1~0.3 間調整)",
        default=0.15,
    )
    optparser.add_option(
        "--no-speed",
        dest="no_speed",
        action="store_true",
        help="停用球速計算功能",
        default=False,
    )
    optparser.add_option(
        "--no-calibration",
        dest="no_calibration",
        action="store_true",
        help="（已廢止）過去用於跳過點選校正；目前已移除點選校正，此參數保留相容",
        default=False,
    )
    optparser.add_option(
        "-d",
        "--distance",
        dest="distance",
        type="float",
        help="手動輸入投手到捕手距離（公尺），例如：18.44 或 15",
        default=None,
    )
    return optparser.parse_args()


if __name__ == "__main__":
    options, args = _parse_cli_args()
    if options.no_calibration:
        print("⚠ 已移除點選校正（Field Calibration）；`--no-calibration` 參數目前不影響結果，保留相容用。")

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
        enable_speed_calculation=not options.no_speed,
        enable_field_calibration=not options.no_calibration,
        manual_distance_meters=options.distance,
    )

