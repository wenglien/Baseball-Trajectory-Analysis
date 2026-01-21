import logging
import os
import sys
import warnings
from optparse import OptionParser
from typing import Optional

from ultralytics import YOLO

from src.get_pitch_frames_yolov8 import get_pitch_frames_yolov8
from src.generate_overlay import generate_overlay
from src.ball_speed_calculator import BallSpeedCalculator, FieldCalibrationTool
from src.release_point_detector import ReleasePointDetector

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
    enable_speed_calculation: bool = True,
    enable_field_calibration: bool = True,
    manual_distance_meters: Optional[float] = None,
) -> None:
    """
    將原本 pitching_overlay_yolov8.py 的流程封裝成函式，方便 GUI 或其他程式呼叫。

    Args:
        video_paths: 要處理的投球影片路徑清單
        weights_path: YOLOv8 權重檔路徑 (.pt)
        conf: YOLOv8 置信度閾值（建議 0.03~0.1，數值越低越容易偵測到小球）
        output_path: 輸出 overlay 影片路徑
        show_preview: 是否顯示預覽畫面
        enable_speed_calculation: 是否啟用球速計算
        enable_field_calibration: 是否啟用場地校正（需要手動標記）
        manual_distance_meters: 手動輸入的場地距離（米），若指定則跳過互動校正
    
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

    # 初始化球速計算器
    speed_calculator = None
    if enable_speed_calculation:
        # 先讀取第一支影片的基本資訊
        import cv2
        cap = cv2.VideoCapture(video_paths[0])
        if cap.isOpened():
            video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            # 讀取第一幀用於校正
            ret, first_frame = cap.read()
            cap.release()
            
            if ret and enable_field_calibration:
                # 檢查是否使用手動輸入距離
                if manual_distance_meters is not None:
                    # 使用手動輸入的距離進行改進的校正
                    print("\n" + "="*60)
                    print(f"使用手動輸入距離進行校正：{manual_distance_meters} 公尺")
                    print("="*60)
                    
                    # 改進的校正邏輯：
                    # 假設畫面覆蓋的實際場地範圍約為輸入距離的 70%
                    # 這是基於典型棒球影片的拍攝角度估算
                    coverage_ratio = 0.7
                    estimated_field_coverage = manual_distance_meters * coverage_ratio
                    estimated_pixels_per_meter = video_height / estimated_field_coverage
                    
                    speed_calculator = BallSpeedCalculator(
                        fps=video_fps,
                        video_width=video_width,
                        video_height=video_height,
                        pixels_per_meter=estimated_pixels_per_meter,
                        theoretical_distance=manual_distance_meters  # 記錄理論距離
                    )
                    print(f"✓ 球速計算功能已啟用")
                    print(f"  估計畫面覆蓋範圍: {estimated_field_coverage:.1f} 公尺")
                    print(f"  估計比例: {estimated_pixels_per_meter:.1f} pixels/meter\n")
                else:
                    # 啟動互動式場地校正工具
                    print("\n" + "="*60)
                    print("啟動場地校正工具...")
                    print("="*60)
                    
                    calib_tool = FieldCalibrationTool(first_frame)
                    ref_points = calib_tool.mark_reference_points()
                    
                    if ref_points:
                        # 創建球速計算器並校正
                        speed_calculator = BallSpeedCalculator(
                            fps=video_fps,
                            video_width=video_width,
                            video_height=video_height
                        )
                        speed_calculator.calibrate_from_reference(
                            pitcher_mound_pixel=ref_points[0],
                            home_plate_pixel=ref_points[1],
                            real_distance=18.44  # 標準棒球場距離
                        )
                        print("✓ 球速計算功能已啟用\n")
                    else:
                        print("⚠ 校正已取消，將不計算球速")
                        enable_speed_calculation = False
            elif ret and not enable_field_calibration:
                # 不啟用校正，使用預設值
                print("⚠ 未啟用場地校正，球速計算可能不準確")
                speed_calculator = BallSpeedCalculator(
                    fps=video_fps,
                    video_width=video_width,
                    video_height=video_height,
                    pixels_per_meter=35.0  # 預設值，需要根據實際情況調整
                )

    pitch_frames = []
    width = height = fps = None
    all_speed_info = []  # 收集所有影片的球速資訊

    for idx, video_path in enumerate(video_paths):
        print(f"Processing Video {idx + 1}: {video_path}")
        if not os.path.isfile(video_path):
            print(f"Warning: video file not found, skip: {video_path}")
            continue

        try:
            # Use the adjustable conf_threshold, so that the trained YOLOv8 is easier to detect the ball
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
            # Log the error but continue processing other videos
            continue

    # Only generate overlay when at least one video detects enough ball trajectories
    valid_pitch_frames = [pf for pf in pitch_frames if pf and len(pf) > 0]

    if valid_pitch_frames and width is not None and height is not None and fps is not None:
        # 使用第一支影片的球速資訊（如果有多支影片）
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
        help="跳過場地校正（不會彈出標記視窗）",
        default=False,
    )
    optparser.add_option(
        "-d",
        "--distance",
        dest="distance",
        type="float",
        help="手動輸入場地距離（公尺），例如：18.44 或 15。若指定則跳過互動校正",
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

