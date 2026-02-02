import logging
import os
import sys
import warnings
from optparse import OptionParser

from typing import Optional

from src.pipelines.yolov8_pipeline import run_yolov8_pipeline as _run_yolov8_pipeline
from src.generate_overlay import generate_overlay
from src.logging_utils import configure_logging, get_logger

# Ignore warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")


def _enable_gpu_memory_growth() -> None:
    """
    啟用 TensorFlow GPU 記憶體漸進分配（若有 GPU）。
    這段會在 CLI 與其他呼叫端共用，避免重複程式碼。
    """
    try:
        import tensorflow as tf  # noqa: F401

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

    # 只在 YOLOv4 模式才需要 TensorFlow
    try:
        import tensorflow as tf  # type: ignore
        from tensorflow.python.saved_model import tag_constants  # type: ignore
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "你啟用了 YOLOv4（legacy）流程，但目前環境未安裝 TensorFlow。\n"
            "請先安裝 legacy 依賴：\n"
            "  python3 -m pip install -r requirements-yolov4.txt\n"
            "（平常使用 Ultralytics YOLO：YOLO11/YOLOv8 不需要 TensorFlow）"
        ) from e

    # 延遲載入 YOLOv4 實作（避免一般使用者不小心 import 到 TF 依賴）
    from src.get_pitch_frames import get_pitch_frames

    tf.get_logger().setLevel(logging.ERROR)
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


def run_yolov8_overlay(
    video_paths: list[str],
    output_path: str,
    *,
    weights_path: str = os.path.join(
        "yolov8", "runs", "baseball_yolo11n", "weights", "best.pt"
    ),
    conf: float = 0.1,
    show_preview: bool = False,
    manual_distance_meters: Optional[float] = None,
    enable_speed_calculation: bool = True,
    debug: bool = False,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    以 Ultralytics YOLO 產生 overlay（YOLO11 / YOLOv8 均可；此為專案目前建議的預設流程）。

    注意：舊的 YOLOv4 流程仍保留在 run_yolov4_pipeline（legacy）。
    """
    _run_yolov8_pipeline(
        video_paths,
        weights_path=weights_path,
        conf=conf,
        output_path=output_path,
        show_preview=show_preview,
        enable_speed_calculation=enable_speed_calculation,
        enable_field_calibration=False,
        manual_distance_meters=manual_distance_meters,
        debug=debug,
        logger=logger,
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
        help="Path to Ultralytics YOLO weights (.pt) (YOLO11/YOLOv8)",
        default=os.path.join("yolov8", "runs", "baseball_yolo11n", "weights", "best.pt"),
    )
    optparser.add_option(
        "-c",
        "--conf",
        dest="conf",
        type="float",
        help="YOLO confidence threshold (default: 0.1，建議 0.03~0.2)",
        default=0.1,
    )
    optparser.add_option(
        "-d",
        "--distance",
        dest="distance",
        type="float",
        help="手動輸入投手到捕手距離（公尺），例如：18.44 或 15",
        default=None,
    )
    optparser.add_option(
        "--no-speed",
        dest="no_speed",
        action="store_true",
        help="停用球速計算功能（Ultralytics YOLO）",
        default=False,
    )
    optparser.add_option(
        "--legacy-yolov4",
        dest="legacy_yolov4",
        action="store_true",
        help="使用舊的 YOLOv4/TensorFlow 流程（不建議；軌跡穩定度較差）",
        default=False,
    )
    optparser.add_option(
        "--no-calibration",
        dest="no_calibration",
        action="store_true",
        help="（已廢止）保留相容用",
        default=False,
    )
    optparser.add_option(
        "--debug",
        dest="debug",
        action="store_true",
        help="顯示完整 traceback（除錯用）",
        default=False,
    )
    optparser.add_option(
        "--quiet",
        dest="quiet",
        action="store_true",
        help="只輸出錯誤訊息（ERROR）",
        default=False,
    )
    optparser.add_option(
        "-V",
        "--verbose",
        dest="verbose",
        action="count",
        help="輸出更多除錯資訊（可重複，例如 -VV）",
        default=0,
    )
    return optparser.parse_args()


def cli_main() -> None:
    """單一 CLI 入口。"""
    options, args = _parse_cli_args()
    configure_logging(verbose=int(options.verbose or 0), quiet=bool(options.quiet))
    log = get_logger("pitching_overlay")

    if getattr(options, "no_calibration", False):
        log.warning("已移除點選校正；`--no-calibration` 參數保留相容用。")

    video_paths: list[str] = []
    if options.videoFile:
        video_file = options.videoFile
        if not os.path.isfile(video_file):
            log.error("video file not found: %s", video_file)
            sys.exit(1)
        video_paths = [video_file]
        output_path = os.path.join(os.path.dirname(video_file), "Overlay.mp4")
    else:
        rootDir = options.rootDir
        if not os.path.isdir(rootDir):
            log.error("videos folder not found: %s", rootDir)
            sys.exit(1)
        output_path = os.path.join(rootDir, "Overlay.mp4")
        for path in os.listdir(rootDir):
            if path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                video_paths.append(os.path.join(rootDir, path))

    if not video_paths:
        log.warning("No video files found to process.")
        sys.exit(0)

    if options.legacy_yolov4:
        try:
            run_yolov4_pipeline(video_paths, output_path, show_preview=True)
        except RuntimeError as e:
            if options.debug:
                log.exception("YOLOv4 legacy 處理失敗")
            else:
                log.error("%s", str(e))
            sys.exit(1)
    else:
        try:
            run_yolov8_overlay(
                video_paths,
                output_path,
                weights_path=options.weights,
                conf=float(options.conf),
                show_preview=True,
                manual_distance_meters=options.distance,
                enable_speed_calculation=not options.no_speed,
                debug=bool(options.debug),
                logger=log,
            )
        except Exception as e:
            if options.debug:
                log.exception("Ultralytics YOLO 處理失敗")
            else:
                log.error("Ultralytics YOLO 處理失敗：%s", str(e))
            sys.exit(1)


if __name__ == "__main__":
    cli_main()
