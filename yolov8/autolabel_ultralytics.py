"""
使用現有的 Ultralytics YOLO（YOLO11/YOLOv8）權重，對指定影片做自動標註，
產生 YOLO 格式的訓練資料（images/train + labels/train），可作為新訓練資料。

改進版：
- 長寬比過濾：棒球應該接近正圓
- 尺寸範圍過濾：根據畫面大小動態調整
- NMS 去重：避免同一顆球被標註多次
- 信心度過濾：只保留高信心度的偵測
- 預覽模式：可視覺化驗證標註結果

用法：
  cd yolov8
  python autolabel_ultralytics.py ../videos/16_120fps_4k.mp4

  # 指定權重、抽幀間隔、置信度
  python autolabel_ultralytics.py ../videos/16_120fps_4k.mp4 -w runs/baseball_yolo11n/weights/best.pt --stride 2 --conf 0.15

  # 預覽模式（按空白鍵下一幀，Q 退出，S 儲存當前幀）
  python autolabel_ultralytics.py ../videos/16_120fps_4k.mp4 --preview
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# 腳本在 yolov8/autolabel_ultralytics.py，ROOT = 專案根目錄
ROOT = Path(__file__).resolve().parents[1]
YOLOV8_DIR = Path(__file__).resolve().parent
DATASET_ROOT = YOLOV8_DIR / "datasets" / "baseball"
IMAGES_TRAIN = DATASET_ROOT / "images" / "train"
LABELS_TRAIN = DATASET_ROOT / "labels" / "train"

DEFAULT_WEIGHTS = YOLOV8_DIR / "runs" / "baseball_yolo11n" / "weights" / "best.pt"

# ===== 棒球過濾參數 =====
# 尺寸範圍（相對於畫面寬高的比例）
MIN_SIZE = 0.003   # 最小 0.3%（避免噪點）
MAX_SIZE = 0.20    # 最大 20%（放寬以適應不同模型）

# 長寬比（棒球應該接近正圓，但運動模糊可能導致變形）
MIN_ASPECT_RATIO = 0.4   # 最小長寬比（放寬以適應運動模糊）
MAX_ASPECT_RATIO = 2.5   # 最大長寬比

# NMS 參數
NMS_IOU_THRESHOLD = 0.3  # IoU 超過此值視為重複

# 畫面位置過濾（可選：排除畫面邊緣的誤偵測）
EDGE_MARGIN = 0.005  # 距離邊緣 0.5% 內的偵測可能是誤報


def _infer_ball_class_ids(model: YOLO, first_result) -> set[int] | None:
    """推斷「球」的 class id；若僅單一類別則全接受。"""
    names = getattr(first_result, "names", None) or getattr(model, "names", None)
    if names is None:
        return None
    if isinstance(names, dict):
        name_map = {int(k): str(v) for k, v in names.items()}
    elif isinstance(names, (list, tuple)):
        name_map = {i: str(n) for i, n in enumerate(names)}
    else:
        return None
    if len(name_map) == 1:
        return set(name_map.keys())
    keywords = ("baseball", "ball")
    ball_ids = {cid for cid, name in name_map.items() if any(k in name.lower() for k in keywords)}
    return ball_ids or None


def compute_iou(box1: Tuple[float, float, float, float], 
                box2: Tuple[float, float, float, float]) -> float:
    """計算兩個 (cx, cy, w, h) 格式 bbox 的 IoU"""
    cx1, cy1, w1, h1 = box1
    cx2, cy2, w2, h2 = box2
    
    # 轉換為 (x1, y1, x2, y2) 格式
    x1_min, y1_min = cx1 - w1/2, cy1 - h1/2
    x1_max, y1_max = cx1 + w1/2, cy1 + h1/2
    x2_min, y2_min = cx2 - w2/2, cy2 - h2/2
    x2_max, y2_max = cx2 + w2/2, cy2 + h2/2
    
    # 計算交集
    inter_x1 = max(x1_min, x2_min)
    inter_y1 = max(y1_min, y2_min)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)
    
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def nms_filter(detections: List[Tuple[float, float, float, float, float]], 
               iou_threshold: float = NMS_IOU_THRESHOLD) -> List[Tuple[float, float, float, float]]:
    """
    對偵測結果做 NMS，只保留最高信心度的非重複偵測。
    輸入：[(cx, cy, w, h, score), ...]
    輸出：[(cx, cy, w, h), ...]
    """
    if not detections:
        return []
    
    # 按信心度排序（高到低）
    sorted_dets = sorted(detections, key=lambda x: x[4], reverse=True)
    
    kept = []
    while sorted_dets:
        best = sorted_dets.pop(0)
        best_box = (best[0], best[1], best[2], best[3])
        kept.append(best_box)
        
        # 過濾掉與最佳偵測重疊的其他偵測
        sorted_dets = [
            d for d in sorted_dets 
            if compute_iou(best_box, (d[0], d[1], d[2], d[3])) < iou_threshold
        ]
    
    return kept


def is_valid_baseball(cx: float, cy: float, w: float, h: float) -> bool:
    """檢查偵測是否符合棒球特徵"""
    # 尺寸檢查
    if w < MIN_SIZE or h < MIN_SIZE:
        return False
    if w > MAX_SIZE or h > MAX_SIZE:
        return False
    
    # 長寬比檢查（棒球應該接近正圓）
    aspect_ratio = w / h if h > 0 else 0
    if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
        return False
    
    # 邊緣過濾（可選）
    if cx < EDGE_MARGIN or cx > (1 - EDGE_MARGIN):
        return False
    if cy < EDGE_MARGIN or cy > (1 - EDGE_MARGIN):
        return False
    
    return True


def process_video(
    video_path: Path,
    model: YOLO,
    *,
    stride: int = 2,
    conf: float = 0.15,
    imgsz: int = 1280,
    preview: bool = False,
) -> int:
    """
    對單一影片逐幀用 YOLO 偵測球，將有偵測到的影格存成 jpg + YOLO 格式 txt（class 0: ball）。
    回傳寫入的樣本數。
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  無法開啟影片：{video_path}", file=sys.stderr)
        return 0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if width <= 0 or height <= 0:
        print("  讀不到影片尺寸，略過。", file=sys.stderr)
        cap.release()
        return 0

    saved_count = 0
    skipped_count = 0
    frame_idx = 0
    ball_class_ids: set[int] | None = None

    if preview:
        print("\n預覽模式：")
        print("  空白鍵/N：下一幀")
        print("  P：上一幀")
        print("  S：儲存當前幀")
        print("  Q/ESC：退出")
        print("-" * 40)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        # Ultralytics 接受 BGR numpy array
        results = model.predict(
            source=frame,
            conf=conf,
            iou=0.3,
            imgsz=imgsz,
            verbose=False,
        )
        if not results:
            frame_idx += 1
            continue

        result = results[0]
        if ball_class_ids is None:
            ball_class_ids = _infer_ball_class_ids(model, result)

        # 收集所有候選偵測（包含信心度）
        candidates = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            score = float(box.conf[0].item())
            cls_id = int(box.cls[0].item()) if box.cls is not None else 0
            
            if ball_class_ids is not None and cls_id not in ball_class_ids:
                continue
            
            # YOLO 格式：class cx cy w h（相對 0~1）
            cx = ((x1 + x2) / 2.0) / width
            cy = ((y1 + y2) / 2.0) / height
            w = (x2 - x1) / width
            h = (y2 - y1) / height
            
            if w <= 0 or h <= 0:
                continue
            
            # 棒球過濾
            if not is_valid_baseball(cx, cy, w, h):
                continue
            
            candidates.append((cx, cy, w, h, score))

        # NMS 去重
        dets = nms_filter(candidates)

        if preview:
            # 預覽模式：顯示偵測結果
            display = frame.copy()
            for cx, cy, w, h in dets:
                px = int(cx * width)
                py = int(cy * height)
                pw = int(w * width)
                ph = int(h * height)
                cv2.rectangle(display, (px - pw//2, py - ph//2), (px + pw//2, py + ph//2), (0, 255, 0), 2)
                cv2.circle(display, (px, py), 3, (0, 165, 255), -1)
            
            # 顯示資訊
            info = f"Frame {frame_idx}/{total_frames} | Detections: {len(dets)} | Saved: {saved_count}"
            cv2.putText(display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 縮放顯示
            scale = min(1280 / width, 720 / height, 1.0)
            if scale < 1.0:
                display = cv2.resize(display, None, fx=scale, fy=scale)
            
            cv2.imshow("Auto Label Preview", display)
            key = cv2.waitKey(0) & 0xFF
            
            if key in [ord('q'), 27]:  # Q or ESC
                break
            elif key == ord('s'):  # Save
                if dets:
                    base_name = f"{video_path.stem}_{frame_idx:06d}"
                    img_path = IMAGES_TRAIN / f"{base_name}.jpg"
                    label_path = LABELS_TRAIN / f"{base_name}.txt"
                    cv2.imwrite(str(img_path), frame)
                    with open(label_path, "w", encoding="utf-8") as f:
                        for cx, cy, w, h in dets:
                            f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                    saved_count += 1
                    print(f"✓ 已儲存：{base_name}")
            elif key in [ord('p')]:  # Previous
                frame_idx = max(0, frame_idx - stride * 2)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            # N, Space -> continue to next
            
            frame_idx += 1
            continue

        # 自動模式
        if not dets:
            skipped_count += 1
            frame_idx += 1
            continue

        base_name = f"{video_path.stem}_{frame_idx:06d}"
        img_path = IMAGES_TRAIN / f"{base_name}.jpg"
        label_path = LABELS_TRAIN / f"{base_name}.txt"

        cv2.imwrite(str(img_path), frame)
        with open(label_path, "w", encoding="utf-8") as f:
            for cx, cy, w, h in dets:
                f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        saved_count += 1
        frame_idx += 1
        
        # 進度顯示
        if saved_count % 100 == 0:
            print(f"  已處理 {frame_idx}/{total_frames} 幀，儲存 {saved_count} 筆...")

    if preview:
        cv2.destroyAllWindows()
    
    cap.release()
    print(f"  略過 {skipped_count} 幀（無有效偵測）")
    return saved_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="使用 Ultralytics YOLO 對影片做自動標註，輸出為 YOLO 訓練資料。"
    )
    parser.add_argument(
        "video",
        type=Path,
        help="要標註的影片路徑（例如 ../videos/16_120fps_4k.mp4）",
    )
    parser.add_argument(
        "-w", "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help=f"YOLO 權重檔 .pt（預設：{DEFAULT_WEIGHTS}）",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=2,
        help="每 N 幀取一張（預設 2，避免資料量過大）",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.15,
        help="偵測置信度閾值（預設 0.15）",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="YOLO 推論輸入尺寸（預設 1280）",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="預覽模式：可視覺化驗證標註結果後再儲存",
    )
    args = parser.parse_args()

    video_path = args.video
    if not video_path.is_absolute():
        # 支援相對於專案根目錄或相對於 yolov8/ 的路徑
        for base in (ROOT, YOLOV8_DIR):
            p = (base / video_path).resolve()
            if p.exists():
                video_path = p
                break
        else:
            video_path = (ROOT / video_path).resolve()
    if not video_path.exists():
        print(f"找不到影片：{video_path}", file=sys.stderr)
        sys.exit(1)

    weights_path = args.weights
    if not weights_path.is_absolute():
        weights_path = (YOLOV8_DIR / weights_path).resolve()
        if not weights_path.exists():
            weights_path = (ROOT / args.weights).resolve()
    if not weights_path.exists():
        print(f"找不到權重檔：{weights_path}", file=sys.stderr)
        sys.exit(1)

    for d in [IMAGES_TRAIN, LABELS_TRAIN]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"載入模型：{weights_path}")
    model = YOLO(str(weights_path))
    print(f"處理影片：{video_path}（stride={args.stride}, conf={args.conf}）")

    n = process_video(
        video_path,
        model,
        stride=args.stride,
        conf=args.conf,
        imgsz=args.imgsz,
        preview=args.preview,
    )

    print(f"\n完成。共寫入 {n} 筆自動標註樣本。")
    print(f"  影像：{IMAGES_TRAIN}")
    print(f"  標註：{LABELS_TRAIN}")
    print("可執行 train_yolo11.py 或 train_yolov8.py 將此資料一併納入訓練。")


if __name__ == "__main__":
    main()
