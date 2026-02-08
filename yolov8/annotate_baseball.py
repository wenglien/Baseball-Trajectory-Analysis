"""
棒球標註工具 - 視覺化 GUI 介面

功能：
1. 從影片中逐幀標註棒球位置
2. 使用備份的 YOLO 權重自動預標註（可選）
3. 手動校正標註
4. 輸出 YOLO 格式訓練資料

操作說明：
- 左鍵點擊：標註棒球中心位置
- 右鍵點擊：刪除最近的標註
- N / → / 空白鍵：下一幀
- P / ←：上一幀
- S：儲存當前標註
- A：自動標註當前幀（需要 YOLO 權重）
- Q / ESC：退出
- +/-：調整 bbox 大小

用法：
  cd yolov8
  python annotate_baseball.py ../videos/your_video.mp4
  
  # 使用自動預標註
  python annotate_baseball.py ../videos/your_video.mp4 --auto-label
  
  # 指定權重
  python annotate_baseball.py ../videos/your_video.mp4 -w ../backup_weights/baseball_yolov8n2_best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
YOLOV8_DIR = Path(__file__).resolve().parent
DATASET_ROOT = YOLOV8_DIR / "datasets" / "baseball"
IMAGES_TRAIN = DATASET_ROOT / "images" / "train"
LABELS_TRAIN = DATASET_ROOT / "labels" / "train"

# 預設權重路徑
DEFAULT_WEIGHTS = ROOT / "backup_weights" / "baseball_yolov8n2_best.pt"

# 標註設定
DEFAULT_BBOX_SIZE = 30  # 預設 bbox 大小（直徑像素）
MIN_BBOX_SIZE = 10
MAX_BBOX_SIZE = 100


class BaseballAnnotator:
    def __init__(
        self,
        video_path: Path,
        weights_path: Optional[Path] = None,
        auto_label: bool = False,
        stride: int = 1,
    ):
        self.video_path = video_path
        self.weights_path = weights_path
        self.auto_label = auto_label
        self.stride = stride
        
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise ValueError(f"無法開啟影片：{video_path}")
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        self.current_frame_idx = 0
        self.current_frame = None
        self.annotations: dict[int, list[tuple[float, float, float, float]]] = {}
        self.bbox_size = DEFAULT_BBOX_SIZE
        self.saved_count = 0
        
        # YOLO 模型（延遲載入）
        self.yolo_model = None
        if weights_path and weights_path.exists():
            print(f"載入 YOLO 模型：{weights_path}")
            from ultralytics import YOLO
            self.yolo_model = YOLO(str(weights_path))
        elif auto_label:
            print("警告：未找到 YOLO 權重，無法自動標註")
        
        # 確保輸出目錄存在
        IMAGES_TRAIN.mkdir(parents=True, exist_ok=True)
        LABELS_TRAIN.mkdir(parents=True, exist_ok=True)
        
        # 視窗名稱
        self.window_name = "Baseball Annotator"
    
    def read_frame(self, idx: int) -> Optional[np.ndarray]:
        """讀取指定幀"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def auto_detect(self, frame: np.ndarray) -> list[tuple[float, float, float, float]]:
        """使用 YOLO 自動偵測"""
        if self.yolo_model is None:
            return []
        
        results = self.yolo_model.predict(
            source=frame,
            conf=0.1,
            iou=0.3,
            imgsz=1280,
            verbose=False,
        )
        
        detections = []
        if results:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = ((x1 + x2) / 2.0) / self.width
                cy = ((y1 + y2) / 2.0) / self.height
                w = (x2 - x1) / self.width
                h = (y2 - y1) / self.height
                if w > 0 and h > 0:
                    detections.append((cx, cy, w, h))
        
        return detections
    
    def draw_frame(self, frame: np.ndarray) -> np.ndarray:
        """繪製標註和 UI"""
        display = frame.copy()
        
        # 繪製標註框
        annotations = self.annotations.get(self.current_frame_idx, [])
        for i, (cx, cy, w, h) in enumerate(annotations):
            px = int(cx * self.width)
            py = int(cy * self.height)
            pw = int(w * self.width)
            ph = int(h * self.height)
            
            x1 = px - pw // 2
            y1 = py - ph // 2
            x2 = px + pw // 2
            y2 = py + ph // 2
            
            # 綠色框 + 橘色中心點
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(display, (px, py), 3, (0, 165, 255), -1)
            cv2.putText(display, f"{i+1}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 繪製 UI 資訊
        info_lines = [
            f"Frame: {self.current_frame_idx + 1}/{self.total_frames} ({self.fps} FPS)",
            f"Annotations: {len(annotations)} | Saved: {self.saved_count}",
            f"BBox size: {self.bbox_size}px | Stride: {self.stride}",
            "",
            "Controls:",
            "  Click: Add ball | Right-click: Remove",
            "  N/Space/->: Next | P/<-: Prev",
            "  S: Save | A: Auto-detect | Q/ESC: Quit",
            "  +/-: Adjust bbox size",
        ]
        
        y_offset = 30
        for line in info_lines:
            cv2.putText(display, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
        
        return display
    
    def mouse_callback(self, event, x, y, flags, param):
        """滑鼠事件處理"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # 左鍵：新增標註
            cx = x / self.width
            cy = y / self.height
            w = self.bbox_size / self.width
            h = self.bbox_size / self.height
            
            if self.current_frame_idx not in self.annotations:
                self.annotations[self.current_frame_idx] = []
            self.annotations[self.current_frame_idx].append((cx, cy, w, h))
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # 右鍵：刪除最近的標註
            annotations = self.annotations.get(self.current_frame_idx, [])
            if annotations:
                # 找最近的標註
                min_dist = float('inf')
                min_idx = -1
                for i, (cx, cy, w, h) in enumerate(annotations):
                    px = cx * self.width
                    py = cy * self.height
                    dist = (x - px) ** 2 + (y - py) ** 2
                    if dist < min_dist:
                        min_dist = dist
                        min_idx = i
                
                if min_idx >= 0:
                    annotations.pop(min_idx)
    
    def save_current_frame(self):
        """儲存當前幀的標註"""
        if self.current_frame is None:
            return
        
        annotations = self.annotations.get(self.current_frame_idx, [])
        if not annotations:
            print(f"幀 {self.current_frame_idx}: 沒有標註，略過")
            return
        
        base_name = f"{self.video_path.stem}_{self.current_frame_idx:06d}"
        img_path = IMAGES_TRAIN / f"{base_name}.jpg"
        lbl_path = LABELS_TRAIN / f"{base_name}.txt"
        
        cv2.imwrite(str(img_path), self.current_frame)
        with open(lbl_path, "w", encoding="utf-8") as f:
            for cx, cy, w, h in annotations:
                f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        
        self.saved_count += 1
        print(f"✓ 已儲存：{base_name} ({len(annotations)} 個標註)")
    
    def run(self):
        """主迴圈"""
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print(f"\n開始標註：{self.video_path}")
        print(f"影片資訊：{self.width}x{self.height} @ {self.fps} FPS, {self.total_frames} 幀")
        print("-" * 50)
        
        while True:
            self.current_frame = self.read_frame(self.current_frame_idx)
            if self.current_frame is None:
                print("無法讀取幀，可能已到影片結尾")
                break
            
            # 自動標註模式
            if self.auto_label and self.current_frame_idx not in self.annotations:
                auto_dets = self.auto_detect(self.current_frame)
                if auto_dets:
                    self.annotations[self.current_frame_idx] = auto_dets
            
            display = self.draw_frame(self.current_frame)
            cv2.imshow(self.window_name, display)
            
            key = cv2.waitKey(0) & 0xFF
            
            if key in [ord('q'), 27]:  # Q or ESC
                break
            elif key in [ord('n'), ord(' '), 83]:  # N, Space, Right arrow
                self.current_frame_idx = min(self.current_frame_idx + self.stride, self.total_frames - 1)
            elif key in [ord('p'), 81]:  # P, Left arrow
                self.current_frame_idx = max(self.current_frame_idx - self.stride, 0)
            elif key == ord('s'):  # S: Save
                self.save_current_frame()
            elif key == ord('a'):  # A: Auto-detect
                auto_dets = self.auto_detect(self.current_frame)
                if auto_dets:
                    self.annotations[self.current_frame_idx] = auto_dets
                    print(f"自動偵測到 {len(auto_dets)} 個目標")
                else:
                    print("未偵測到目標")
            elif key == ord('=') or key == ord('+'):  # +: Increase bbox
                self.bbox_size = min(self.bbox_size + 5, MAX_BBOX_SIZE)
            elif key == ord('-'):  # -: Decrease bbox
                self.bbox_size = max(self.bbox_size - 5, MIN_BBOX_SIZE)
        
        cv2.destroyAllWindows()
        self.cap.release()
        
        print(f"\n標註完成！")
        print(f"  總共儲存：{self.saved_count} 筆訓練資料")
        print(f"  圖片目錄：{IMAGES_TRAIN}")
        print(f"  標籤目錄：{LABELS_TRAIN}")


def main():
    parser = argparse.ArgumentParser(description="棒球標註工具")
    parser.add_argument("video", type=Path, help="要標註的影片路徑")
    parser.add_argument("-w", "--weights", type=Path, default=DEFAULT_WEIGHTS, help="YOLO 權重路徑")
    parser.add_argument("--auto-label", action="store_true", help="啟用自動預標註")
    parser.add_argument("--stride", type=int, default=1, help="每 N 幀移動一次（預設 1）")
    args = parser.parse_args()
    
    video_path = args.video
    if not video_path.is_absolute():
        for base in [ROOT, YOLOV8_DIR]:
            p = (base / video_path).resolve()
            if p.exists():
                video_path = p
                break
    
    if not video_path.exists():
        print(f"找不到影片：{video_path}")
        sys.exit(1)
    
    annotator = BaseballAnnotator(
        video_path=video_path,
        weights_path=args.weights if args.weights.exists() else None,
        auto_label=args.auto_label,
        stride=args.stride,
    )
    annotator.run()


if __name__ == "__main__":
    main()
