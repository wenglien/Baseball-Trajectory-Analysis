# 投球姿勢與球軌跡分析

深度學習技術分析棒球投球姿勢與球軌跡的開源專案，支援影片分析和即時處理。

## 功能

- YOLOv4-tiny或YOLOv8 模型偵測棒球位置
- 整合MediaPipe Pose分析投球姿勢與關鍵點
- 追蹤球的飛行軌跡並繪製 overlay
- 測量球速，包含出手球速、最大速度和飛行距離
- 使用手腕關節自動修正出球點位置，讓軌跡更準確
- 考慮相機透視變形，提供更準確的距離和速度計算
- 自動生成帶有分析結果的視覺化影片（包含姿勢骨架、球軌跡、球速資訊）


## 系統需求

- Python 3.10+
- pip（搭配 venv 虛擬環境）
- macOS / Windows / Linux 皆可
- TensorFlow、OpenCV、MediaPipe、Ultralytics（YOLOv8）

## 安裝步驟

1. **下載專案**

```bash
git clone https://github.com/yourusername/speedgun-mobile.git
cd speedgun-mobile
```

2. **建立虛擬環境**

macOS / Linux：

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
```

Windows（PowerShell）：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip
```

3. **安裝依賴套件**

```bash
python3 -m pip install -r requirements.txt
```

4. **準備模型權重**

本專案**不一定包含**可直接使用的權重檔。有兩種做法：

- **A：使用你已經有的權重（推薦）**
  - YOLOv8：準備一個 `.pt` 權重檔，之後用參數 `-w/--weights` 指向它即可。
- **B：自己訓練（需要時才做）**
  - 參考下方「訓練自訂模型」。

若你想沿用 README 的預設路徑：

- **YOLOv8 權重**：放在 `yolov8/runs/baseball_yolov8n2/weights/best.pt`
- **YOLOv4-tiny 模型**（只有選 YOLOv4 才需要）： `model/yolov4-tiny-baseball-416/`

5. **快速驗證是否安裝成功**

```bash
python3 -c "import cv2, mediapipe, ultralytics, tensorflow as tf; print('OK')"
```

## 使用方法

### GUI 應用程式

執行圖形介面應用程式：

```bash
python3 gui_app.py
```

操作步驟：

1. 點擊「選擇影片」按鈕，選擇 1~2 支投球影片（支援 mp4/avi/mov/mkv 格式）
2. 選擇 YOLO 版本（v4 或 v8）
3. 點擊「開始分析」
4. 分析完成後，會在影片同一個資料夾輸出 overlay 影片（YOLOv4：`Overlay.mp4`，YOLOv8：`Overlay_yolov8.mp4`）


# Video Preprocessing

This folder contains `preprocess_video.py` — a small helper that uses `ffmpeg` to:

- Increase FPS to 120 (via interpolation or duplication)
- Scale and pad to 1920x1080 (1080p) while preserving aspect ratio

- Optionally target 4K (3840x2160) using `--resolution 4k`

Prerequisites:

- `ffmpeg` installed and available on `PATH`.

Usage examples:

Process a single file (interpolation) to 1080p:

```
python preprocess_video.py input.mp4
```

Process a single file to 4K:

```
python preprocess_video.py input.mp4 --resolution 4k
```

Force simple duplication instead of interpolation:

```
python preprocess_video.py input.mp4 --method duplicate
```

Process all supported files in a directory (output folder created):

```
python preprocess_video.py ./videos
```

Specify an output file or folder with `-o` / `--output`.

Notes:

- Interpolation (the default) produces smoother motion but takes more CPU/time.
- The script encodes video with `libx264` (CRF 18 by default). Adjust `--crf` and `--preset` as needed.

#### 使用 YOLOv4

```bash
python3 pitching_overlay.py --video path/to/video.mp4 --output output.mp4
```

#### 使用 YOLOv8（推薦，支援球速測定）

**基本使用（啟用球速測定，改用手動輸入距離）：**
```bash
# 單一影片模式
python3 pitching_overlay_yolov8.py -v path/to/video.mp4 --conf 0.03

# 預設距離為 18.44m（投手丘到本壘板）
# 若要改距離，請加上 --distance（公尺）
```

**指定投手到捕手距離（公尺）：**
```bash
python3 pitching_overlay_yolov8.py -v path/to/video.mp4 --conf 0.03 --distance 18.44
```

**停用球速計算：**
```bash
python3 pitching_overlay_yolov8.py -v path/to/video.mp4 --conf 0.03 --no-speed
```

**處理資料夾內所有影片：**
```bash
python3 pitching_overlay_yolov8.py -f videos/videos1 --conf 0.03
```

**參數：**

- `-v, --video_file`: 單一影片檔案路徑
- `-f, --videos_folder`: 包含多個影片的資料夾路徑
- `-w, --weights`: YOLOv8 權重檔路徑（預設：`yolov8/runs/baseball_yolov8n2/weights/best.pt`）
- `-c, --conf`: YOLOv8 置信度閾值（預設：0.1，建議 0.03~0.1 之間調整）
- `--no-speed`: 停用球速計算功能
- `-d, --distance`: 手動輸入投手到捕手距離（公尺），例如：18.44 或 15
- `--no-calibration`: （已廢止）過去用於停用點選校正；目前已移除點選校正，此參數保留相容

**輸出：**

- 單一影片模式：輸出到與輸入影片相同的資料夾，檔名為 `Overlay_yolov8.mp4`
- 包含視覺化的球速資訊：出手球速、最大速度、飛行距離

## 專案結構

```
speedgun-mobile/
├── gui_app.py                 # GUI 主程式
├── pitching_overlay.py        # YOLOv4 處理流程
├── pitching_overlay_yolov8.py # YOLOv8 處理流程
├── export_mediapipe_pose.py   # MediaPipe Pose 匯出工具
├── requirements.txt           # Python 依賴套件
├── src/                       # 核心模組
│   ├── FrameInfo.py          # 影格資訊類別
│   ├── get_pitch_frames.py   # YOLOv4 投球影格擷取
│   ├── get_pitch_frames_yolov8.py  # YOLOv8 投球影格擷取
│   ├── generate_overlay.py   # 生成 overlay 影片
│   ├── ball_speed_calculator.py  # 球速計算模組
│   ├── release_point_detector.py # 出球點偵測模組
│   ├── utils.py              # tool函數
│   └── SORT_tracker/         # SORT 演算法
│       ├── kalman_filter.py
│       ├── sort.py
│       └── tracker.py
├── model/                     # 模型檔案、轉換工具
│   ├── convert_yolov4_tiny_to_coreml.py  # CoreML轉換腳本
│   └── yolov4-tiny-baseball-416/  # YOLOv4 模型
├── yolov8/                    # YOLOv8 訓練與模型
│   ├── train_yolov8.py       # 訓練腳本
│   └── runs/                  # 訓練結果與權重
├── ios/                       # iOS 版本（Swift）
│   ├── SpeedgunMobileApp.swift
│   ├── ContentView.swift
│   ├── CameraViewModel.swift
│   ├── CameraPreview.swift
│   ├── FrameProcessor.swift
│   └── PoseAndBallOverlay.swift
└── img/                       # 專案圖片與範例
    └── *.gif                 # 範例動畫
```

## 架構

- **物件偵測**：YOLOv4-tiny / YOLOv8 (Ultralytics)
- **姿勢標記**：MediaPipe Pose
- **物件追蹤**：SORT (Simple Online and Realtime Tracking) / 簡化追蹤（YOLOv8 版本）
- **影像處理**：OpenCV
- **深度學習框架**：TensorFlow, Ultralytics

### YOLOv8 版本改進
- YOLOv8 官方影片串流介面，提升偵測穩定性
- 用手腕關節修正出球點，讓軌跡起點更準確
- 優化軌跡顯示：固定長度尾巴，讓速度感更貼合實際飛行

### 訓練自訂模型

YOLOv8 模型訓練：

```bash
cd yolov8
python train_yolov8.py
```

## 授權

本專案採用 MIT 授權條款 - 詳見 [LICENSE](LICENSE) 檔案

- [YOLO](https://github.com/AlexeyAB/darknet) - 物件偵測模型
- [MediaPipe](https://mediapipe.dev/) - 姿勢估計
- [SORT](https://github.com/abewley/sort) - 物件追蹤演算法
- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLOv8 實作
