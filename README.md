# 投球姿勢與球軌跡分析

深度學習技術分析棒球投球姿勢與球軌跡的開源專案，支援影片分析和即時處理。

## 功能

- Ultralytics YOLO（**推薦/預設：YOLO11**；亦相容 YOLOv8；YOLOv4 為 legacy）偵測棒球位置
- 整合MediaPipe Pose分析投球姿勢與關鍵點
- 追蹤球的飛行軌跡並繪製 overlay
- 測量球速，包含出手球速、最大速度和飛行距離
- 使用手腕關節自動修正出球點位置，讓軌跡更準確
- 考慮相機透視變形，提供更準確的距離和速度計算
- 自動生成帶有分析結果的視覺化影片（包含姿勢骨架、球軌跡、球速資訊）


## 系統需求

- Python 3.10 / 3.11（**建議 3.11**；Python 3.12+ 可能會遇到 mediapipe / tensorflow 相容性問題）
- pip（搭配 venv 虛擬環境）
- macOS / Windows / Linux 皆可
- OpenCV、MediaPipe、Ultralytics（YOLO11 / YOLOv8）
- （選用）TensorFlow：只有在使用舊的 YOLOv4 流程或轉換/自動標註腳本時才需要

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

若你需要舊的 YOLOv4 / TensorFlow（不建議，且軌跡穩定度較差）：

```bash
python3 -m pip install -r requirements-yolov4.txt
```

（推薦）**一鍵安裝 + 環境健檢（macOS / Linux）**

```bash
./scripts/bootstrap_dev.sh
```

只裝 Ultralytics YOLO（YOLO11/YOLOv8）依賴時：

```bash
REQ_FILE=requirements-yolov8.txt ./scripts/bootstrap_dev.sh
```

4. **準備模型權重**

本專案**不一定包含**可直接使用的權重檔。有兩種做法：

- **A：使用你已經有的權重（推薦）**
  - Ultralytics YOLO（YOLO11/YOLOv8）：準備一個 `.pt` 權重檔，之後用參數 `-w/--weights` 指向它即可。
- **B：自己訓練（需要時才做）**
  - 參考下方「訓練自訂模型」。

若你想沿用 README 的預設路徑：

- **YOLO（預設 YOLO11）權重**：放在 `yolov8/runs/baseball_yolo11n/weights/best.pt`
- **YOLOv4-tiny 模型**（只有選 YOLOv4 才需要）： `model/yolov4-tiny-baseball-416/`

5. **快速驗證是否安裝成功**

```bash
python3 -c "import cv2, mediapipe, ultralytics; print('OK')"
```

### 常見環境問題排查

#### `AttributeError: module 'mediapipe' has no attribute 'solutions'`

這通常不是你的程式碼問題，而是 **import 到錯的 `mediapipe`**（被同名檔案/資料夾覆蓋、或裝到不同 Python 環境）。
另外，若安裝到較新的 mediapipe 發佈版本，可能不再提供 `mp.solutions`（Legacy Solutions）。

請在專案根目錄執行：

```bash
python scripts/doctor.py
```

（建議）快速 smoke test（不會跑推論、不會開 GUI）：

```bash
python scripts/smoke_test.py
```

`doctor` 會輸出：
- `mediapipe.__file__`（實際載入的來源路徑）
- 是否有 `mediapipe.py` / `mediapipe/` 造成覆蓋
- Python 版本是否太新（3.12+）

若你看到 `mediapipe.__file__` 指向專案內的同名檔案，請改名或移除後重試。
若 `mediapipe.__version__` 很新且 `mediapipe.has_solutions: False`，請改裝本專案固定版本：

```bash
python3 -m pip uninstall -y mediapipe
python3 -m pip install mediapipe==0.10.21
```

## 使用方法

### GUI 應用程式

執行圖形介面應用程式：

```bash
python3 gui_app.py
```

操作步驟：

1. 點擊「選擇影片」按鈕，選擇 1~2 支投球影片（支援 mp4/avi/mov/mkv 格式）
2. 點擊「開始分析」
3. 分析完成後，會在影片同一個資料夾輸出 overlay 影片：`Overlay.mp4`


#### 使用 Ultralytics YOLO（預設/推薦：YOLO11；支援球速測定）

```bash
# 單一影片模式
python3 pitching_overlay.py -v path/to/video.mp4 --conf 0.03
```

（除錯/輸出控制）

- `--debug`: 顯示完整 traceback（方便定位錯誤）
- `-V / --verbose`：輸出更多除錯資訊（可重複，例如 `-VV`）
- `--quiet`：只輸出錯誤訊息（ERROR）


**基本使用（啟用球速測定，改用手動輸入距離）：**
```bash
# 單一影片模式
python3 pitching_overlay.py -v path/to/video.mp4 --conf 0.03

# 預設距離為 18.44m（投手丘到本壘板）
# 若要改距離，請加上 --distance（公尺）
```

**指定投手到捕手距離（公尺）：**
```bash
python3 pitching_overlay.py -v path/to/video.mp4 --conf 0.03 --distance 18.44
```

**停用球速計算：**
```bash
python3 pitching_overlay.py -v path/to/video.mp4 --conf 0.03 --no-speed
```

**處理資料夾內所有影片：**
```bash
python3 pitching_overlay.py -f videos/videos1 --conf 0.03
```

**參數：**

- `-v, --video_file`: 單一影片檔案路徑
- `-f, --videos_folder`: 包含多個影片的資料夾路徑
- `-w, --weights`: Ultralytics YOLO 權重檔路徑（預設：`yolov8/runs/baseball_yolo11n/weights/best.pt`）
- `-c, --conf`: YOLO 置信度閾值（預設：0.1，建議 0.03~0.2 之間調整）
- `--no-speed`: 停用球速計算功能
- `-d, --distance`: 手動輸入投手到捕手距離（公尺），例如：18.44 或 15
- `--no-calibration`: （已廢止）過去用於停用點選校正；目前已移除點選校正，此參數保留相容

**輸出：**

- 單一影片模式：輸出到與輸入影片相同的資料夾，檔名為 `Overlay.mp4`
- 包含視覺化的球速資訊：出手球速、最大速度、飛行距離

## 專案結構

```
speedgun-mobile/
├── legacy/                    # legacy 功能集中（保留相容路徑）
│   └── yolov4/
│       ├── get_pitch_frames.py # YOLOv4 投球影格擷取（實作）
│       └── convert_yolov4_tiny_to_coreml.py  # YOLOv4 → CoreML 轉換（實作）
├── gui_app.py                 # GUI 主程式
├── pitching_overlay.py        # CLI 入口（YOLO11/YOLOv8 + 選用 legacy YOLOv4）
├── scripts/                   # 開發/輔助腳本
│   ├── export_mediapipe_pose.py  # MediaPipe Pose 匯出（姿勢骨架影片）
│   └── preprocess_video.py    # 影片前處理（120fps/1080p）
├── requirements.txt           # Python 依賴套件
├── requirements-yolov4.txt    # 舊 YOLOv4 / TensorFlow 依賴（legacy）
├── src/                       # 核心模組
│   ├── pipelines/            # 共用 pipeline（GUI/CLI 共用）
│   │   └── yolov8_pipeline.py
│   ├── FrameInfo.py          # 影格資訊類別
│   ├── get_pitch_frames.py   # YOLOv4 相容 shim（延遲載入；實作在 legacy/yolov4/）
│   ├── get_pitch_frames_yolov8.py  # Ultralytics YOLO 投球影格擷取（歷史檔名）
│   ├── generate_overlay.py   # 生成 overlay 影片
│   ├── ball_speed_calculator.py  # 球速計算模組
│   ├── release_point_detector.py # 出球點偵測模組
│   ├── utils.py              # tool函數
│   └── SORT_tracker/         # SORT 演算法
│       ├── kalman_filter.py
│       ├── sort.py
│       └── tracker.py
├── model/                     # 模型檔案、轉換工具
│   ├── convert_yolov4_tiny_to_coreml.py  # CoreML 轉換 shim（實作在 legacy/yolov4/）
│   └── yolov4-tiny-baseball-416/  # YOLOv4 模型
├── yolov8/                    # Ultralytics YOLO 訓練與模型（歷史資料夾名稱）
│   ├── train_yolov8.py       # 訓練腳本（YOLOv8）
│   ├── train_yolo11.py       # 訓練腳本（YOLO11）
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

- **物件偵測**：YOLOv4-tiny / Ultralytics YOLO（YOLO11/YOLOv8）
- **姿勢標記**：MediaPipe Pose
- **物件追蹤**：SORT (Simple Online and Realtime Tracking) / 簡化追蹤（Ultralytics YOLO 版本）
- **影像處理**：OpenCV
- **深度學習框架**：Ultralytics；（legacy）TensorFlow

### Ultralytics YOLO 版本改進
- 使用 Ultralytics 官方影片串流介面，提升偵測穩定性
- 用手腕關節修正出球點，讓軌跡起點更準確
- 優化軌跡顯示：固定長度尾巴，讓速度感更貼合實際飛行

### 訓練自訂模型

YOLO11（推薦）模型訓練：

```bash
cd yolov8
python train_yolo11.py
```

YOLOv8（備案/相容）模型訓練：

```bash
cd yolov8
python train_yolov8.py
```

## 授權

本專案採用 MIT 授權條款 - 詳見 [LICENSE](LICENSE) 檔案

- [YOLO](https://github.com/AlexeyAB/darknet) - 物件偵測模型
- [MediaPipe](https://mediapipe.dev/) - 姿勢估計
- [SORT](https://github.com/abewley/sort) - 物件追蹤演算法
- [Ultralytics](https://github.com/ultralytics/ultralytics) - Ultralytics YOLO（YOLO11/YOLOv8）
