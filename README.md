# Speedgun - 投球姿勢與球軌跡分析

深度學習技術分析棒球投球姿勢與球軌跡的開源專案，支援影片分析和即時處理。

## 功能

- **棒球偵測**：使用 YOLOv4-tiny 或 YOLOv8 模型偵測棒球位置
- **姿勢分析**：整合 MediaPipe Pose 分析投球姿勢與關鍵點
- **軌跡追蹤**：自動追蹤球的飛行軌跡並繪製 overlay
- **出球點修正**：使用手腕關節自動修正出球點位置，讓軌跡更準確
- **影片 Overlay**：自動生成帶有分析結果的視覺化影片（包含姿勢骨架、球軌跡）
- **圖形介面**：提供 GUI 應用程式
- **iOS 支援**：包含 iOS 版本的實作（Swift + CoreML）

## 系統需求

- Python 3.10+
- TensorFlow 2.16+
- OpenCV 4.9+
- MediaPipe 0.10+
- Ultralytics (YOLOv8)

## 安裝步驟

1. **clone 專案**

   ```bash
   git clone https://github.com/yourusername/speedgun-mobile.git
   cd speedgun-mobile
   ```

2. **建立虛擬環境**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

3. **安裝依賴套件**

   ```bash
   pip install -r requirements.txt
   ```

4. **下載模型檔案**
   - YOLOv4-tiny 模型應放置在 `model/yolov4-tiny-baseball-416/` 目錄
   - YOLOv8 模型應放置在 `yolov8/runs/baseball_yolov8n2/weights/` 目錄

## 使用方法

### GUI 應用程式

執行圖形介面應用程式：

```bash
python gui_app.py
```

操作步驟：

1. 點擊「選擇影片」按鈕，選擇 1~2 支投球影片（支援 mp4/avi/mov/mkv 格式）
2. 選擇 YOLO 版本（v4 或 v8）
3. 點擊「開始分析」
4. 分析完成後，會在影片同一個資料夾輸出 `Overlay.mp4`


#### 使用 YOLOv4

```bash
python pitching_overlay.py --video path/to/video.mp4 --output output.mp4
```

#### 使用 YOLOv8

```bash
# 單一影片模式
python pitching_overlay_yolov8.py -v path/to/video.mp4 --conf 0.03

# 指定權重檔案和置信度閾值
python pitching_overlay_yolov8.py -v path/to/video.mp4 -w yolov8/runs/baseball_yolov8n2/weights/best.pt --conf 0.05

# 處理資料夾內所有影片
python pitching_overlay_yolov8.py -f videos/videos1 --conf 0.03
```

**參數：**

- `-v, --video_file`: 單一影片檔案路徑
- `-f, --videos_folder`: 包含多個影片的資料夾路徑
- `-w, --weights`: YOLOv8 權重檔路徑（預設：`yolov8/runs/baseball_yolov8n2/weights/best.pt`）
- `-c, --conf`: YOLOv8 置信度閾值（預設：0.1，建議 0.03~0.1 之間調整，數值越低越容易偵測到小球）

**輸出：**

- 單一影片模式：輸出到與輸入影片相同的資料夾，檔名為 `Overlay_yolov8.mp4`
- 資料夾模式：輸出到該資料夾，檔名為 `Overlay_yolov8.mp4`

## 專案結構

```
speedgun-mobile/
├── gui_app.py                 # GUI 主程式
├── pitching_overlay.py        # YOLOv4 處理流程
├── pitching_overlay_yolov8.py # YOLOv8 處理流程
├── requirements.txt           # Python 依賴套件
├── src/                       # 核心模組
│   ├── FrameInfo.py          # 影格資訊類別
│   ├── get_pitch_frames.py   # YOLOv4 投球影格提取
│   ├── get_pitch_frames_yolov8.py  # YOLOv8 投球影格提取
│   ├── generate_overlay.py   # 生成 overlay 影片
│   ├── utils.py              # 工具函數
│   └── SORT_tracker/         # SORT 追蹤演算法
├── model/                     # 模型檔案
│   └── yolov4-tiny-baseball-416/  # YOLOv4 模型
├── yolov8/                    # YOLOv8 訓練與模型
│   ├── train_yolov8.py       # 訓練腳本
│   └── runs/                  # 訓練結果與權重
├── ios/                       # iOS 版本（Swift）
│   ├── SpeedgunMobileApp.swift
│   ├── ContentView.swift
│   ├── CameraViewModel.swift
│   └── FrameProcessor.swift
└── test_scripts/              # 測試腳本
```

## 架構

- **物件偵測**：YOLOv4-tiny / YOLOv8 (Ultralytics)
- **姿勢估計**：MediaPipe Pose
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
