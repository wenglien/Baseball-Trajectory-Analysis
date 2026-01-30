import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import List

import tensorflow as tf
from tensorflow.python.saved_model import tag_constants

from src.FrameInfo import FrameInfo
from pitching_overlay import run_yolov4_pipeline
from pitching_overlay_yolov8 import run_yolov8_pipeline


class SpeedgunGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Speedgun - 投球姿勢與球軌跡分析")
        self.root.geometry("720x380")
        self.video_paths: List[str] = []
        self.status_text = tk.StringVar(value="請先選擇 1~2 支投球影片。")

        self.infer = None

        self.yolo_version = tk.StringVar(value="v4")
        self.yolov8_weights = tk.StringVar(
            value=os.path.join(
                "yolov8", "runs", "baseball_yolov8n2", "weights", "best.pt"
            )
        )
        self.yolov8_conf = tk.DoubleVar(value=0.1)
        self.pitch_distance_meters = tk.DoubleVar(value=18.44)

        self._build_ui()

    def _build_ui(self):
        padding = {"padx": 16, "pady": 8}

        frame_select = tk.Frame(self.root)
        frame_select.pack(fill="x", **padding)

        btn_select = tk.Button(
            frame_select, text="選擇影片（可多選）...", width=18, command=self.choose_videos
        )
        btn_select.pack(side="left")

        self.entry_videos = tk.Entry(frame_select)
        self.entry_videos.pack(side="left", fill="x", expand=True, padx=(8, 0))

        frame_yolo = tk.Frame(self.root)
        frame_yolo.pack(fill="x", **padding)

        tk.Label(frame_yolo, text="偵測引擎：").pack(side="left")
        tk.Radiobutton(
            frame_yolo, text="YOLOv4 (TF)", variable=self.yolo_version, value="v4"
        ).pack(side="left")
        tk.Radiobutton(
            frame_yolo, text="YOLOv8 (PyTorch)", variable=self.yolo_version, value="v8"
        ).pack(side="left", padx=(8, 0))

        frame_yolo8 = tk.Frame(self.root)
        frame_yolo8.pack(fill="x", **padding)

        tk.Label(frame_yolo8, text="YOLOv8 weights:").pack(side="left")
        entry_weights = tk.Entry(frame_yolo8, textvariable=self.yolov8_weights)
        entry_weights.pack(side="left", fill="x", expand=True, padx=(4, 0))

        tk.Label(frame_yolo8, text="conf:").pack(side="left", padx=(8, 0))
        entry_conf = tk.Entry(frame_yolo8, textvariable=self.yolov8_conf, width=6)
        entry_conf.pack(side="left")

        frame_distance = tk.Frame(self.root)
        frame_distance.pack(fill="x", **padding)
        tk.Label(frame_distance, text="投手到捕手距離（公尺）:").pack(side="left")
        entry_distance = tk.Entry(
            frame_distance, textvariable=self.pitch_distance_meters, width=8
        )
        entry_distance.pack(side="left", padx=(4, 0))
        tk.Label(frame_distance, text="（預設 18.44m）").pack(side="left", padx=(8, 0))

        frame_actions = tk.Frame(self.root)
        frame_actions.pack(fill="x", **padding)

        self.btn_run = tk.Button(
            frame_actions, text="開始分析", width=15, command=self.run_analysis_clicked
        )
        self.btn_run.pack(side="left")

        self.lbl_status = tk.Label(
            self.root, textvariable=self.status_text, anchor="w", justify="left"
        )
        self.lbl_status.pack(fill="x", padx=16, pady=(4, 0))

        info_text = (
            "說明：\n"
            "1. 按「選擇影片」挑一支投球影片（mp4/avi/mov/mkv）。\n"
            "2. 按「開始分析」，程式會執行：YOLO 棒球偵測 + Mediapipe 姿勢 + overlay。\n"
            "3. 球速會使用「投手到捕手距離（公尺）」做計算（不再需要點選校正）。\n"
            "4. 分析完成後，會在同一個資料夾輸出 Overlay.mp4。"
        )
        lbl_info = tk.Label(self.root, text=info_text, justify="left")
        lbl_info.pack(fill="x", padx=16, pady=(8, 0))

    def choose_videos(self):
        paths = filedialog.askopenfilenames(
            title="選擇 1~2 支投球影片",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*"),
            ],
        )
        if not paths:
            return
        self.video_paths = list(paths[:2])
        self.entry_videos.delete(0, tk.END)
        self.entry_videos.insert(0, "; ".join(self.video_paths))
        self.status_text.set(f"已選擇影片（最多 2 支）：{'; '.join(self.video_paths)}")

    def run_analysis_clicked(self):
        if not self.video_paths:
            messagebox.showwarning("提示", "請先選擇至少一支影片（建議 1~2 支）。")
            return

        t = threading.Thread(target=self._run_analysis, daemon=True)
        t.start()

    def _run_analysis(self):
        try:
            self._set_running(True)
            if len(self.video_paths) == 1:
                base_dir = os.path.dirname(self.video_paths[0])
            else:
                base_dir = os.path.dirname(self.video_paths[0])

            yolo_ver = self.yolo_version.get()
            if yolo_ver == "v4":
                self.status_text.set("使用 YOLOv4 分析影片中，請稍候...")
                output_path = os.path.join(base_dir, "Overlay.mp4")
                run_yolov4_pipeline(self.video_paths, output_path, show_preview=False)
            else:
                self.status_text.set("使用 YOLOv8 分析影片中，請稍候...")
                output_path = os.path.join(base_dir, "Overlay_yolov8.mp4")
                run_yolov8_pipeline(
                    self.video_paths,
                    weights_path=self.yolov8_weights.get().strip(),
                    conf=float(self.yolov8_conf.get()),
                    output_path=output_path,
                    show_preview=False,
                    manual_distance_meters=float(self.pitch_distance_meters.get()),
                )

            self.status_text.set(f"完成！結果已輸出到：{output_path}")
            messagebox.showinfo("完成", f"分析完成，輸出檔案：\n{output_path}")

        except Exception as e:
            self.status_text.set(f"發生錯誤：{e}")
            messagebox.showerror("錯誤", str(e))
        finally:
            self._set_running(False)

    def _set_running(self, running: bool):
        state = tk.DISABLED if running else tk.NORMAL
        self.btn_run.config(state=state)


def main():
    try:
        physical_devices = tf.config.experimental.list_physical_devices("GPU")
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except Exception:
        pass

    root = tk.Tk()
    app = SpeedgunGUI(root)
    root.mainloop()


if __name__ == "__main__":
    ROOT = os.path.dirname(os.path.abspath(__file__))
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    main()

