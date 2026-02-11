import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import List, Optional
import cv2
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# Adjust import path if needed
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from src.pipelines.yolov8_pipeline import run_yolov8_pipeline
except ImportError:
    pass

class SpeedgunApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Speedgun Mobile - Pro Analysis Dashboard")
        self.geometry("1400x900")
        
        # Configure Dark Theme Colors
        self.colors = {
            "bg_dark": "#1e1e1e",
            "bg_panel": "#2d2d2d",
            "fg_text": "#ffffff",
            "fg_subtext": "#aaaaaa",
            "accent": "#3498db",
            "accent_hover": "#2980b9",
            "success": "#2ecc71",
            "warning": "#f1c40f",
            "danger": "#e74c3c"
        }
        
        self.configure(bg=self.colors["bg_dark"])
        self._setup_styles()
        
        # Data Variables
        self.video_paths: List[str] = []
        self.current_video_frame = None
        self.pipeline_running = False
        
        # UI State Variables
        self.status_msg = tk.StringVar(value="Ready to analyze.")
        self.yolov8_weights = tk.StringVar(value=os.path.join("yolov8", "best_baseball.pt"))
        self.pitch_dist = tk.DoubleVar(value=18.44)
        
        # Analysis Results
        self.result_data = {}
        
        self._build_ui()
        
    def _setup_styles(self):
        style = ttk.Style(self)
        style.theme_use('clam')
        
        # General Frame
        style.configure("TFrame", background=self.colors["bg_dark"])
        style.configure("Panel.TFrame", background=self.colors["bg_panel"])
        
        # Labels
        style.configure("TLabel", background=self.colors["bg_dark"], foreground=self.colors["fg_text"], font=("Segoe UI", 10))
        style.configure("Panel.TLabel", background=self.colors["bg_panel"], foreground=self.colors["fg_text"])
        style.configure("Header.TLabel", font=("Segoe UI", 16, "bold"), background=self.colors["bg_dark"], foreground=self.colors["accent"])
        style.configure("SubHeader.TLabel", font=("Segoe UI", 12, "bold"), background=self.colors["bg_panel"], foreground=self.colors["fg_subtext"])
        
        # Metrics
        style.configure("MetricVal.TLabel", font=("Segoe UI", 24, "bold"), background=self.colors["bg_panel"], foreground=self.colors["accent"])
        style.configure("MetricLabel.TLabel", font=("Segoe UI", 10), background=self.colors["bg_panel"], foreground=self.colors["fg_subtext"])

        # Buttons
        style.configure("Accent.TButton", background=self.colors["accent"], foreground="white", borderwidth=0, font=("Segoe UI", 10, "bold"))
        style.map("Accent.TButton", background=[("active", self.colors["accent_hover"])])
        
        style.configure("TButton", background=self.colors["bg_panel"], foreground="white", borderwidth=1, font=("Segoe UI", 10))
        style.map("TButton", background=[("active", "#404040")])
        
        # Inputs
        style.configure("TEntry", fieldbackground="#404040", foreground="white", insertcolor="white")
        
    def _build_ui(self):
        # 1. Top Bar (Logo + Title)
        top_bar = ttk.Frame(self, height=60)
        top_bar.pack(side="top", fill="x", padx=20, pady=10)
        
        lbl_title = ttk.Label(top_bar, text="SPEEDGUN MOBILE PRO", style="Header.TLabel")
        lbl_title.pack(side="left")
        
        # 2. Main Container (Sidebar + Content)
        main_container = ttk.Frame(self)
        main_container.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        
        # --- LEFT SIDEBAR (Controls) ---
        sidebar = ttk.Frame(main_container, width=300, style="Panel.TFrame")
        sidebar.pack(side="left", fill="y", padx=(0, 20))
        
        # Inner padding for sidebar
        sidebar_content = ttk.Frame(sidebar, style="Panel.TFrame", padding=15)
        sidebar_content.pack(fill="both", expand=True)
        
        ttk.Label(sidebar_content, text="CONFIGURATION", style="SubHeader.TLabel").pack(anchor="w", pady=(0, 15))
        
        # File Selection
        ttk.Label(sidebar_content, text="Input Video", style="MetricLabel.TLabel").pack(anchor="w")
        self.btn_select = ttk.Button(sidebar_content, text="Select Video...", command=self.select_video)
        self.btn_select.pack(fill="x", pady=(5, 15))
        
        self.lbl_filename = ttk.Label(sidebar_content, text="No file selected", style="MetricLabel.TLabel", wraplength=250)
        self.lbl_filename.pack(anchor="w", pady=(0, 20))
        
        ttk.Separator(sidebar_content, orient="horizontal").pack(fill="x", pady=10)
        
        # Parameters
        ttk.Label(sidebar_content, text="Pitch Distance (m)", style="MetricLabel.TLabel").pack(anchor="w")
        entry_dist = ttk.Entry(sidebar_content, textvariable=self.pitch_dist)
        entry_dist.pack(fill="x", pady=(5, 15))
        
        ttk.Label(sidebar_content, text="YOLO Model", style="MetricLabel.TLabel").pack(anchor="w")
        entry_model = ttk.Entry(sidebar_content, textvariable=self.yolov8_weights)
        entry_model.pack(fill="x", pady=(5, 15))
        
        # Action Button
        self.btn_run = ttk.Button(sidebar_content, text="START ANALYSIS", style="Accent.TButton", command=self.run_analysis)
        self.btn_run.pack(fill="x", pady=(20, 0), ipady=10)
        
        # Status Bar in Sidebar
        self.lbl_status = ttk.Label(sidebar_content, textvariable=self.status_msg, style="MetricLabel.TLabel", wraplength=250)
        self.lbl_status.pack(side="bottom", anchor="w", pady=10)

        # --- RIGHT CONTENT (Dashboard) ---
        content_area = ttk.Frame(main_container)
        content_area.pack(side="left", fill="both", expand=True)
        
        # Top: Video Player & Key Metrics
        top_split = ttk.Frame(content_area)
        top_split.pack(fill="both", expand=True) # Takes remaining space
        
        # Video Player (Left of top split)
        video_panel = ttk.Frame(top_split, style="Panel.TFrame")
        video_panel.pack(side="left", fill="both", expand=True, padx=(0, 20))
        
        # Custom Canvas for video
        self.video_canvas = tk.Canvas(video_panel, bg="black", highlightthickness=0)
        self.video_canvas.pack(fill="both", expand=True, padx=2, pady=2)
        
        # Playback Controls
        self.controls_frame = ttk.Frame(video_panel, style="Panel.TFrame")
        self.controls_frame.pack(fill="x", pady=5)
        
        self.btn_play = ttk.Button(self.controls_frame, text="▶ PLAY VIDEO", command=self.play_video, state="disabled")
        self.btn_play.pack(side="left", padx=5)
        
        # Metrics Panel (Right of top split)
        metrics_panel = ttk.Frame(top_split, width=350, style="Panel.TFrame", padding=20)
        metrics_panel.pack(side="right", fill="y")
        
        self._create_metric_card(metrics_panel, "BALL SPEED", "result_release_speed", "0.0", "km/h", row=0, is_title_dynamic=True)
        self._create_metric_card(metrics_panel, "MAX SPEED", "result_max_speed", "0.0", "km/h", row=1)
        self._create_metric_card(metrics_panel, "DISTANCE", "result_distance", "0.00", "m", row=2)

        # Bottom: Charts
        charts_panel = ttk.Frame(content_area, height=300, style="Panel.TFrame", padding=10)
        charts_panel.pack(side="bottom", fill="x", pady=(20, 0)) # Fixed height
        
        self._setup_charts(charts_panel)

    def _create_metric_card(self, parent, title, var_name, default, unit, row, is_title_dynamic=False):
        """Helper to create a consistent metric display."""
        frame = ttk.Frame(parent, style="Panel.TFrame")
        frame.pack(fill="x", pady=(0, 25))
        
        # Initialize variable
        setattr(self, var_name, tk.StringVar(value=default))
        
        lbl = ttk.Label(frame, text=title, style="MetricLabel.TLabel")
        lbl.pack(anchor="w")
        
        if is_title_dynamic:
            self.lbl_speed_title = lbl
        
        # Value + Unit row
        val_row = ttk.Frame(frame, style="Panel.TFrame")
        val_row.pack(anchor="w")
        
        ttk.Label(val_row, textvariable=getattr(self, var_name), style="MetricVal.TLabel").pack(side="left")
        ttk.Label(val_row, text=unit, style="MetricLabel.TLabel", padding=(5, 0, 0, 5)).pack(side="left", anchor="sw")

    def _setup_charts(self, parent):
        """Embed Matplotlib figure."""
        self.fig, self.ax = plt.subplots(figsize=(8, 2), dpi=100)
        self.fig.patch.set_facecolor(self.colors["bg_panel"])
        self.ax.set_facecolor(self.colors["bg_panel"])
        
        # Style axis
        self.ax.tick_params(colors='white')
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.ax.spines['bottom'].set_color('#555555')
        self.ax.spines['top'].set_color('#555555') 
        self.ax.spines['left'].set_color('#555555')
        self.ax.spines['right'].set_color('#555555')
        
        self.ax.set_title("Velocity Breakdown (Time vs Speed)", color='white', fontsize=10)
        self.ax.set_xlabel("Frame", fontsize=8)
        self.ax.set_ylabel("Speed (km/h)", fontsize=8)
        
        self.chart_canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.chart_canvas.draw()
        self.chart_canvas.get_tk_widget().pack(fill="both", expand=True)

    def select_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.avi *.mov")])
        if path:
            self.video_paths = [path]
            self.lbl_filename.config(text=os.path.basename(path))
            self.status_msg.set("Video loaded. Click 'Start Analysis'.")
            
            # Show preview
            self._show_preview_frame(path)

    def _show_preview_frame(self, video_path):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if ret:
            self._update_video_canvas(frame)
        cap.release()

    def play_video(self):
        if not hasattr(self, 'current_output_path') or not self.current_output_path:
            return
            
        if getattr(self, 'is_playing', False):
            # Pause
            self.is_playing = False
            self.btn_play.config(text="▶ PLAY VIDEO")
            return
            
        # Start Playing
        self.is_playing = True
        self.btn_play.config(text="⏸ PAUSE")
        
        self.cap = cv2.VideoCapture(self.current_output_path)
        self._video_loop()
        
    def _video_loop(self):
        if not getattr(self, 'is_playing', False):
            return
            
        if not hasattr(self, 'cap') or not self.cap.isOpened():
            return
            
        ret, frame = self.cap.read()
        if ret:
            self._update_video_canvas(frame)
            # Approx 30 FPS = 33ms
            self.after(33, self._video_loop)
        else:
            # End of video
            self.is_playing = False
            self.btn_play.config(text="↺ REPLAY")
            self.cap.release()

    def _update_video_canvas(self, frame):
        # Resize to fit canvas
        canvas_w = self.video_canvas.winfo_width()
        canvas_h = self.video_canvas.winfo_height()
        
        if canvas_w < 10 or canvas_h < 10:
             # Wait for UI to render
             canvas_w, canvas_h = 800, 450
        
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        
        # Keep aspect ratio
        img_w, img_h = img_pil.size
        ratio = min(canvas_w/img_w, canvas_h/img_h)
        new_w = int(img_w * ratio)
        new_h = int(img_h * ratio)
        
        img_pil = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(img_pil)
        
        self.video_canvas.create_image(canvas_w//2, canvas_h//2, anchor="center", image=self.tk_image)

    def run_analysis(self):
        if not self.video_paths:
            messagebox.showwarning("Warning", "Please select a video first.")
            return
            
        self.pipeline_running = True
        self.btn_run.config(state="disabled", text="ANALYZING...")
        self.btn_select.config(state="disabled")
        self.btn_play.config(state="disabled") # Disable play during analysis
        self.status_msg.set("Initializing AI model...")
        
        # Start worker thread
        threading.Thread(target=self._worker_thread, daemon=True).start()

    def _worker_thread(self):
        try:
            output_filename = "Overlay_Pro.mp4"
            base_dir = os.path.dirname(self.video_paths[0])
            output_path = os.path.join(base_dir, output_filename)
            
            # Run Pipeline
            results = run_yolov8_pipeline(
                self.video_paths,
                weights_path=os.path.abspath(self.yolov8_weights.get()),
                output_path=output_path,
                manual_distance_meters=self.pitch_dist.get(),
                show_preview=False,
                debug=True 
            )
            
            if results and len(results) > 0:
                data = results[0]
                self.after(0, lambda: self._update_ui_with_results(data, output_path))
            else:
                self.after(0, lambda: messagebox.showwarning("No Data", "Could not detect enough ball trajectory."))
                
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            self.after(0, self._reset_ui_state)

    def _update_ui_with_results(self, data: dict, output_path: str):
        def fmt_speed(kmh):
            if kmh is None: return "N/A"
            mph = kmh * 0.621371
            return f"{kmh:.1f} km/h\n({mph:.1f} mph)"

        # 1. Ball Speed (Release or Initial)
        # Match overlay logic: prefer release, else initial
        rel_speed = data.get('release_speed_kmh')
        init_speed = data.get('initial_speed_kmh')
        
        if rel_speed:
            self.result_release_speed.set(fmt_speed(rel_speed))
            if hasattr(self, 'lbl_speed_title'):
                self.lbl_speed_title.config(text="BALL SPEED")
        elif init_speed:
            self.result_release_speed.set(fmt_speed(init_speed))
            if hasattr(self, 'lbl_speed_title'):
                self.lbl_speed_title.config(text="SPEED")
        else:
            self.result_release_speed.set("N/A")
        
        # 2. Max Speed
        self.result_max_speed.set(fmt_speed(data.get('max_speed_kmh', 0)))
        
        # 3. Distance
        dist = data.get('total_distance_m', 0)
        self.result_distance.set(f"{dist:.1f} m")
        
        # Update Chart
        details = data.get('frame_details', [])
        if details:
            frames = [d['frame'] for d in details]
            speeds = [d['speed_kmh'] for d in details]
            
            self.ax.clear()
            self.ax.set_facecolor(self.colors["bg_panel"])
            # We need to manually set the spine colors again after clearing
            self.ax.spines['bottom'].set_color('#555555')
            self.ax.spines['top'].set_color('#555555') 
            self.ax.spines['left'].set_color('#555555')
            self.ax.spines['right'].set_color('#555555')
            self.ax.tick_params(colors='white')
            self.ax.grid(True, color='#444444', linestyle='--')
            
            # Plot line
            self.ax.plot(frames, speeds, color=self.colors["accent"], linewidth=2, marker='o', markersize=4)
            self.ax.fill_between(frames, speeds, color=self.colors["accent"], alpha=0.1)
            
            self.ax.set_title("Velocity Breakdown", color='white')
            # Reset labels color if lost
            self.ax.xaxis.label.set_color('white')
            self.ax.yaxis.label.set_color('white')
            self.ax.set_xlabel("Frame Index", fontsize=8)
            self.ax.set_ylabel("Speed (km/h)", fontsize=8)
            
            self.chart_canvas.draw()
            
        self.status_msg.set(f"Analysis Complete! Output: {os.path.basename(output_path)}")
        
        # Enable Playback
        self.current_output_path = output_path
        self.btn_play.config(state="normal", text="▶ PLAY VIDEO")
        
        # Play the result video (first frame)
        self._show_preview_frame(output_path)

    def _reset_ui_state(self):
        self.pipeline_running = False
        self.btn_run.config(state="normal", text="START ANALYSIS")
        self.btn_select.config(state="normal")

if __name__ == "__main__":
    app = SpeedgunApp()
    app.mainloop()
