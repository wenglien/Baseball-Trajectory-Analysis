import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

from src.FrameInfo import FrameInfo
from src.utils import fill_lost_tracking

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def get_pitch_frames_yolov8(
    video_path: str,
    yolo_model: YOLO,
    # Default using low threshold, make the small ball easier to be detected; Can be overridden by external
    conf_threshold: float = 0.1,
    show_preview: bool = False,
):
    """
    使用 YOLOv8 模型偵測棒球，搭配 Mediapipe Pose + SORT 追蹤，
    輸出與原本 get_pitch_frames 類似的 pitch_frames 結構。
    """
    print("Video from: ", video_path)
    # Use OpenCV to read the video information (width, height, FPS), the actual frame is read by YOLOv8 later
    meta_cap = cv2.VideoCapture(video_path)
    if not meta_cap.isOpened():
        raise ValueError(
            f"無法開啟影片檔案：{video_path}\n請確認檔案格式是否支援（mp4/avi/mov/mkv）。"
        )

    width = int(meta_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(meta_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(meta_cap.get(cv2.CAP_PROP_FPS))
    meta_cap.release()

    if width <= 0 or height <= 0:
        raise ValueError(
            f"無法讀取影片尺寸，可能是檔案損壞或格式不支援：{video_path}"
        )
    if fps <= 0:
        fps = 30
        print("警告：無法讀取 fps，使用預設值 30")

    pitch_frames: list[FrameInfo] = []
    frame_id = 0
    # Record whether the "out-of-point" has been corrected by the hand joint
    first_release_adjusted = False

    # Debug statistics: the detection status of the whole video (for observing the performance of YOLO on the video)
    total_frames = 0
    frames_with_dets = 0
    total_dets = 0

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Use the YOLOv8 video streaming interface, the same as the way you tested the video with CLI
    results_generator = yolo_model.predict(
        source=video_path,
        conf=conf_threshold,
        iou=0.3,
        imgsz=1280,
        stream=True,
        verbose=False,
    )

    for result in results_generator:
        # The original image returned by YOLO is BGR
        frame_bgr = result.orig_img
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Mediapipe pose
        results = pose.process(frame_rgb)
        has_pose = results is not None and results.pose_landmarks is not None
        if has_pose:
            mp_drawing.draw_landmarks(
                frame_rgb,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
            )

        # Directly extract the detection box from the YOLO result (same as CLI)
        dets_list = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            score = float(box.conf[0].item())
            dets_list.append(np.array([x1, y1, x2, y2, score]))

        # Statistics the detection status
        total_frames += 1
        num_dets = len(dets_list)
        total_dets += num_dets
        if num_dets > 0:
            frames_with_dets += 1

            # Take the box with the highest confidence as the "only one ball"
            best_det = max(dets_list, key=lambda d: float(d[4]))
            x1, y1, x2, y2, score = best_det
            centerX = int((x1 + x2) / 2)
            centerY = int((y1 + y2) / 2)

            # Only adjust the out-of-point when the first time the ball is detected, and then use the trajectory of the ball itself
            if not first_release_adjusted and has_pose:
                image_h, image_w, _ = frame_rgb.shape
                wrist_indices = [15, 16]  # Left and right wrist
                wrist_points = []
                for idx in wrist_indices:
                    lm = results.pose_landmarks.landmark[idx]
                    wx = int(lm.x * image_w)
                    wy = int(lm.y * image_h)
                    wrist_points.append((wx, wy))

                if wrist_points:
                    # Select the wrist point closest to the ball as the out-of-point
                    wx, wy = min(
                        wrist_points,
                        key=lambda p: (p[0] - centerX) ** 2 + (p[1] - centerY) ** 2,
                    )
                    centerX, centerY = wx, wy
                    first_release_adjusted = True

            # Give the trajectory a fixed color (the overlay will use this color to draw the ball and trajectory)
            color = (255, 255, 0)
            pitch_frames.append(
                FrameInfo(frame_rgb, True, (centerX, centerY), color)
            )
        else:
            # Even if the ball is not detected, still store it in the list, so that the polyfit can be used to fill the missing points or at least keep the frame
            pitch_frames.append(FrameInfo(frame_rgb, False))

        if show_preview:
            vis = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            cv2.imshow("yolov8_result", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_id += 1

    print("Processing complete")
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

    avg_dets = total_dets / total_frames if total_frames > 0 else 0.0
    print(
        f"偵測統計：總 frame 數 = {total_frames}, "
        f"有偵測到至少一顆球的 frame 數 = {frames_with_dets}, "
        f"平均每 frame 偵測框數 = {avg_dets:.3f}"
    )

    fill_lost_tracking(pitch_frames)

    return pitch_frames, width, height, fps


    return pitch_frames, width, height, fps

