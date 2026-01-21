import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
from typing import Optional

from src.FrameInfo import FrameInfo
from src.utils import fill_lost_tracking
from src.ball_speed_calculator import BallSpeedCalculator
from src.release_point_detector import ReleasePointDetector

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def get_pitch_frames_yolov8(
    video_path: str,
    yolo_model: YOLO,
    conf_threshold: float = 0.15,
    show_preview: bool = False,
    speed_calculator: Optional[BallSpeedCalculator] = None,
) -> tuple[list[FrameInfo], int, int, int, dict]:
    """
    使用 YOLOv8 模型偵測棒球，搭配 Mediapipe Pose 追蹤，
    輸出與原本 get_pitch_frames 類似的 pitch_frames 結構。
    
    Args:
        video_path: 影片檔案路徑
        yolo_model: 已載入的 YOLOv8 模型
        conf_threshold: YOLOv8 置信度閾值（預設 0.15，建議 0.1-0.3 之間）
        show_preview: 是否顯示即時預覽視窗
        speed_calculator: 球速計算器（可選，用於計算球速）
    
    Returns:
        tuple: (pitch_frames, width, height, fps, speed_info)
            - pitch_frames: FrameInfo 列表，包含每一 frame 的球位置資訊
            - width: 影片寬度
            - height: 影片高度
            - fps: 影片幀率
            - speed_info: 球速資訊字典
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
    raw_detections: list = []  # 儲存原始偵測結果（尚未修正出球點）
    frame_id = 0
    release_point = None  # 記錄出手點
    
    # 初始化多訊號出球點檢測器
    release_detector = ReleasePointDetector(fps=fps)

    # Debug statistics
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

    try:
        # Use the YOLOv8 video streaming interface, the same as the way you tested the video with CLI
        results_generator = yolo_model.predict(
            source=video_path,
            conf=conf_threshold,
            iou=0.3,
            imgsz=1280,
            stream=True,
            verbose=False,
        )
    except Exception as e:
        raise RuntimeError(
            f"無法使用 YOLOv8 處理影片：{video_path}\n錯誤：{e}"
        ) from e

    for result in results_generator:
        # The original image returned by YOLO is BGR
        frame_bgr = result.orig_img
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Mediapipe pose
        results = pose.process(frame_rgb)
        has_pose = results is not None and results.pose_landmarks is not None
        if has_pose:
            # 添加到出球點檢測器
            release_detector.add_frame(results.pose_landmarks, width, height)
            
            mp_drawing.draw_landmarks(
                frame_rgb,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
            )
        else:
            # 沒有 pose，添加 None
            release_detector.add_frame(None, width, height)

        # Directly extract the detection box from the YOLO result
        dets_list = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            score = float(box.conf[0].item())
            dets_list.append(np.array([x1, y1, x2, y2, score]))

        # 儲存原始偵測資料（第一階段：收集數據）
        raw_detections.append({
            'frame_rgb': frame_rgb,
            'frame_id': frame_id,
            'dets_list': dets_list,
            'has_pose': has_pose,
            'pose_landmarks': results.pose_landmarks if has_pose else None
        })

        # Statistics
        total_frames += 1
        num_dets = len(dets_list)
        total_dets += num_dets
        if num_dets > 0:
            frames_with_dets += 1

        if show_preview:
            vis = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            cv2.imshow("yolov8_result", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_id += 1

    print("Processing complete - Phase 1: Data collection")
    
    # ===== 第二階段：執行多訊號檢測並生成軌跡 =====
    
    # 執行多訊號出球點檢測
    optimal_release_frame_idx = None
    release_detection = None
    
    if release_detector.frame_count >= 10:
        release_detection = release_detector.detect_release_point()
        
        if release_detection and release_detection['confidence'] > 0.3:
            optimal_release_frame_idx = release_detection['frame_idx']
            
            print(f"\n{'='*60}")
            print(f"🎯 多訊號出球點檢測結果")
            print(f"{'='*60}")
            print(f"  檢測幀索引: {optimal_release_frame_idx}")
            print(f"  信心度: {release_detection['confidence']:.2f}")
            
            signals = release_detection['signals']
            print(f"  訊號 S1 (手腕速度峰值): {signals['s1_wrist_speed']}")
            print(f"  訊號 S2 (肘部伸展): {signals['s2_elbow_extension']}")
            print(f"  訊號 S3 (前腳落地窗口): {signals['s3_foot_window']}")
            print(f"{'='*60}\n")
    
    # 第二階段：使用檢測結果處理每一幀
    print("Processing Phase 2: Applying release point detection")
    
    first_ball_frame_idx = None  # 第一次偵測到球的幀
    first_release_adjusted = False
    
    for frame_data in raw_detections:
        frame_rgb = frame_data['frame_rgb']
        frame_id = frame_data['frame_id']
        dets_list = frame_data['dets_list']
        has_pose = frame_data['has_pose']
        pose_landmarks = frame_data['pose_landmarks']
        
        num_dets = len(dets_list)
        
        if num_dets > 0:
            # Take the box with the highest confidence
            best_det = max(dets_list, key=lambda d: float(d[4]))
            x1, y1, x2, y2, score = best_det
            centerX = int((x1 + x2) / 2)
            centerY = int((y1 + y2) / 2)
            box_width = x2 - x1
            box_height = y2 - y1
            
            # 記錄第一次偵測到球的幀
            if first_ball_frame_idx is None:
                first_ball_frame_idx = frame_id
            
            # 驗證軌跡點的合理性（改進的驗證邏輯）
            is_valid_point = True
            rejection_reason = ""
            
            # 檢查偵測框大小
            min_box_size = 3  # 降低最小尺寸要求（從 5 降至 3）
            max_box_size = min(width, height) / 2.5  # 放寬最大尺寸（從 /3 改為 /2.5）
            if box_width < min_box_size or box_height < min_box_size:
                is_valid_point = False
                rejection_reason = f"偵測框太小 ({box_width:.0f}x{box_height:.0f})"
            elif box_width > max_box_size or box_height > max_box_size:
                is_valid_point = False
                rejection_reason = f"偵測框太大 ({box_width:.0f}x{box_height:.0f})"
            
            # 檢查與上一個點的關係
            if is_valid_point and len(pitch_frames) > 0:
                recent_valid_balls = []
                for pf in reversed(pitch_frames):
                    if pf.ball_in_frame:
                        recent_valid_balls.append(pf.ball)
                        if len(recent_valid_balls) >= 3:
                            break
                
                if recent_valid_balls:
                    last_valid_ball = recent_valid_balls[0]
                    distance = np.sqrt(
                        (centerX - last_valid_ball[0])**2 + 
                        (centerY - last_valid_ball[1])**2
                    )
                    
                    # 改進的距離驗證：考慮置信度和幀間時間
                    confidence_factor = min(score / 0.3, 1.5)  # 調整置信度因子
                    max_reasonable_distance = (width / 2.5) * confidence_factor  # 放寬距離限制
                    
                    # 如果是第一個點之後的點，更寬鬆一些
                    if len(recent_valid_balls) == 1:
                        max_reasonable_distance *= 1.5
                    
                    if distance > max_reasonable_distance:
                        is_valid_point = False
                        rejection_reason = f"移動距離異常 ({distance:.0f} 像素，置信度 {score:.2f})"
                    
                    # 方向檢查：只在有足夠歷史點時才進行
                    if is_valid_point and len(recent_valid_balls) >= 3:
                        # 使用最近 3 個點來判斷趨勢
                        prev_dx = recent_valid_balls[0][0] - recent_valid_balls[1][0]
                        prev_dy = recent_valid_balls[0][1] - recent_valid_balls[1][1]
                        curr_dx = centerX - recent_valid_balls[0][0]
                        curr_dy = centerY - recent_valid_balls[0][1]
                        
                        prev_len = np.sqrt(prev_dx**2 + prev_dy**2)
                        curr_len = np.sqrt(curr_dx**2 + curr_dy**2)
                        
                        if prev_len > 5 and curr_len > 5:  # 只在有明顯移動時才檢查方向
                            prev_dir = (prev_dx / prev_len, prev_dy / prev_len)
                            curr_dir = (curr_dx / curr_len, curr_dy / curr_len)
                            dot_product = prev_dir[0] * curr_dir[0] + prev_dir[1] * curr_dir[1]
                            
                            # 放寬方向檢查：從 -0.5 改為 -0.7（更容許方向變化）
                            if dot_product < -0.7:
                                is_valid_point = False
                                rejection_reason = f"移動方向異常（反向，點積 {dot_product:.2f}）"
            
            if not is_valid_point:
                if rejection_reason:
                    print(f"⚠ Frame {frame_id}: {rejection_reason}")
                pitch_frames.append(FrameInfo(frame_rgb, False))
                continue

            # ✨ 關鍵：記錄出球點（用於球速計算）但不修改球的實際位置
            if not first_release_adjusted:
                # 如果當前幀是最佳出球點，或者是第一個有球的幀
                should_record_release = False
                is_multi_signal = False
                
                if optimal_release_frame_idx is not None:
                    # 使用多訊號檢測的結果 - 但要確保這個幀有球
                    if frame_id == optimal_release_frame_idx:
                        should_record_release = True
                        is_multi_signal = True
                        print(f"✅ 檢測到出球點（幀 {frame_id}）")
                
                # 如果多訊號檢測的幀沒有球，或者沒有多訊號檢測結果
                # 則使用第一個有球的幀
                if not should_record_release and frame_id == first_ball_frame_idx:
                    should_record_release = True
                    print(f"⚠️ 使用第一個球偵測點作為出球點（幀 {frame_id}）")
                
                if should_record_release and has_pose:
                    image_h, image_w, _ = frame_rgb.shape
                    
                    # 優先使用食指指尖作為放球點
                    finger_indices = [19, 20]  # 左右手食指指尖
                    finger_points = []
                    
                    for idx in finger_indices:
                        lm = pose_landmarks.landmark[idx]
                        visibility = lm.visibility if hasattr(lm, 'visibility') else 1.0
                        
                        if visibility > 0.5:
                            fx = int(lm.x * image_w)
                            fy = int(lm.y * image_h)
                            finger_points.append((fx, fy))
                    
                    # 降級策略：如果食指不可見，使用手腕
                    if not finger_points:
                        wrist_indices = [15, 16]  # 左右手腕
                        for idx in wrist_indices:
                            lm = pose_landmarks.landmark[idx]
                            visibility = lm.visibility if hasattr(lm, 'visibility') else 1.0
                            if visibility > 0.3:  # 降低可見度要求
                                wx = int(lm.x * image_w)
                                wy = int(lm.y * image_h)
                                finger_points.append((wx, wy))
                    
                    if finger_points:
                        # 選擇最接近球的手指/手腕點
                        fx, fy = min(
                            finger_points,
                            key=lambda p: (p[0] - centerX) ** 2 + (p[1] - centerY) ** 2,
                        )
                        
                        # 驗證：出球點和球的距離必須在合理範圍內
                        distance_to_ball = np.sqrt((fx - centerX)**2 + (fy - centerY)**2)
                        max_reasonable_distance = width / 3  # 最多畫面寬度的 1/3
                        
                        if distance_to_ball <= max_reasonable_distance:
                            release_point = (fx, fy)
                            first_release_adjusted = True
                            print(f"   出球點座標: ({fx}, {fy})")
                            print(f"   球偵測座標: ({centerX}, {centerY})")
                            print(f"   距離: {distance_to_ball:.1f} 像素")
                        else:
                            print(f"   ⚠️ 手指位置 ({fx}, {fy}) 距離球太遠 ({distance_to_ball:.0f} 像素)")
                            print(f"   ⚠️ 使用球的偵測位置 ({centerX}, {centerY}) 作為出球點")
                            # 退回使用球的位置作為出球點
                            release_point = (centerX, centerY)
                            first_release_adjusted = True
                        # ⚠️ 重要：不要修改 centerX, centerY！保持球的實際偵測位置
                    else:
                        # 沒有找到手指/手腕，使用球的位置
                        print(f"   ⚠️ 無法偵測手指/手腕，使用球的位置作為出球點")
                        release_point = (centerX, centerY)
                        first_release_adjusted = True
                elif should_record_release and not has_pose:
                    # 沒有 pose 資料，直接使用球的位置
                    print(f"   ⚠️ 無姿態資料，使用球的位置作為出球點")
                    release_point = (centerX, centerY)
                    first_release_adjusted = True

            # Give the trajectory a fixed color
            color = (255, 255, 0)
            pitch_frames.append(
                FrameInfo(frame_rgb, True, (centerX, centerY), color)
            )
        else:
            pitch_frames.append(FrameInfo(frame_rgb, False))
    # 計算球速
    speed_info = {}
    if speed_calculator and len(pitch_frames) > 0:
        # 提取所有有球的 frame 的座標
        ball_trajectory = [
            frame.ball for frame in pitch_frames 
            if frame.ball_in_frame
        ]
        
        if len(ball_trajectory) >= 2:
            speed_info = speed_calculator.calculate_speed_detailed(
                ball_trajectory,
                release_point=release_point
            )
            
            # 添加 release_point 到 speed_info 以便在 overlay 中繪製
            if release_point:
                speed_info['release_point'] = release_point
            
            # 顯示計算結果
            if speed_info and not speed_info.get('error'):
                calc_method = speed_info.get('calculation_method', 'unknown')
                
                # 轉換函數
                def kmh_to_mph(kmh):
                    return kmh * 0.621371 if kmh else None
                
                print(f"\n{'='*60}")
                print(f"球速測定結果")
                if calc_method == 'theoretical':
                    print(f"（基於輸入距離計算）")
                print(f"{'='*60}")
                
                if speed_info.get('release_speed_kmh'):
                    kmh = speed_info['release_speed_kmh']
                    mph = kmh_to_mph(kmh)
                    print(f"  出手球速: {kmh:.1f} km/h ({mph:.1f} mph) ⚾")
                
                if speed_info.get('initial_speed_kmh'):
                    kmh = speed_info['initial_speed_kmh']
                    mph = kmh_to_mph(kmh)
                    print(f"  初速度:   {kmh:.1f} km/h ({mph:.1f} mph)")
                
                if speed_info.get('max_speed_kmh'):
                    kmh = speed_info['max_speed_kmh']
                    mph = kmh_to_mph(kmh)
                    print(f"  最大速度: {kmh:.1f} km/h ({mph:.1f} mph)")
                
                if speed_info.get('average_speed_kmh'):
                    kmh = speed_info['average_speed_kmh']
                    mph = kmh_to_mph(kmh)
                    print(f"  平均速度: {kmh:.1f} km/h ({mph:.1f} mph)")
                
                if speed_info.get('total_distance_m'):
                    print(f"  飛行距離: {speed_info['total_distance_m']:.2f} m")
                
                if speed_info.get('num_frames'):
                    print(f"  追蹤幀數: {speed_info['num_frames']} frames")
                
                print(f"{'='*60}\n")

    return pitch_frames, width, height, fps, speed_info

