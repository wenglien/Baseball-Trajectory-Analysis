import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import deque


class ReleasePointDetector:
    
    def __init__(self, fps: int = 30):
        self.fps = fps
        self.pose_history = []  # 儲存每一幀的 pose landmarks
        self.frame_count = 0
        
    def add_frame(self, pose_landmarks, frame_width: int, frame_height: int) -> None:
        """
        添加一幀的 pose 資料
        
        Args:
            pose_landmarks: MediaPipe pose landmarks
            frame_width: 畫面寬度
            frame_height: 畫面高度
        """
        if pose_landmarks is None:
            self.pose_history.append(None)
            self.frame_count += 1
            return
        
        # 提取關鍵點座標
        landmarks = {}
        for idx in [11, 12, 13, 14, 15, 16, 19, 20, 27, 28]:  # 肩、肘、腕、指、踝
            lm = pose_landmarks.landmark[idx]
            landmarks[idx] = {
                'x': lm.x * frame_width,
                'y': lm.y * frame_height,
                'visibility': lm.visibility if hasattr(lm, 'visibility') else 1.0
            }
        
        self.pose_history.append(landmarks)
        self.frame_count += 1
    
    def _calculate_velocity(self, points: List[Tuple[float, float]], window: int = 3) -> List[float]:
        """
        計算速度（使用中心差分）
        
        Args:
            points: 位置點列表 [(x, y), ...]
            window: 平滑窗口
            
        Returns:
            速度列表（像素/幀）
        """
        if len(points) < 2:
            return [0.0] * len(points)
        
        velocities = []
        for i in range(len(points)):
            if i == 0:
                # 前向差分
                dx = points[i+1][0] - points[i][0]
                dy = points[i+1][1] - points[i][1]
            elif i == len(points) - 1:
                # 後向差分
                dx = points[i][0] - points[i-1][0]
                dy = points[i][1] - points[i-1][1]
            else:
                # 中心差分（更準確）
                dx = (points[i+1][0] - points[i-1][0]) / 2
                dy = (points[i+1][1] - points[i-1][1]) / 2
            
            v = np.sqrt(dx**2 + dy**2)
            velocities.append(v)
        
        # 移動平均平滑
        smoothed = []
        for i in range(len(velocities)):
            start = max(0, i - window//2)
            end = min(len(velocities), i + window//2 + 1)
            smoothed.append(np.mean(velocities[start:end]))
        
        return smoothed
    
    def _calculate_angle(self, p1: Tuple[float, float], p2: Tuple[float, float], 
                        p3: Tuple[float, float]) -> float:
        """
        計算三點形成的角度（p1-p2-p3）
        
        Returns:
            角度（度）
        """
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle) * 180 / np.pi
        
        return angle
    
    def _detect_signal_s1_wrist_speed_peak(self) -> Optional[int]:
        """
        訊號 S1：手腕速度峰值後的第一個減速點
        
        Returns:
            出球幀索引，如果找不到則返回 None
        """
        if len(self.pose_history) < 10:
            return None
        
        # 提取手腕位置（選擇可見度最高的手）
        wrist_points = []
        for frame_data in self.pose_history:
            if frame_data is None:
                wrist_points.append(None)
                continue
            
            # 選擇左右手腕中可見度較高的
            left_wrist = frame_data.get(15)
            right_wrist = frame_data.get(16)
            
            if left_wrist and right_wrist:
                wrist = left_wrist if left_wrist['visibility'] > right_wrist['visibility'] else right_wrist
            elif left_wrist:
                wrist = left_wrist
            elif right_wrist:
                wrist = right_wrist
            else:
                wrist_points.append(None)
                continue
            
            wrist_points.append((wrist['x'], wrist['y']))
        
        # 過濾 None
        valid_indices = [i for i, p in enumerate(wrist_points) if p is not None]
        if len(valid_indices) < 10:
            return None
        
        valid_points = [wrist_points[i] for i in valid_indices]
        
        # 計算速度
        velocities = self._calculate_velocity(valid_points, window=3)
        
        # 找速度峰值
        peak_idx = np.argmax(velocities)
        peak_value = velocities[peak_idx]
        
        # 在峰值後找第一個減速拐點（速度下降超過 20%）
        threshold = peak_value * 0.8
        for i in range(peak_idx + 1, len(velocities) - 2):
            if velocities[i] < threshold:
                # 檢查是否持續減速（連續 2-3 幀）
                if i + 2 < len(velocities):
                    if velocities[i+1] < velocities[i] and velocities[i+2] < velocities[i+1]:
                        return valid_indices[i]
        
        # 如果沒找到明顯拐點，返回峰值後 2-3 幀
        if peak_idx + 3 < len(valid_indices):
            return valid_indices[peak_idx + 2]
        
        return None
    
    def _detect_signal_s2_elbow_extension(self) -> Optional[int]:
        """
        訊號 S2：肘關節伸直 + 手腕鞭打動作
        
        Returns:
            出球幀索引
        """
        if len(self.pose_history) < 10:
            return None
        
        # 提取肩-肘-腕的角度和手腕相對位置
        elbow_angles = []
        wrist_velocities = []
        valid_indices = []
        
        for i, frame_data in enumerate(self.pose_history):
            if frame_data is None:
                continue
            
            # 選擇投球手（可見度較高的）
            left_shoulder = frame_data.get(11)
            right_shoulder = frame_data.get(12)
            left_elbow = frame_data.get(13)
            right_elbow = frame_data.get(14)
            left_wrist = frame_data.get(15)
            right_wrist = frame_data.get(16)
            
            # 判斷投球手（通常是舉高的那隻手）
            if left_shoulder and left_elbow and left_wrist:
                if right_shoulder and right_elbow and right_wrist:
                    # 選擇 Y 座標較小（較高）的手
                    if left_wrist['y'] < right_wrist['y']:
                        shoulder, elbow, wrist = left_shoulder, left_elbow, left_wrist
                    else:
                        shoulder, elbow, wrist = right_shoulder, right_elbow, right_wrist
                else:
                    shoulder, elbow, wrist = left_shoulder, left_elbow, left_wrist
            elif right_shoulder and right_elbow and right_wrist:
                shoulder, elbow, wrist = right_shoulder, right_elbow, right_wrist
            else:
                continue
            
            # 計算肘關節角度
            angle = self._calculate_angle(
                (shoulder['x'], shoulder['y']),
                (elbow['x'], elbow['y']),
                (wrist['x'], wrist['y'])
            )
            
            elbow_angles.append(angle)
            wrist_velocities.append((wrist['x'], wrist['y']))
            valid_indices.append(i)
        
        if len(elbow_angles) < 5:
            return None
        
        # 計算手腕相對速度
        wrist_vels = self._calculate_velocity(wrist_velocities, window=2)
        
        # 找肘關節接近伸直（角度接近最大）且手腕速度高的時刻
        # 肘關節伸直通常是 140-180 度
        max_angle = max(elbow_angles)
        candidates = []
        
        for i in range(len(elbow_angles)):
            if elbow_angles[i] > max_angle * 0.9:  # 接近最大伸展
                if wrist_vels[i] > np.mean(wrist_vels) * 1.5:  # 手腕速度較高
                    candidates.append((i, wrist_vels[i]))
        
        if candidates:
            # 選擇手腕速度最高的候選點
            best_idx = max(candidates, key=lambda x: x[1])[0]
            return valid_indices[best_idx]
        
        return None
    
    def _detect_signal_s3_foot_contact(self) -> Optional[Tuple[int, int]]:
        """
        訊號 S3：前腳落地檢測，返回時間窗口
        
        Returns:
            (落地幀, 窗口結束幀) 或 None
        """
        if len(self.pose_history) < 10:
            return None
        
        # 提取前腳踝位置（選擇移動較多的腳）
        ankle_points = []
        valid_indices = []
        
        for i, frame_data in enumerate(self.pose_history):
            if frame_data is None:
                continue
            
            left_ankle = frame_data.get(27)
            right_ankle = frame_data.get(28)
            
            if left_ankle and right_ankle:
                # 選擇移動較多的腳（前腳）
                ankle = left_ankle  # 簡化：先用左腳
            elif left_ankle:
                ankle = left_ankle
            elif right_ankle:
                ankle = right_ankle
            else:
                continue
            
            ankle_points.append((ankle['x'], ankle['y']))
            valid_indices.append(i)
        
        if len(ankle_points) < 10:
            return None
        
        # 計算 Y 方向速度（垂直方向）
        y_positions = [p[1] for p in ankle_points]
        y_velocities = []
        for i in range(1, len(y_positions)):
            y_velocities.append(abs(y_positions[i] - y_positions[i-1]))
        
        # 找前腳落地：Y 方向速度突然接近 0 並持續
        foot_contact_idx = None
        velocity_threshold = np.mean(y_velocities) * 0.3
        
        for i in range(len(y_velocities) - 3):
            # 連續 3 幀速度都很小
            if (y_velocities[i] < velocity_threshold and 
                y_velocities[i+1] < velocity_threshold and 
                y_velocities[i+2] < velocity_threshold):
                foot_contact_idx = valid_indices[i]
                break
        
        if foot_contact_idx is not None:
            # 出球通常在落地後 3-20 幀
            window_start = foot_contact_idx + 3
            window_end = min(foot_contact_idx + 20, len(self.pose_history) - 1)
            return (window_start, window_end)
        
        return None
    
    def detect_release_point(self) -> Optional[Dict]:
        if len(self.pose_history) < 10:
            return None
        
        # 檢測各個訊號
        s1_frame = self._detect_signal_s1_wrist_speed_peak()
        s2_frame = self._detect_signal_s2_elbow_extension()
        s3_window = self._detect_signal_s3_foot_contact()
        
        signals = {
            's1_wrist_speed': s1_frame,
            's2_elbow_extension': s2_frame,
            's3_foot_window': s3_window
        }
        
        # 融合策略
        candidates = []
        
        # S1 權重 0.4
        if s1_frame is not None:
            candidates.append((s1_frame, 0.4))
        
        # S2 權重 0.3
        if s2_frame is not None:
            candidates.append((s2_frame, 0.3))
        
        # S3 約束：如果有落地窗口，只考慮窗口內的候選
        if s3_window is not None:
            window_start, window_end = s3_window
            filtered_candidates = [
                (frame, weight) for frame, weight in candidates
                if window_start <= frame <= window_end
            ]
            
            if filtered_candidates:
                candidates = filtered_candidates
                # S3 約束成功，增加信心度
                for i in range(len(candidates)):
                    frame, weight = candidates[i]
                    candidates[i] = (frame, weight * 1.2)
        
        if not candidates:
            # 沒有任何訊號，退回到簡單的第一幀有球的點
            return None
        
        # 加權平均（取最接近加權中心的候選點）
        weighted_sum = sum(frame * weight for frame, weight in candidates)
        total_weight = sum(weight for _, weight in candidates)
        
        if total_weight == 0:
            return None
        
        target_frame = weighted_sum / total_weight
        
        # 選擇最接近加權中心的候選點
        best_frame = min(candidates, key=lambda x: abs(x[0] - target_frame))[0]
        
        # 計算信心度
        confidence = total_weight / 1.0  # 最大權重和為 0.4 + 0.3 + S3加成
        confidence = min(confidence, 1.0)
        
        return {
            'frame_idx': best_frame,
            'confidence': confidence,
            'signals': signals
        }
