import numpy as np
from typing import List, Tuple, Optional, Dict


class ReleasePointDetector:
    
    def __init__(self, fps: int = 30):
        self.fps = fps
        self.pose_history = []  # 儲存每一幀的 pose landmarks
        self.frame_count = 0
        self._cached_throwing_wrist_idx: Optional[int] = None
        
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

        # pose history 更新後，丟棄快取（避免資料增加造成快取過期）
        self._cached_throwing_wrist_idx = None

    def infer_throwing_hand(self) -> Optional[Dict[str, int]]:
        """
        推斷投球手（左/右）並回傳該手的關鍵點 index。

        Returns:
            dict: {'wrist': 15/16, 'index_finger': 19/20, 'elbow': 13/14, 'shoulder': 11/12}
        """
        wrist_idx = self._infer_throwing_wrist_idx()
        if wrist_idx is None:
            return None
        if wrist_idx == 15:
            return {"wrist": 15, "index_finger": 19, "elbow": 13, "shoulder": 11}
        return {"wrist": 16, "index_finger": 20, "elbow": 14, "shoulder": 12}

    def _infer_throwing_wrist_idx(self) -> Optional[int]:
        """
        以「可見度覆蓋率 + 高分位數腕速」推斷投球手，避免逐幀用可見度切換造成左右手跳動。
        """
        if self._cached_throwing_wrist_idx is not None:
            return self._cached_throwing_wrist_idx

        if len(self.pose_history) < 8:
            return None

        def build_series(wrist_idx: int) -> tuple[list[Tuple[float, float]], list[int], float]:
            pts: list[Tuple[float, float]] = []
            frame_ids: list[int] = []
            for i, frame_data in enumerate(self.pose_history):
                if frame_data is None:
                    continue
                w = frame_data.get(wrist_idx)
                if not w:
                    continue
                if w.get("visibility", 1.0) < 0.35:
                    continue
                pts.append((float(w["x"]), float(w["y"])))
                frame_ids.append(i)
            visible_ratio = (len(frame_ids) / len(self.pose_history)) if self.pose_history else 0.0
            return pts, frame_ids, float(visible_ratio)

        def score_wrist(wrist_idx: int) -> float:
            pts, frame_ids, visible_ratio = build_series(wrist_idx)
            if len(pts) < 6:
                return -1e9
            vel_window = max(3, int(round(self.fps / 10)))
            vels = self._calculate_velocity(pts, window=vel_window)
            # 使用高分位數避免單幀雜訊峰值
            peak = float(np.percentile(vels, 90)) if vels else 0.0
            return peak * (0.5 + visible_ratio)

        left_score = score_wrist(15)
        right_score = score_wrist(16)

        if left_score <= -1e8 and right_score <= -1e8:
            return None

        self._cached_throwing_wrist_idx = 15 if left_score >= right_score else 16
        return self._cached_throwing_wrist_idx
    
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
        訊號 S1：手腕速度峰值附近（更接近真實放球瞬間）

        以前版本用「峰值後的減速拐點」會系統性偏晚（尤其在高 FPS）。
        這裡改為直接取速度峰值（並允許依 FPS 做輕微前移校正）。

        Returns:
            出球幀索引，如果找不到則返回 None
        """
        if len(self.pose_history) < 10:
            return None

        # 提取手腕位置（固定投球手，避免左右切換）
        throwing = self.infer_throwing_hand()
        if not throwing:
            return None
        wrist_idx = throwing["wrist"]

        wrist_points = []
        for frame_data in self.pose_history:
            if frame_data is None:
                wrist_points.append(None)
                continue

            wrist = frame_data.get(wrist_idx)
            if not wrist or wrist.get("visibility", 1.0) < 0.35:
                wrist_points.append(None)
                continue
            wrist_points.append((float(wrist["x"]), float(wrist["y"])))

        # 過濾 None
        valid_indices = [i for i, p in enumerate(wrist_points) if p is not None]
        if len(valid_indices) < 10:
            return None

        valid_points = [wrist_points[i] for i in valid_indices]

        # 計算速度（平滑窗口隨 FPS 縮放：30fps≈3, 60fps≈6, 120fps≈12）
        vel_window = max(3, int(round(self.fps / 10)))
        velocities = self._calculate_velocity(valid_points, window=vel_window)

        # 找速度峰值
        peak_local_idx = int(np.argmax(velocities))

        # 高 FPS 下，真實放球通常更接近峰值或峰值前 0~2 幀
        # 以「秒」表示的前移量，讓不同 FPS 表現一致
        advance_sec = 0.008  # 8ms（120fps 約 -1 幀；60fps 約 -0~ -1 幀；30fps 約 0 幀）
        offset = -int(round(advance_sec * self.fps))

        chosen_local_idx = max(0, min(len(valid_indices) - 1, peak_local_idx + offset))
        return int(valid_indices[chosen_local_idx])


    def _detect_signal_s2_elbow_extension(self) -> Optional[int]:
        """
        訊號 S2：肘關節伸直 + 手腕鞭打動作
        
        Returns:
            出球幀索引
        """
        if len(self.pose_history) < 10:
            return None
        
        # 提取肩-肘-腕的角度和手腕相對位置（固定投球手）
        throwing = self.infer_throwing_hand()
        if not throwing:
            return None
        shoulder_idx = throwing["shoulder"]
        elbow_idx = throwing["elbow"]
        wrist_idx = throwing["wrist"]

        elbow_angles = []
        wrist_velocities = []
        valid_indices = []
        
        for i, frame_data in enumerate(self.pose_history):
            if frame_data is None:
                continue
            
            shoulder = frame_data.get(shoulder_idx)
            elbow = frame_data.get(elbow_idx)
            wrist = frame_data.get(wrist_idx)
            if not shoulder or not elbow or not wrist:
                continue
            if min(
                shoulder.get("visibility", 1.0),
                elbow.get("visibility", 1.0),
                wrist.get("visibility", 1.0),
            ) < 0.35:
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
        
        # 提取前腳踝位置：用「總位移較大」的腳當前腳，避免固定用左腳造成誤判
        left_series: list[Tuple[int, Tuple[float, float]]] = []
        right_series: list[Tuple[int, Tuple[float, float]]] = []
        
        for i, frame_data in enumerate(self.pose_history):
            if frame_data is None:
                continue
            
            left_ankle = frame_data.get(27)
            right_ankle = frame_data.get(28)

            if left_ankle and left_ankle.get("visibility", 1.0) >= 0.35:
                left_series.append((i, (float(left_ankle["x"]), float(left_ankle["y"]))))
            if right_ankle and right_ankle.get("visibility", 1.0) >= 0.35:
                right_series.append((i, (float(right_ankle["x"]), float(right_ankle["y"]))))

        def total_displacement(series: list[Tuple[int, Tuple[float, float]]]) -> float:
            if len(series) < 6:
                return 0.0
            pts = [p for _, p in series]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            return float((max(xs) - min(xs)) + (max(ys) - min(ys)))

        use_left = total_displacement(left_series) >= total_displacement(right_series)
        series = left_series if use_left else right_series

        if len(series) < 10:
            return None

        valid_indices = [fid for fid, _ in series]
        ankle_points = [pt for _, pt in series]

        # 計算 Y 方向速度（垂直方向）
        y_positions = [p[1] for p in ankle_points]
        y_velocities = []
        for i in range(1, len(y_positions)):
            y_velocities.append(abs(y_positions[i] - y_positions[i-1]))
        
        # 找前腳落地：Y 方向速度突然接近 0 並持續
        foot_contact_idx = None
        velocity_threshold = np.mean(y_velocities) * 0.25
        
        for i in range(len(y_velocities) - 3):
            # 連續 3 幀速度都很小
            if (y_velocities[i] < velocity_threshold and 
                y_velocities[i+1] < velocity_threshold and 
                y_velocities[i+2] < velocity_threshold):
                foot_contact_idx = valid_indices[i]
                break
        
        if foot_contact_idx is not None:
            # 出球通常在落地後約 0.10s ~ 0.67s（用 FPS 換算成幀數，避免高 FPS 錯位）
            start_delay_sec = 0.10
            end_delay_sec = 0.67
            window_start = foot_contact_idx + int(round(start_delay_sec * self.fps))
            window_end = min(foot_contact_idx + int(round(end_delay_sec * self.fps)), len(self.pose_history) - 1)
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
