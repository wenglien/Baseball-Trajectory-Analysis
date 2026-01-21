"""
棒球球速計算模組
包含相機校正、透視校正和精確的球速計算
"""
import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict


class BallSpeedCalculator:
    """精確的球速計算器，考慮透視變形和深度因素"""
    
    def __init__(
        self, 
        fps: int,
        video_width: int,
        video_height: int,
        pixels_per_meter: Optional[float] = None,
        theoretical_distance: Optional[float] = None
    ):
        """
        Args:
            fps: 影片幀率
            video_width: 影片寬度
            video_height: 影片高度
            pixels_per_meter: 每米多少像素（需要校正得出）
            theoretical_distance: 理論距離（手動輸入時使用，例如 18.44m）
        """
        self.fps = fps
        self.video_width = video_width
        self.video_height = video_height
        self.pixels_per_meter = pixels_per_meter
        self.perspective_matrix = None
        self.near_y = None
        self.far_y = None
        self.theoretical_distance = theoretical_distance
        
    def calibrate_from_reference(
        self, 
        pitcher_mound_pixel: Tuple[int, int],
        home_plate_pixel: Tuple[int, int],
        real_distance: float = 18.44
    ) -> None:
        """
        使用已知的投手板到本壘板距離進行校正
        
        Args:
            pitcher_mound_pixel: 投手板的像素座標 (x, y)
            home_plate_pixel: 本壘板的像素座標 (x, y)
            real_distance: 真實距離（米），標準棒球場為 18.44m
        """
        pixel_distance = np.sqrt(
            (home_plate_pixel[0] - pitcher_mound_pixel[0])**2 + 
            (home_plate_pixel[1] - pitcher_mound_pixel[1])**2
        )
        self.pixels_per_meter = pixel_distance / real_distance
        
        # 計算透視校正參數（基於 Y 座標的深度因素）
        self._calculate_perspective_factors(pitcher_mound_pixel, home_plate_pixel)
        
        print(f"✓ 校正完成：每米 = {self.pixels_per_meter:.2f} 像素")
        
    def _calculate_perspective_factors(
        self,
        near_point: Tuple[int, int],
        far_point: Tuple[int, int]
    ) -> None:
        """
        計算透視校正因子
        近處的物體移動相同距離，像素變化較大
        遠處的物體移動相同距離，像素變化較小
        """
        # 簡化的透視模型：基於 Y 座標的線性插值
        self.near_y = min(near_point[1], far_point[1])
        self.far_y = max(near_point[1], far_point[1])
        
    def _apply_perspective_correction(self, point: Tuple[int, int]) -> float:
        """
        根據點的 Y 座標應用透視校正因子
        
        Returns:
            校正後的比例因子
        """
        if self.near_y is None or self.far_y is None:
            return 1.0
        
        # Y 座標越大（畫面下方），物體越近，需要的校正因子越小
        # Y 座標越小（畫面上方），物體越遠，需要的校正因子越大
        y = point[1]
        
        if self.far_y == self.near_y:
            return 1.0
        
        # 線性插值：近處 0.85，遠處 1.15
        t = (y - self.near_y) / (self.far_y - self.near_y)
        t = np.clip(t, 0, 1)
        correction_factor = 1.15 - (0.3 * t)  # 從 1.15 到 0.85
        
        return correction_factor
    
    def calculate_release_speed(
        self,
        release_point: Tuple[int, int],
        first_ball_point: Tuple[int, int],
        frames_elapsed: float = 2.5
    ) -> Optional[float]:
        """
        計算出手球速（從投手手腕到第一個偵測點）
        這是最準確的初速度測量
        
        Args:
            release_point: 出手點（投手手腕位置）
            first_ball_point: 第一個偵測到球的位置
            frames_elapsed: 從出手到第一次偵測的估計幀數
            
        Returns:
            出手球速（km/h），如果無法計算則返回 None
        """
        if self.pixels_per_meter is None:
            return None
        
        # 計算像素距離
        pixel_dist = np.sqrt(
            (first_ball_point[0] - release_point[0])**2 + 
            (first_ball_point[1] - release_point[1])**2
        )
        
        # 驗證距離合理性（避免異常的出球點導致錯誤計算）
        # 如果距離太小（< 10 像素），可能是同一個點，不計算
        if pixel_dist < 10:
            return None
        
        # 如果距離太大（> 畫面寬度的一半），可能是錯誤的檢測
        max_reasonable_pixel_dist = self.video_width / 2
        if pixel_dist > max_reasonable_pixel_dist:
            return None
        
        # 應用透視校正
        mid_point_y = (release_point[1] + first_ball_point[1]) / 2
        correction = self._apply_perspective_correction((0, int(mid_point_y)))
        
        # 轉換為真實距離（米）
        real_dist = (pixel_dist / self.pixels_per_meter) * correction
        
        # 計算時間
        time = frames_elapsed / self.fps
        
        # 計算速度
        speed_ms = real_dist / time
        speed_kmh = speed_ms * 3.6
        
        # 驗證速度合理性（職業投手最快約 170 km/h，業餘通常 < 150 km/h）
        # 如果超過 250 km/h，很可能是計算錯誤
        if speed_kmh > 250:
            return None
        
        return speed_kmh
    
    def calculate_speed_detailed(
        self, 
        trajectory_points: List[Tuple[int, int]],
        release_point: Optional[Tuple[int, int]] = None
    ) -> Dict:
        """
        計算詳細的球速資訊
        
        Args:
            trajectory_points: 球的軌跡點列表 [(x1, y1), (x2, y2), ...]
            release_point: 出手點（投手手腕位置），可選
            
        Returns:
            包含各種球速資訊的字典
        """
        if self.pixels_per_meter is None:
            return {"error": "請先執行 calibrate_from_reference 進行校正"}
        
        if len(trajectory_points) < 2:
            return {"error": "軌跡點數不足（至少需要 2 個點）"}
        
        # 如果有理論距離，使用簡化的計算方式
        if self.theoretical_distance:
            # 使用理論距離和追蹤幀數計算平均球速
            num_frames = len(trajectory_points)
            total_time = num_frames / self.fps  # 總時間（秒）
            
            # 平均速度 = 距離 / 時間
            avg_speed_ms = self.theoretical_distance / total_time
            avg_speed_kmh = avg_speed_ms * 3.6
            
            # 假設初速度和最大速度略高於平均速度（基於物理）
            # 球在飛行中會減速，所以初速 > 平均速 > 末速
            initial_speed = avg_speed_kmh * 1.1  # 初速約為平均速的 110%
            max_speed = avg_speed_kmh * 1.15     # 最大速約為平均速的 115%
            
            # 計算出手球速（如果有 release_point）
            release_speed = None
            if release_point and len(trajectory_points) > 0:
                release_speed = self.calculate_release_speed(
                    release_point, 
                    trajectory_points[0],
                    frames_elapsed=2.5
                )
            
            return {
                'release_speed_kmh': release_speed,      # 出手球速
                'initial_speed_kmh': initial_speed,      # 初速度（估算）
                'max_speed_kmh': max_speed,              # 最大速度（估算）
                'average_speed_kmh': avg_speed_kmh,      # 平均速度（基於理論距離）
                'total_distance_m': self.theoretical_distance,  # 理論距離
                'num_frames': num_frames,                # 追蹤的 frame 數
                'calculation_method': 'theoretical'      # 計算方法標記
            }
        
        # 原本的像素計算方式（沒有理論距離時使用）
        speeds = []
        time_interval = 1.0 / self.fps
        
        # 計算每一段的速度
        for i in range(len(trajectory_points) - 1):
            p1 = trajectory_points[i]
            p2 = trajectory_points[i + 1]
            
            # 計算像素距離
            pixel_dist = np.sqrt(
                (p2[0] - p1[0])**2 + 
                (p2[1] - p1[1])**2
            )
            
            # 應用透視校正（使用兩點中點的 Y 座標）
            mid_y = (p1[1] + p2[1]) / 2
            correction = self._apply_perspective_correction((0, int(mid_y)))
            
            # 轉換為真實距離（米）
            real_dist = (pixel_dist / self.pixels_per_meter) * correction
            
            # 計算速度
            speed_ms = real_dist / time_interval
            speed_kmh = speed_ms * 3.6
            
            speeds.append({
                'frame': i,
                'speed_kmh': speed_kmh,
                'speed_ms': speed_ms,
                'distance_m': real_dist,
                'correction_factor': correction
            })
        
        # 計算出手球速（如果有 release_point）
        release_speed = None
        if release_point and len(trajectory_points) > 0:
            release_speed = self.calculate_release_speed(
                release_point, 
                trajectory_points[0],
                frames_elapsed=2.5
            )
        
        # 計算初速度（前 3-5 個 frame 的平均）
        initial_speeds = [s['speed_kmh'] for s in speeds[:min(5, len(speeds))]]
        initial_speed = np.mean(initial_speeds) if initial_speeds else 0
        
        # 計算最大速度
        max_speed = max([s['speed_kmh'] for s in speeds]) if speeds else 0
        
        # 計算平均速度
        avg_speed = np.mean([s['speed_kmh'] for s in speeds]) if speeds else 0
        
        # 計算總飛行距離
        calculated_distance = sum([s['distance_m'] for s in speeds])
        
        return {
            'release_speed_kmh': release_speed,  # 出手球速（最準確）
            'initial_speed_kmh': initial_speed,   # 初速度（前幾個 frame 平均）
            'max_speed_kmh': max_speed,           # 最大速度
            'average_speed_kmh': avg_speed,       # 平均速度
            'total_distance_m': calculated_distance,  # 計算的距離
            'num_frames': len(trajectory_points), # 追蹤的 frame 數
            'frame_details': speeds,              # 每個 frame 的詳細資訊
            'calculation_method': 'pixel_based'   # 計算方法標記
        }


class FieldCalibrationTool:
    """場地校正工具，用於手動標記參考點"""
    
    def __init__(self, first_frame: np.ndarray):
        """
        Args:
            first_frame: 影片的第一幀（BGR 格式）
        """
        self.frame = first_frame.copy()
        self.display_frame = first_frame.copy()
        self.reference_points = []
        
    def mark_reference_points(self) -> Optional[List[Tuple[int, int]]]:
        """
        讓使用者在第一幀上標記參考點
        
        Returns:
            [投手板座標, 本壘板座標] 或 None
        """
        points = []
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(points) < 2:
                    points.append((x, y))
                    
                    # 重新繪製所有標記
                    self.display_frame = self.frame.copy()
                    
                    # 繪製點和標籤
                    for i, pt in enumerate(points):
                        cv2.circle(self.display_frame, pt, 8, (0, 255, 0), -1)
                        cv2.circle(self.display_frame, pt, 12, (0, 255, 0), 2)
                        
                        label = "1. Pitcher Mound" if i == 0 else "2. Home Plate"
                        cv2.putText(
                            self.display_frame, 
                            label, 
                            (pt[0] + 15, pt[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.8, 
                            (0, 255, 0), 
                            2
                        )
                    
                    # 如果有兩個點，繪製連線
                    if len(points) == 2:
                        cv2.line(
                            self.display_frame, 
                            points[0], 
                            points[1], 
                            (0, 255, 255), 
                            2
                        )
                        
                        # 計算並顯示距離
                        dist = np.sqrt(
                            (points[1][0] - points[0][0])**2 + 
                            (points[1][1] - points[0][1])**2
                        )
                        mid_x = (points[0][0] + points[1][0]) // 2
                        mid_y = (points[0][1] + points[1][1]) // 2
                        cv2.putText(
                            self.display_frame,
                            f"{dist:.1f} pixels",
                            (mid_x, mid_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 255),
                            2
                        )
                    
                    cv2.imshow("Field Calibration", self.display_frame)
        
        # 添加說明文字
        instructions = [
            "Field Calibration Tool",
            "",
            "Instructions:",
            "1. Click on the pitcher's mound",
            "2. Click on the home plate",
            "3. Press ENTER to confirm",
            "4. Press ESC to cancel"
        ]
        
        y_offset = 30
        for i, text in enumerate(instructions):
            color = (255, 255, 255) if i == 0 else (200, 200, 200)
            thickness = 2 if i == 0 else 1
            cv2.putText(
                self.display_frame,
                text,
                (20, y_offset + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                thickness
            )
        
        cv2.namedWindow("Field Calibration")
        cv2.setMouseCallback("Field Calibration", mouse_callback)
        cv2.imshow("Field Calibration", self.display_frame)
        
        print("\n" + "="*60)
        print("場地校正工具")
        print("="*60)
        print("請依序點擊：")
        print("  1. 投手板位置（投手站立的橡膠板）")
        print("  2. 本壘板位置（捕手後方的白色板）")
        print("\n標準棒球場距離：18.44 公尺")
        print("\n完成標記後按 ENTER 確認，按 ESC 取消")
        print("="*60 + "\n")
        
        while True:
            key = cv2.waitKey(10) & 0xFF
            
            if key == 13:  # ENTER
                if len(points) == 2:
                    break
                else:
                    print("⚠ 請標記兩個點後再按 ENTER")
            elif key == 27:  # ESC
                print("✗ 校正已取消")
                cv2.destroyAllWindows()
                return None
        
        cv2.destroyAllWindows()
        
        if len(points) == 2:
            self.reference_points = points
            print(f"✓ 已標記參考點：")
            print(f"  投手板: {points[0]}")
            print(f"  本壘板: {points[1]}")
            return points
        
        return None
