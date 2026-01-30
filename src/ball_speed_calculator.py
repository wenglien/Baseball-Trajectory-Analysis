import numpy as np
from typing import List, Tuple, Optional, Dict


class BallSpeedCalculator:
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
        
        y = point[1]
        
        if self.far_y == self.near_y:
            return 1.0
        
        t = (y - self.near_y) / (self.far_y - self.near_y)
        t = np.clip(t, 0, 1)
        correction_factor = 1.15 - (0.3 * t)
        
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
        
        pixel_dist = np.sqrt(
            (first_ball_point[0] - release_point[0])**2 + 
            (first_ball_point[1] - release_point[1])**2
        )
        
        if pixel_dist < 10:
            return None
        
        max_reasonable_pixel_dist = self.video_width / 2
        if pixel_dist > max_reasonable_pixel_dist:
            return None
        
        mid_point_y = (release_point[1] + first_ball_point[1]) / 2
        correction = self._apply_perspective_correction((0, int(mid_point_y)))
        
        real_dist = (pixel_dist / self.pixels_per_meter) * correction
        
        time = frames_elapsed / self.fps
        
        speed_ms = real_dist / time
        speed_kmh = speed_ms * 3.6
        
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
        if len(trajectory_points) < 2:
            return {"error": "軌跡點數不足（至少需要 2 個點）"}
        
        if self.theoretical_distance:
            num_frames = len(trajectory_points)
            if not self.fps:
                return {"error": "fps 不可為 0，無法計算時間"}
            total_time = num_frames / self.fps
            
            avg_speed_ms = self.theoretical_distance / total_time
            avg_speed_kmh = avg_speed_ms * 3.6
            
            initial_speed = avg_speed_kmh * 1.1
            max_speed = avg_speed_kmh * 1.15
            
            release_speed = None
            if self.pixels_per_meter is not None and release_point and len(trajectory_points) > 0:
                release_speed = self.calculate_release_speed(
                    release_point, 
                    trajectory_points[0],
                    frames_elapsed=2.5
                )
            
            return {
                'release_speed_kmh': release_speed,
                'initial_speed_kmh': initial_speed,
                'max_speed_kmh': max_speed,
                'average_speed_kmh': avg_speed_kmh,
                'total_distance_m': self.theoretical_distance,
                'num_frames': num_frames,
                'calculation_method': 'theoretical'
            }
        
        if self.pixels_per_meter is None:
            return {
                "error": "缺少距離資訊：請提供 theoretical_distance（手動輸入投手到捕手距離），或先執行 calibrate_from_reference 進行像素校正"
            }
        
        speeds = []
        time_interval = 1.0 / self.fps
        
        for i in range(len(trajectory_points) - 1):
            p1 = trajectory_points[i]
            p2 = trajectory_points[i + 1]
            
            pixel_dist = np.sqrt(
                (p2[0] - p1[0])**2 + 
                (p2[1] - p1[1])**2
            )
            
            mid_y = (p1[1] + p2[1]) / 2
            correction = self._apply_perspective_correction((0, int(mid_y)))
            
            real_dist = (pixel_dist / self.pixels_per_meter) * correction
            
            speed_ms = real_dist / time_interval
            speed_kmh = speed_ms * 3.6
            
            speeds.append({
                'frame': i,
                'speed_kmh': speed_kmh,
                'speed_ms': speed_ms,
                'distance_m': real_dist,
                'correction_factor': correction
            })
        
        release_speed = None
        if release_point and len(trajectory_points) > 0:
            release_speed = self.calculate_release_speed(
                release_point, 
                trajectory_points[0],
                frames_elapsed=2.5
            )
        
        initial_speeds = [s['speed_kmh'] for s in speeds[:min(5, len(speeds))]]
        initial_speed = np.mean(initial_speeds) if initial_speeds else 0
        
        max_speed = max([s['speed_kmh'] for s in speeds]) if speeds else 0
        
        avg_speed = np.mean([s['speed_kmh'] for s in speeds]) if speeds else 0
        
        calculated_distance = sum([s['distance_m'] for s in speeds])
        
        return {
            'release_speed_kmh': release_speed,
            'initial_speed_kmh': initial_speed,
            'max_speed_kmh': max_speed,
            'average_speed_kmh': avg_speed,
            'total_distance_m': calculated_distance,
            'num_frames': len(trajectory_points),
            'frame_details': speeds,
            'calculation_method': 'pixel_based'
        }
