# core/swept_volume_analyzer.py
"""
扫掠体积分析器
实现高效的扫掠体积计算和可视化
"""
import numpy as np
from typing import List, Tuple, Optional, Dict
from concurrent.futures import ThreadPoolExecutor
import threading
from utils.math_utils import MathUtils, GeometryUtils
from utils.config import config
from core.sdf_calculator import SDFCalculator

class SweptVolumeAnalyzer:
    """
    扫掠体积分析器
    提供高效的扫掠体积计算、分析和可视化功能
    """
    
    def __init__(self, robot_length: float, robot_width: float):
        self.robot_length = robot_length
        self.robot_width = robot_width
        
        # SDF计算器
        self.sdf_calculator = SDFCalculator(robot_length, robot_width)
        
        # 缓存
        self._sdf_cache = {}
        self._cache_lock = threading.Lock()
        
        # 性能监控
        self.computation_times = {}
        self.cache_hit_rate = 0.0
        
    def compute_swept_volume_boundary(self, trajectory: List[np.ndarray],
                                    resolution: float = 0.05) -> List[np.ndarray]:
        """
        计算扫掠体积边界
        返回边界点列表用于可视化
        """
        if not trajectory:
            return []
        
        print(f"计算扫掠体积边界，轨迹点数: {len(trajectory)}")
        
        # 收集所有机器人角点
        all_corners = []
        
        for traj_point in trajectory:
            robot_pose = traj_point[:3]
            corners = self._get_robot_corners(robot_pose)
            all_corners.extend(corners)
        
        if not all_corners:
            return []
        
        # 计算凸包作为扫掠体积边界
        try:
            boundary_points = GeometryUtils.convex_hull_2d(all_corners)
            print(f"扫掠体积边界点数: {len(boundary_points)}")
            return boundary_points
        except Exception as e:
            print(f"计算凸包失败: {e}")
            return all_corners
    
    def compute_swept_volume_area(self, trajectory: List[np.ndarray]) -> float:
        """计算扫掠体积面积"""
        boundary_points = self.compute_swept_volume_boundary(trajectory)
        if len(boundary_points) < 3:
            return 0.0
        
        area = GeometryUtils.polygon_area(boundary_points)
        return area
    
    def compute_detailed_swept_volume(self, trajectory: List[np.ndarray],
                                    grid_resolution: float = 0.1) -> Dict:
        """
        计算详细的扫掠体积信息
        包括体积、边界、密度分布等
        """
        import time
        start_time = time.time()
        
        if not trajectory:
            return {
                'volume': 0.0,
                'area': 0.0,
                'boundary_points': [],
                'density_grid': None,
                'statistics': {}
            }
        
        # 1. 计算边界
        boundary_points = self.compute_swept_volume_boundary(trajectory)
        area = GeometryUtils.polygon_area(boundary_points) if len(boundary_points) >= 3 else 0.0
        
        # 2. 计算密度网格
        density_grid, grid_bounds = self._compute_density_grid(trajectory, grid_resolution)
        
        # 3. 计算统计信息
        statistics = self._compute_volume_statistics(trajectory, density_grid, grid_bounds)
        
        computation_time = time.time() - start_time
        self.computation_times['detailed_volume'] = computation_time
        
        result = {
            'volume': area,  # 2D情况下体积就是面积
            'area': area,
            'boundary_points': boundary_points,
            'density_grid': density_grid,
            'grid_bounds': grid_bounds,
            'grid_resolution': grid_resolution,
            'statistics': statistics,
            'computation_time': computation_time
        }
        
        print(f"详细扫掠体积计算完成，面积: {area:.3f} m², 耗时: {computation_time:.3f}s")
        return result
    
    def _compute_density_grid(self, trajectory: List[np.ndarray],
                            grid_resolution: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算扫掠密度网格
        每个网格点记录机器人经过该点的次数/时间
        """
        # 计算轨迹边界
        positions = np.array([point[:2] for point in trajectory])
        x_min, y_min = np.min(positions, axis=0)
        x_max, y_max = np.max(positions, axis=0)
        
        # 添加机器人尺寸的边界
        margin = max(self.robot_length, self.robot_width) / 2.0 + 0.5
        x_min -= margin
        x_max += margin
        y_min -= margin
        y_max += margin
        
        # 网格大小
        grid_width = int((x_max - x_min) / grid_resolution) + 1
        grid_height = int((y_max - y_min) / grid_resolution) + 1
        
        # 初始化密度网格
        density_grid = np.zeros((grid_height, grid_width))
        grid_bounds = np.array([x_min, y_min, x_max, y_max])
        
        # 计算每个网格点的密度
        for i in range(grid_height):
            for j in range(grid_width):
                # 网格点世界坐标
                world_x = x_min + j * grid_resolution
                world_y = y_min + i * grid_resolution
                query_point = np.array([world_x, world_y])
                
                # 计算该点被机器人覆盖的时间
                coverage_time = 0.0
                
                for k, traj_point in enumerate(trajectory):
                    robot_pose = traj_point[:3]
                    sdf = self.sdf_calculator.compute_robot_sdf(query_point, robot_pose)
                    
                    if sdf <= 0:  # 点在机器人内部
                        # 计算时间权重
                        if k < len(trajectory) - 1:
                            dt = trajectory[k + 1][3] - trajectory[k][3]
                        else:
                            dt = 0.1  # 默认时间步长
                        
                        coverage_time += dt
                
                density_grid[i, j] = coverage_time
        
        return density_grid, grid_bounds
    
    def _compute_volume_statistics(self, trajectory: List[np.ndarray],
                                 density_grid: np.ndarray,
                                 grid_bounds: np.ndarray) -> Dict:
        """计算扫掠体积统计信息"""
        statistics = {}
        
        # 基本统计
        total_time = trajectory[-1][3] - trajectory[0][3] if len(trajectory) > 1 else 0.0
        path_length = 0.0
        
        for i in range(1, len(trajectory)):
            path_length += MathUtils.euclidean_distance(
                trajectory[i][:2], trajectory[i-1][:2])
        
        statistics['total_time'] = total_time
        statistics['path_length'] = path_length
        statistics['average_speed'] = path_length / total_time if total_time > 0 else 0.0
        
        # 密度统计
        if density_grid is not None:
            non_zero_mask = density_grid > 0
            statistics['max_density'] = np.max(density_grid)
            statistics['mean_density'] = np.mean(density_grid[non_zero_mask]) if np.any(non_zero_mask) else 0.0
            statistics['coverage_ratio'] = np.sum(non_zero_mask) / density_grid.size
        
        # 机器人姿态统计
        orientations = [point[2] for point in trajectory]
        angular_velocities = []
        
        for i in range(1, len(orientations)):
            dt = trajectory[i][3] - trajectory[i-1][3]
            if dt > 0:
                dtheta = MathUtils.normalize_angle(orientations[i] - orientations[i-1])
                angular_velocities.append(abs(dtheta) / dt)
        
        if angular_velocities:
            statistics['max_angular_velocity'] = max(angular_velocities)
            statistics['mean_angular_velocity'] = np.mean(angular_velocities)
            statistics['total_rotation'] = sum(abs(MathUtils.normalize_angle(
                orientations[i] - orientations[i-1])) for i in range(1, len(orientations)))
        
        return statistics
    
    def _get_robot_corners(self, robot_pose: np.ndarray) -> List[np.ndarray]:
        """获取机器人四个角点的世界坐标"""
        x, y, theta = robot_pose[0], robot_pose[1], robot_pose[2]
        
        # 机器人局部坐标系中的四个角点
        half_length = self.robot_length / 2.0
        half_width = self.robot_width / 2.0
        
        local_corners = [
            np.array([-half_length, -half_width]),
            np.array([half_length, -half_width]),
            np.array([half_length, half_width]),
            np.array([-half_length, half_width])
        ]
        
        # 转换到世界坐标系
        world_corners = []
        R = MathUtils.rotation_matrix_2d(theta)
        translation = np.array([x, y])
        
        for corner in local_corners:
            world_corner = R @ corner + translation
            world_corners.append(world_corner)
        
        return world_corners
    
    def analyze_swept_volume_optimization(self, trajectory_before: List[np.ndarray],
                                        trajectory_after: List[np.ndarray]) -> Dict:
        """
        分析扫掠体积优化效果
        比较优化前后的扫掠体积
        """
        # 计算优化前的扫掠体积
        before_info = self.compute_detailed_swept_volume(trajectory_before)
        
        # 计算优化后的扫掠体积
        after_info = self.compute_detailed_swept_volume(trajectory_after)
        
        # 计算改进指标
        volume_reduction = before_info['area'] - after_info['area']
        volume_reduction_ratio = volume_reduction / before_info['area'] if before_info['area'] > 0 else 0.0
        
        analysis = {
            'before': before_info,
            'after': after_info,
            'improvement': {
                'volume_reduction': volume_reduction,
                'volume_reduction_ratio': volume_reduction_ratio,
                'percentage_improvement': volume_reduction_ratio * 100
            }
        }
        
        print(f"扫掠体积优化分析:")
        print(f"  优化前面积: {before_info['area']:.3f} m²")
        print(f"  优化后面积: {after_info['area']:.3f} m²")
        print(f"  面积减少: {volume_reduction:.3f} m² ({volume_reduction_ratio*100:.1f}%)")
        
        return analysis
    
    def compute_swept_volume_for_minco(self, segments) -> float:
        """
        为MINCO优化提供的扫掠体积计算函数
        """
        try:
            # 从MINCO段生成轨迹点
            trajectory_points = []
            current_time = 0.0
            
            for segment in segments:
                duration = segment.duration
                num_samples = max(5, int(duration / 0.1))  # 至少5个采样点
                
                for i in range(num_samples + 1):
                    t_local = i * duration / num_samples
                    if t_local > duration:
                        t_local = duration
                    
                    # 评估轨迹段
                    pos = segment.evaluate_position(t_local)
                    
                    # 构建轨迹点 [x, y, theta, time]
                    traj_point = np.array([pos[0], pos[1], pos[2], current_time + t_local])
                    trajectory_points.append(traj_point)
                
                current_time += duration
            
            # 计算扫掠体积
            if len(trajectory_points) > 0:
                area = self.compute_swept_volume_area(trajectory_points)
                return area
            else:
                return 0.0
                
        except Exception as e:
            print(f"MINCO扫掠体积计算异常: {e}")
            return 1e6  # 返回大值表示异常
    
    def get_performance_metrics(self) -> Dict:
        """获取性能指标"""
        return {
            'computation_times': self.computation_times.copy(),
            'cache_hit_rate': self.cache_hit_rate,
            'cache_size': len(self._sdf_cache)
        }
    
    def clear_cache(self):
        """清理缓存"""
        with self._cache_lock:
            self._sdf_cache.clear()
            self.cache_hit_rate = 0.0