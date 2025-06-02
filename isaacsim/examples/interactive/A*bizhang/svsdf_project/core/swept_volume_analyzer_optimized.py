"""
高性能扫掠体积分析器
实现工业级的并行扫掠体积计算、分析和优化
"""
import numpy as np
import numba as nb
from typing import List, Tuple, Optional, Dict, Union, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import threading
import time
import multiprocessing
from functools import lru_cache
import warnings

from utils.math_utils import MathUtils, GeometryUtils
from utils.config import config

# Suppress numba warnings for cleaner output
warnings.filterwarnings('ignore', category=nb.NumbaTypeSafetyWarning)

@dataclass
class SweptVolumeData:
    """扫掠体积数据结构"""
    volume: float = 0.0
    area: float = 0.0
    boundary_points: List[np.ndarray] = field(default_factory=list)
    density_grid: Optional[np.ndarray] = None
    grid_bounds: Optional[np.ndarray] = None
    grid_resolution: float = 0.1
    statistics: Dict[str, Any] = field(default_factory=dict)
    computation_time: float = 0.0
    quality_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class SweptVolumeMetrics:
    """扫掠体积性能指标"""
    total_computations: int = 0
    total_computation_time: float = 0.0
    average_computation_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_rate: float = 0.0
    parallel_speedup: float = 1.0
    memory_usage_mb: float = 0.0
    grid_efficiency: float = 0.0

@nb.jit(nopython=True, cache=True)
def compute_robot_coverage_jit(robot_corners: np.ndarray, grid_x: float, grid_y: float) -> bool:
    """JIT优化的机器人覆盖检测"""
    # 使用射线投射算法检测点是否在多边形内
    x, y = grid_x, grid_y
    n = len(robot_corners)
    inside = False
    
    j = n - 1
    for i in range(n):
        xi, yi = robot_corners[i, 0], robot_corners[i, 1]
        xj, yj = robot_corners[j, 0], robot_corners[j, 1]
        
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    
    return inside

@nb.jit(nopython=True, cache=True)
def compute_density_grid_jit(trajectory_positions: np.ndarray, 
                           trajectory_orientations: np.ndarray,
                           trajectory_times: np.ndarray,
                           robot_length: float, 
                           robot_width: float,
                           grid_bounds: np.ndarray,
                           grid_resolution: float) -> np.ndarray:
    """JIT优化的密度网格计算"""
    x_min, y_min, x_max, y_max = grid_bounds
    grid_width = int((x_max - x_min) / grid_resolution) + 1
    grid_height = int((y_max - y_min) / grid_resolution) + 1
    
    density_grid = np.zeros((grid_height, grid_width))
    half_length = robot_length / 2.0
    half_width = robot_width / 2.0
    
    # 机器人局部角点
    local_corners = np.array([
        [-half_length, -half_width],
        [half_length, -half_width],
        [half_length, half_width],
        [-half_length, half_width]
    ])
    
    for i in range(grid_height):
        for j in range(grid_width):
            # 网格点世界坐标
            world_x = x_min + j * grid_resolution
            world_y = y_min + i * grid_resolution
            
            coverage_time = 0.0
            
            # 检查每个轨迹点
            for k in range(len(trajectory_positions)):
                x, y = trajectory_positions[k, 0], trajectory_positions[k, 1]
                theta = trajectory_orientations[k]
                
                # 计算旋转矩阵
                cos_theta = np.cos(theta)
                sin_theta = np.sin(theta)
                
                # 计算机器人世界角点
                robot_corners = np.zeros((4, 2))
                for corner_idx in range(4):
                    local_x, local_y = local_corners[corner_idx, 0], local_corners[corner_idx, 1]
                    world_corner_x = cos_theta * local_x - sin_theta * local_y + x
                    world_corner_y = sin_theta * local_x + cos_theta * local_y + y
                    robot_corners[corner_idx, 0] = world_corner_x
                    robot_corners[corner_idx, 1] = world_corner_y
                
                # 检查覆盖
                if compute_robot_coverage_jit(robot_corners, world_x, world_y):
                    if k < len(trajectory_times) - 1:
                        dt = trajectory_times[k + 1] - trajectory_times[k]
                    else:
                        dt = 0.1
                    coverage_time += dt
            
            density_grid[i, j] = coverage_time
    
    return density_grid

@nb.jit(nopython=True, cache=True)
def compute_convex_hull_area_jit(points: np.ndarray) -> float:
    """JIT优化的凸包面积计算"""
    if len(points) < 3:
        return 0.0
    
    area = 0.0
    n = len(points)
    
    for i in range(n):
        j = (i + 1) % n
        area += points[i, 0] * points[j, 1]
        area -= points[j, 0] * points[i, 1]
    
    return abs(area) / 2.0

class SweptVolumeAnalyzerOptimized:
    """
    高性能扫掠体积分析器
    提供工业级的并行计算、缓存、自适应优化等功能
    """
    
    def __init__(self, robot_length: float, robot_width: float, 
                 enable_parallel: bool = True, max_workers: Optional[int] = None):
        self.robot_length = robot_length
        self.robot_width = robot_width
        
        # 并行处理配置
        self.enable_parallel = enable_parallel
        self.max_workers = max_workers or min(multiprocessing.cpu_count(), 8)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers) if enable_parallel else None
        
        # 高性能缓存
        self._volume_cache = {}
        self._density_cache = {}
        self._boundary_cache = {}
        self._cache_lock = threading.Lock()
        self.cache_max_size = 1000
        
        # 性能监控
        self.metrics = SweptVolumeMetrics()
        self._computation_history = []
        
        # 自适应参数
        self.adaptive_grid_resolution = True
        self.min_grid_resolution = 0.02
        self.max_grid_resolution = 0.2
        self.target_computation_time = 0.05  # 目标计算时间50ms
        
        print(f"SweptVolumeAnalyzerOptimized初始化完成")
        print(f"  并行处理: {'启用' if enable_parallel else '禁用'}")
        print(f"  最大工作线程: {self.max_workers}")
        print(f"  自适应网格分辨率: {'启用' if self.adaptive_grid_resolution else '禁用'}")
    
    def compute_swept_volume_boundary_optimized(self, trajectory: List[np.ndarray],
                                              resolution: float = 0.05,
                                              use_cache: bool = True) -> List[np.ndarray]:
        """
        优化的扫掠体积边界计算
        支持并行处理和智能缓存
        """
        if not trajectory:
            return []
        
        start_time = time.time()
        
        # 生成缓存键
        cache_key = self._generate_trajectory_cache_key(trajectory, resolution)
        
        # 检查缓存
        if use_cache and cache_key in self._boundary_cache:
            self.metrics.cache_hits += 1
            return self._boundary_cache[cache_key].copy()
        
        self.metrics.cache_misses += 1
        
        try:
            if self.enable_parallel and len(trajectory) > 100:
                # 并行计算大轨迹的边界
                boundary_points = self._compute_boundary_parallel(trajectory, resolution)
            else:
                # 串行计算小轨迹的边界
                boundary_points = self._compute_boundary_serial(trajectory)
            
            # 缓存结果
            if use_cache:
                self._cache_boundary_result(cache_key, boundary_points)
            
            computation_time = time.time() - start_time
            self._update_metrics(computation_time)
            
            print(f"扫掠体积边界计算完成，点数: {len(boundary_points)}, 耗时: {computation_time:.3f}s")
            return boundary_points
            
        except Exception as e:
            print(f"扫掠体积边界计算失败: {e}")
            return []
    
    def _compute_boundary_parallel(self, trajectory: List[np.ndarray], 
                                 resolution: float) -> List[np.ndarray]:
        """并行计算扫掠体积边界"""
        # 分割轨迹为块
        chunk_size = max(10, len(trajectory) // self.max_workers)
        trajectory_chunks = [trajectory[i:i + chunk_size] 
                           for i in range(0, len(trajectory), chunk_size)]
        
        all_corners = []
        
        # 并行处理每个块
        futures = []
        for chunk in trajectory_chunks:
            future = self.thread_pool.submit(self._compute_chunk_corners, chunk)
            futures.append(future)
        
        # 收集结果
        for future in as_completed(futures):
            try:
                chunk_corners = future.result()
                all_corners.extend(chunk_corners)
            except Exception as e:
                print(f"并行块处理失败: {e}")
        
        # 计算凸包
        if all_corners:
            try:
                return GeometryUtils.convex_hull_2d(all_corners)
            except Exception as e:
                print(f"凸包计算失败: {e}")
                return all_corners
        
        return []
    
    def _compute_boundary_serial(self, trajectory: List[np.ndarray]) -> List[np.ndarray]:
        """串行计算扫掠体积边界"""
        all_corners = []
        
        for traj_point in trajectory:
            robot_pose = traj_point[:3]
            corners = self._get_robot_corners_optimized(robot_pose)
            all_corners.extend(corners)
        
        if all_corners:
            try:
                return GeometryUtils.convex_hull_2d(all_corners)
            except Exception as e:
                print(f"凸包计算失败: {e}")
                return all_corners
        
        return []
    
    def _compute_chunk_corners(self, trajectory_chunk: List[np.ndarray]) -> List[np.ndarray]:
        """计算轨迹块的角点"""
        chunk_corners = []
        for traj_point in trajectory_chunk:
            robot_pose = traj_point[:3]
            corners = self._get_robot_corners_optimized(robot_pose)
            chunk_corners.extend(corners)
        return chunk_corners
    
    def compute_detailed_swept_volume_optimized(self, trajectory: List[np.ndarray],
                                              grid_resolution: Optional[float] = None,
                                              use_cache: bool = True,
                                              compute_statistics: bool = True) -> SweptVolumeData:
        """
        优化的详细扫掠体积计算
        自适应网格分辨率和并行处理
        """
        start_time = time.time()
        
        if not trajectory:
            return SweptVolumeData()
        
        # 自适应网格分辨率
        if grid_resolution is None:
            grid_resolution = self._adaptive_grid_resolution(trajectory)
        
        # 生成缓存键
        cache_key = self._generate_trajectory_cache_key(trajectory, grid_resolution)
        
        # 检查缓存
        if use_cache and cache_key in self._volume_cache:
            self.metrics.cache_hits += 1
            return self._volume_cache[cache_key]
        
        self.metrics.cache_misses += 1
        
        try:
            # 1. 计算边界（可能使用缓存）
            boundary_points = self.compute_swept_volume_boundary_optimized(
                trajectory, use_cache=use_cache)
            area = self._compute_area_optimized(boundary_points)
            
            # 2. 并行计算密度网格
            density_grid, grid_bounds = self._compute_density_grid_optimized(
                trajectory, grid_resolution)
            
            # 3. 计算统计信息
            statistics = {}
            quality_metrics = {}
            
            if compute_statistics:
                statistics = self._compute_volume_statistics_optimized(
                    trajectory, density_grid, grid_bounds)
                quality_metrics = self._compute_quality_metrics(
                    trajectory, boundary_points, density_grid)
            
            computation_time = time.time() - start_time
            
            # 创建结果
            result = SweptVolumeData(
                volume=area,
                area=area,
                boundary_points=boundary_points,
                density_grid=density_grid,
                grid_bounds=grid_bounds,
                grid_resolution=grid_resolution,
                statistics=statistics,
                computation_time=computation_time,
                quality_metrics=quality_metrics
            )
            
            # 缓存结果
            if use_cache:
                self._cache_volume_result(cache_key, result)
            
            self._update_metrics(computation_time)
            
            print(f"详细扫掠体积计算完成")
            print(f"  面积: {area:.3f} m²")
            print(f"  网格分辨率: {grid_resolution:.3f} m")
            print(f"  计算时间: {computation_time:.3f}s")
            
            return result
            
        except Exception as e:
            print(f"详细扫掠体积计算失败: {e}")
            return SweptVolumeData(computation_time=time.time() - start_time)
    
    def _compute_density_grid_optimized(self, trajectory: List[np.ndarray],
                                      grid_resolution: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        优化的密度网格计算
        使用JIT加速和并行处理
        """
        # 准备轨迹数据
        positions = np.array([point[:2] for point in trajectory])
        orientations = np.array([point[2] for point in trajectory])
        times = np.array([point[3] if len(point) > 3 else i * 0.1 
                         for i, point in enumerate(trajectory)])
        
        # 计算网格边界
        grid_bounds = self._compute_grid_bounds(positions, grid_resolution)
        
        # 使用JIT优化计算
        density_grid = compute_density_grid_jit(
            positions, orientations, times,
            self.robot_length, self.robot_width,
            grid_bounds, grid_resolution
        )
        
        return density_grid, grid_bounds
    
    def _compute_grid_bounds(self, positions: np.ndarray, 
                           grid_resolution: float) -> np.ndarray:
        """计算网格边界"""
        x_min, y_min = np.min(positions, axis=0)
        x_max, y_max = np.max(positions, axis=0)
        
        # 添加机器人尺寸的边界
        margin = max(self.robot_length, self.robot_width) / 2.0 + 0.5
        
        # 对齐到网格
        x_min = np.floor((x_min - margin) / grid_resolution) * grid_resolution
        x_max = np.ceil((x_max + margin) / grid_resolution) * grid_resolution
        y_min = np.floor((y_min - margin) / grid_resolution) * grid_resolution
        y_max = np.ceil((y_max + margin) / grid_resolution) * grid_resolution
        
        return np.array([x_min, y_min, x_max, y_max])
    
    def _adaptive_grid_resolution(self, trajectory: List[np.ndarray]) -> float:
        """自适应网格分辨率选择"""
        if not self.adaptive_grid_resolution:
            return 0.1
        
        # 基于轨迹复杂度选择分辨率
        trajectory_length = len(trajectory)
        
        if trajectory_length < 50:
            resolution = self.min_grid_resolution
        elif trajectory_length > 500:
            resolution = self.max_grid_resolution
        else:
            # 线性插值
            ratio = (trajectory_length - 50) / (500 - 50)
            resolution = (self.min_grid_resolution + 
                         ratio * (self.max_grid_resolution - self.min_grid_resolution))
        
        # 基于机器人尺寸调整
        robot_size = min(self.robot_length, self.robot_width)
        resolution = max(resolution, robot_size / 10.0)
        resolution = min(resolution, robot_size / 2.0)
        
        return resolution
    
    def _compute_area_optimized(self, boundary_points: List[np.ndarray]) -> float:
        """优化的面积计算"""
        if len(boundary_points) < 3:
            return 0.0
        
        try:
            # 转换为numpy数组进行JIT计算
            points_array = np.array(boundary_points)
            return compute_convex_hull_area_jit(points_array)
        except Exception as e:
            print(f"面积计算失败: {e}")
            return GeometryUtils.polygon_area(boundary_points)
    
    def _compute_volume_statistics_optimized(self, trajectory: List[np.ndarray],
                                           density_grid: np.ndarray,
                                           grid_bounds: np.ndarray) -> Dict:
        """优化的统计信息计算"""
        statistics = {}
        
        try:
            # 基本轨迹统计
            if len(trajectory) > 1:
                total_time = trajectory[-1][3] - trajectory[0][3] if len(trajectory[0]) > 3 else len(trajectory) * 0.1
                path_length = 0.0
                
                for i in range(1, len(trajectory)):
                    path_length += np.linalg.norm(trajectory[i][:2] - trajectory[i-1][:2])
                
                statistics['total_time'] = total_time
                statistics['path_length'] = path_length
                statistics['average_speed'] = path_length / total_time if total_time > 0 else 0.0
                
                # 角速度统计
                orientations = [point[2] for point in trajectory]
                angular_velocities = []
                
                for i in range(1, len(orientations)):
                    if len(trajectory[0]) > 3:
                        dt = trajectory[i][3] - trajectory[i-1][3]
                    else:
                        dt = 0.1
                    
                    if dt > 0:
                        dtheta = MathUtils.normalize_angle(orientations[i] - orientations[i-1])
                        angular_velocities.append(abs(dtheta) / dt)
                
                if angular_velocities:
                    statistics['max_angular_velocity'] = max(angular_velocities)
                    statistics['mean_angular_velocity'] = np.mean(angular_velocities)
            
            # 密度网格统计
            if density_grid is not None and density_grid.size > 0:
                non_zero_mask = density_grid > 0
                if np.any(non_zero_mask):
                    statistics['max_density'] = float(np.max(density_grid))
                    statistics['mean_density'] = float(np.mean(density_grid[non_zero_mask]))
                    statistics['coverage_ratio'] = float(np.sum(non_zero_mask) / density_grid.size)
                    statistics['density_std'] = float(np.std(density_grid[non_zero_mask]))
                else:
                    statistics['max_density'] = 0.0
                    statistics['mean_density'] = 0.0
                    statistics['coverage_ratio'] = 0.0
                    statistics['density_std'] = 0.0
        
        except Exception as e:
            print(f"统计信息计算失败: {e}")
        
        return statistics
    
    def _compute_quality_metrics(self, trajectory: List[np.ndarray],
                               boundary_points: List[np.ndarray],
                               density_grid: Optional[np.ndarray]) -> Dict:
        """计算质量指标"""
        quality_metrics = {}
        
        try:
            # 边界质量
            if boundary_points:
                quality_metrics['boundary_smoothness'] = self._compute_boundary_smoothness(boundary_points)
                quality_metrics['boundary_compactness'] = self._compute_boundary_compactness(boundary_points)
            
            # 轨迹质量
            if len(trajectory) > 1:
                quality_metrics['trajectory_smoothness'] = self._compute_trajectory_smoothness(trajectory)
                quality_metrics['trajectory_efficiency'] = self._compute_trajectory_efficiency(trajectory)
            
            # 密度分布质量
            if density_grid is not None:
                quality_metrics['density_uniformity'] = self._compute_density_uniformity(density_grid)
        
        except Exception as e:
            print(f"质量指标计算失败: {e}")
        
        return quality_metrics
    
    def _compute_boundary_smoothness(self, boundary_points: List[np.ndarray]) -> float:
        """计算边界平滑度"""
        if len(boundary_points) < 3:
            return 0.0
        
        total_curvature = 0.0
        n = len(boundary_points)
        
        for i in range(n):
            p1 = boundary_points[i]
            p2 = boundary_points[(i + 1) % n]
            p3 = boundary_points[(i + 2) % n]
            
            # 计算曲率
            v1 = p2 - p1
            v2 = p3 - p2
            
            if np.linalg.norm(v1) > 1e-6 and np.linalg.norm(v2) > 1e-6:
                cross_product = np.cross(v1, v2)
                curvature = abs(cross_product) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                total_curvature += curvature
        
        return 1.0 / (1.0 + total_curvature / n)  # 平滑度越高值越大
    
    def _compute_boundary_compactness(self, boundary_points: List[np.ndarray]) -> float:
        """计算边界紧凑度"""
        if len(boundary_points) < 3:
            return 0.0
        
        area = self._compute_area_optimized(boundary_points)
        perimeter = 0.0
        
        n = len(boundary_points)
        for i in range(n):
            p1 = boundary_points[i]
            p2 = boundary_points[(i + 1) % n]
            perimeter += np.linalg.norm(p2 - p1)
        
        if perimeter > 0:
            # 圆的紧凑度为1，越不规则值越小
            compactness = 4 * np.pi * area / (perimeter * perimeter)
            return min(compactness, 1.0)
        
        return 0.0
    
    def _compute_trajectory_smoothness(self, trajectory: List[np.ndarray]) -> float:
        """计算轨迹平滑度"""
        if len(trajectory) < 3:
            return 1.0
        
        total_curvature = 0.0
        
        for i in range(1, len(trajectory) - 1):
            p1 = trajectory[i - 1][:2]
            p2 = trajectory[i][:2]
            p3 = trajectory[i + 1][:2]
            
            v1 = p2 - p1
            v2 = p3 - p2
            
            if np.linalg.norm(v1) > 1e-6 and np.linalg.norm(v2) > 1e-6:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle_change = np.arccos(cos_angle)
                total_curvature += angle_change
        
        return 1.0 / (1.0 + total_curvature / len(trajectory))
    
    def _compute_trajectory_efficiency(self, trajectory: List[np.ndarray]) -> float:
        """计算轨迹效率"""
        if len(trajectory) < 2:
            return 1.0
        
        start_pos = trajectory[0][:2]
        end_pos = trajectory[-1][:2]
        direct_distance = np.linalg.norm(end_pos - start_pos)
        
        if direct_distance < 1e-6:
            return 1.0
        
        path_length = 0.0
        for i in range(1, len(trajectory)):
            path_length += np.linalg.norm(trajectory[i][:2] - trajectory[i-1][:2])
        
        return direct_distance / path_length if path_length > 0 else 0.0
    
    def _compute_density_uniformity(self, density_grid: np.ndarray) -> float:
        """计算密度分布均匀性"""
        non_zero_mask = density_grid > 0
        if not np.any(non_zero_mask):
            return 1.0
        
        non_zero_values = density_grid[non_zero_mask]
        if len(non_zero_values) == 0:
            return 1.0
        
        mean_density = np.mean(non_zero_values)
        std_density = np.std(non_zero_values)
        
        # 变异系数的倒数作为均匀性指标
        if mean_density > 0:
            cv = std_density / mean_density
            return 1.0 / (1.0 + cv)
        
        return 1.0
    
    def _get_robot_corners_optimized(self, robot_pose: np.ndarray) -> List[np.ndarray]:
        """优化的机器人角点计算"""
        x, y, theta = robot_pose[0], robot_pose[1], robot_pose[2]
        
        half_length = self.robot_length / 2.0
        half_width = self.robot_width / 2.0
        
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        # 直接计算世界坐标角点
        corners = []
        local_corners = [
            (-half_length, -half_width),
            (half_length, -half_width),
            (half_length, half_width),
            (-half_length, half_width)
        ]
        
        for lx, ly in local_corners:
            world_x = cos_theta * lx - sin_theta * ly + x
            world_y = sin_theta * lx + cos_theta * ly + y
            corners.append(np.array([world_x, world_y]))
        
        return corners
    
    def analyze_swept_volume_optimization_advanced(self, 
                                                 trajectory_before: List[np.ndarray],
                                                 trajectory_after: List[np.ndarray],
                                                 detailed_analysis: bool = True) -> Dict:
        """
        高级扫掠体积优化分析
        提供详细的对比和改进建议
        """
        analysis_start = time.time()
        
        # 并行计算优化前后的扫掠体积
        if self.enable_parallel:
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_before = executor.submit(
                    self.compute_detailed_swept_volume_optimized, 
                    trajectory_before, None, True, detailed_analysis
                )
                future_after = executor.submit(
                    self.compute_detailed_swept_volume_optimized, 
                    trajectory_after, None, True, detailed_analysis
                )
                
                before_info = future_before.result()
                after_info = future_after.result()
        else:
            before_info = self.compute_detailed_swept_volume_optimized(
                trajectory_before, None, True, detailed_analysis)
            after_info = self.compute_detailed_swept_volume_optimized(
                trajectory_after, None, True, detailed_analysis)
        
        # 计算改进指标
        improvement = self._compute_improvement_metrics(before_info, after_info)
        
        # 质量分析
        quality_analysis = {}
        if detailed_analysis:
            quality_analysis = self._analyze_quality_improvements(before_info, after_info)
        
        # 性能分析
        performance_analysis = {
            'computation_time_before': before_info.computation_time,
            'computation_time_after': after_info.computation_time,
            'total_analysis_time': time.time() - analysis_start
        }
        
        analysis = {
            'before': before_info,
            'after': after_info,
            'improvement': improvement,
            'quality_analysis': quality_analysis,
            'performance_analysis': performance_analysis,
            'recommendations': self._generate_optimization_recommendations(
                before_info, after_info, improvement)
        }
        
        self._print_optimization_summary(analysis)
        return analysis
    
    def _compute_improvement_metrics(self, before: SweptVolumeData, 
                                   after: SweptVolumeData) -> Dict:
        """计算改进指标"""
        improvement = {}
        
        # 面积改进
        volume_reduction = before.area - after.area
        volume_reduction_ratio = volume_reduction / before.area if before.area > 0 else 0.0
        
        improvement['volume_reduction'] = volume_reduction
        improvement['volume_reduction_ratio'] = volume_reduction_ratio
        improvement['percentage_improvement'] = volume_reduction_ratio * 100
        
        # 统计改进
        if before.statistics and after.statistics:
            for key in ['path_length', 'total_time', 'average_speed']:
                if key in before.statistics and key in after.statistics:
                    before_val = before.statistics[key]
                    after_val = after.statistics[key]
                    if before_val > 0:
                        improvement[f'{key}_change_ratio'] = (after_val - before_val) / before_val
        
        return improvement
    
    def _analyze_quality_improvements(self, before: SweptVolumeData, 
                                    after: SweptVolumeData) -> Dict:
        """分析质量改进"""
        quality_analysis = {}
        
        if before.quality_metrics and after.quality_metrics:
            for metric in before.quality_metrics:
                if metric in after.quality_metrics:
                    before_val = before.quality_metrics[metric]
                    after_val = after.quality_metrics[metric]
                    improvement = after_val - before_val
                    quality_analysis[f'{metric}_improvement'] = improvement
                    quality_analysis[f'{metric}_relative_improvement'] = (
                        improvement / before_val if before_val > 0 else 0.0)
        
        return quality_analysis
    
    def _generate_optimization_recommendations(self, before: SweptVolumeData,
                                             after: SweptVolumeData,
                                             improvement: Dict) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        # 基于改进效果的建议
        if improvement.get('percentage_improvement', 0) > 20:
            recommendations.append("优化效果显著，建议应用此优化")
        elif improvement.get('percentage_improvement', 0) > 5:
            recommendations.append("优化效果良好，可以考虑应用")
        else:
            recommendations.append("优化效果有限，可能需要调整参数")
        
        # 基于质量指标的建议
        if before.quality_metrics and after.quality_metrics:
            boundary_smooth_improvement = (
                after.quality_metrics.get('boundary_smoothness', 0) - 
                before.quality_metrics.get('boundary_smoothness', 0)
            )
            if boundary_smooth_improvement > 0.1:
                recommendations.append("边界平滑度显著改善")
            
            trajectory_efficiency_improvement = (
                after.quality_metrics.get('trajectory_efficiency', 0) - 
                before.quality_metrics.get('trajectory_efficiency', 0)
            )
            if trajectory_efficiency_improvement > 0.1:
                recommendations.append("轨迹效率显著提升")
        
        # 性能相关建议
        if after.computation_time < before.computation_time:
            recommendations.append("计算性能有所提升")
        
        return recommendations
    
    def _print_optimization_summary(self, analysis: Dict):
        """打印优化摘要"""
        before = analysis['before']
        after = analysis['after']
        improvement = analysis['improvement']
        
        print(f"\n=== 扫掠体积优化分析报告 ===")
        print(f"优化前面积: {before.area:.3f} m²")
        print(f"优化后面积: {after.area:.3f} m²")
        print(f"面积减少: {improvement['volume_reduction']:.3f} m² "
              f"({improvement['percentage_improvement']:.1f}%)")
        
        if before.statistics and after.statistics:
            print(f"\n轨迹统计:")
            for key in ['path_length', 'total_time', 'average_speed']:
                if key in before.statistics and key in after.statistics:
                    print(f"  {key}: {before.statistics[key]:.3f} -> {after.statistics[key]:.3f}")
        
        print(f"\n计算性能:")
        print(f"  优化前计算时间: {before.computation_time:.3f}s")
        print(f"  优化后计算时间: {after.computation_time:.3f}s")
        
        print(f"\n优化建议:")
        for rec in analysis['recommendations']:
            print(f"  • {rec}")
    
    def compute_swept_volume_for_minco_optimized(self, segments) -> float:
        """
        为MINCO优化提供的高性能扫掠体积计算
        """
        try:
            if not segments:
                return 0.0
            
            # 智能轨迹采样
            trajectory_points = self._sample_minco_trajectory_adaptive(segments)
            
            if not trajectory_points:
                return 0.0
            
            # 使用优化的面积计算
            boundary_points = self.compute_swept_volume_boundary_optimized(
                trajectory_points, use_cache=True)
            
            return self._compute_area_optimized(boundary_points)
            
        except Exception as e:
            print(f"MINCO扫掠体积计算异常: {e}")
            return 1e6
    
    def _sample_minco_trajectory_adaptive(self, segments) -> List[np.ndarray]:
        """自适应MINCO轨迹采样"""
        trajectory_points = []
        current_time = 0.0
        
        for segment in segments:
            duration = segment.duration
            
            # 基于段长度和曲率自适应采样
            num_samples = self._compute_adaptive_samples(segment, duration)
            
            for i in range(num_samples + 1):
                t_local = i * duration / num_samples
                if t_local > duration:
                    t_local = duration
                
                try:
                    pos = segment.evaluate_position(t_local)
                    traj_point = np.array([pos[0], pos[1], pos[2], current_time + t_local])
                    trajectory_points.append(traj_point)
                except Exception as e:
                    print(f"轨迹段评估失败: {e}")
                    continue
            
            current_time += duration
        
        return trajectory_points
    
    def _compute_adaptive_samples(self, segment, duration: float) -> int:
        """计算自适应采样数量"""
        # 基本采样数量
        base_samples = max(5, int(duration / 0.1))
        
        # 基于轨迹复杂度调整
        try:
            # 检查轨迹曲率变化
            start_pos = segment.evaluate_position(0.0)
            mid_pos = segment.evaluate_position(duration / 2.0)
            end_pos = segment.evaluate_position(duration)
            
            # 简单的曲率估计
            if len(start_pos) >= 2 and len(end_pos) >= 2:
                direct_dist = np.linalg.norm(np.array(end_pos[:2]) - np.array(start_pos[:2]))
                path_dist = (np.linalg.norm(np.array(mid_pos[:2]) - np.array(start_pos[:2])) + 
                           np.linalg.norm(np.array(end_pos[:2]) - np.array(mid_pos[:2])))
                
                if direct_dist > 1e-6:
                    curvature_factor = path_dist / direct_dist
                    base_samples = int(base_samples * min(curvature_factor, 3.0))
        
        except Exception:
            pass
        
        return min(base_samples, 100)  # 限制最大采样数
    
    def _generate_trajectory_cache_key(self, trajectory: List[np.ndarray], 
                                     resolution: float) -> str:
        """生成轨迹缓存键"""
        # 使用轨迹的关键特征生成哈希
        if not trajectory:
            return "empty"
        
        # 采样关键点
        key_points = []
        indices = [0, len(trajectory) // 4, len(trajectory) // 2, 
                  3 * len(trajectory) // 4, len(trajectory) - 1]
        
        for i in indices:
            if i < len(trajectory):
                point = trajectory[i]
                key_points.extend([point[0], point[1], point[2]])
        
        # 添加分辨率和机器人参数
        key_points.extend([resolution, self.robot_length, self.robot_width])
        
        # 生成哈希
        key_str = "_".join([f"{x:.3f}" for x in key_points])
        return str(hash(key_str))
    
    def _cache_boundary_result(self, cache_key: str, boundary_points: List[np.ndarray]):
        """缓存边界结果"""
        with self._cache_lock:
            if len(self._boundary_cache) >= self.cache_max_size:
                # 移除最旧的条目
                oldest_key = next(iter(self._boundary_cache))
                del self._boundary_cache[oldest_key]
            
            self._boundary_cache[cache_key] = boundary_points.copy()
    
    def _cache_volume_result(self, cache_key: str, volume_data: SweptVolumeData):
        """缓存体积结果"""
        with self._cache_lock:
            if len(self._volume_cache) >= self.cache_max_size:
                # 移除最旧的条目
                oldest_key = next(iter(self._volume_cache))
                del self._volume_cache[oldest_key]
            
            self._volume_cache[cache_key] = volume_data
    
    def _update_metrics(self, computation_time: float):
        """更新性能指标"""
        self.metrics.total_computations += 1
        self.metrics.total_computation_time += computation_time
        self.metrics.average_computation_time = (
            self.metrics.total_computation_time / self.metrics.total_computations)
        
        # 更新缓存命中率
        total_requests = self.metrics.cache_hits + self.metrics.cache_misses
        if total_requests > 0:
            self.metrics.cache_hit_rate = self.metrics.cache_hits / total_requests
        
        # 记录计算历史
        self._computation_history.append(computation_time)
        if len(self._computation_history) > 100:
            self._computation_history.pop(0)
        
        # 计算并行加速比
        if len(self._computation_history) > 10:
            recent_times = self._computation_history[-10:]
            avg_recent = np.mean(recent_times)
            if hasattr(self, '_serial_baseline') and self._serial_baseline > 0:
                self.metrics.parallel_speedup = self._serial_baseline / avg_recent
    
    def get_performance_metrics_detailed(self) -> SweptVolumeMetrics:
        """获取详细性能指标"""
        # 更新内存使用
        try:
            import psutil
            process = psutil.Process()
            self.metrics.memory_usage_mb = process.memory_info().rss / 1024 / 1024
        except ImportError:
            pass
        
        # 计算网格效率
        if hasattr(self, '_last_grid_size') and hasattr(self, '_last_computation_time'):
            self.metrics.grid_efficiency = (
                self._last_grid_size / self._last_computation_time 
                if self._last_computation_time > 0 else 0.0)
        
        return self.metrics
    
    def optimize_cache_settings(self, target_hit_rate: float = 0.8):
        """优化缓存设置"""
        current_hit_rate = self.metrics.cache_hit_rate
        
        if current_hit_rate < target_hit_rate:
            # 增加缓存大小
            self.cache_max_size = min(self.cache_max_size * 2, 5000)
            print(f"缓存大小增加至: {self.cache_max_size}")
        elif current_hit_rate > 0.95:
            # 可以适当减少缓存大小
            self.cache_max_size = max(self.cache_max_size // 2, 100)
            print(f"缓存大小减少至: {self.cache_max_size}")
    
    def clear_cache_smart(self, keep_recent: int = 10):
        """智能清理缓存"""
        with self._cache_lock:
            # 保留最近使用的缓存项
            if len(self._volume_cache) > keep_recent:
                keys_to_remove = list(self._volume_cache.keys())[:-keep_recent]
                for key in keys_to_remove:
                    del self._volume_cache[key]
            
            if len(self._boundary_cache) > keep_recent:
                keys_to_remove = list(self._boundary_cache.keys())[:-keep_recent]
                for key in keys_to_remove:
                    del self._boundary_cache[key]
            
            if len(self._density_cache) > keep_recent:
                keys_to_remove = list(self._density_cache.keys())[:-keep_recent]
                for key in keys_to_remove:
                    del self._density_cache[key]
        
        print(f"智能缓存清理完成，保留最近 {keep_recent} 项")
    
    def __del__(self):
        """析构函数"""
        if hasattr(self, 'thread_pool') and self.thread_pool:
            self.thread_pool.shutdown(wait=True)
