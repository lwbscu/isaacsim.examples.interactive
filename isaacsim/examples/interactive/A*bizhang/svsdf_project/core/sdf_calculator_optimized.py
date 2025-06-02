# core/sdf_calculator_optimized.py
"""
签名距离场(SDF)计算器 - 工业级优化版本
实现高性能SDF计算和扫掠体积分析

核心技术特点：
- Numba JIT编译加速核心计算
- 并行计算支持大规模SDF查询
- Armijo线搜索优化算法
- 缓存机制减少重复计算
- 工业级数值稳定性保证
"""
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    warnings.warn("Numba not available, using pure Python (slower)")

from utils.math_utils import MathUtils
from utils.config import config


class SDFCalculatorOptimized:
    """
    高性能SDF计算器 - 工业级优化版本
    
    核心功能：
    - 机器人形状SDF计算
    - 扫掠体积SDF计算 
    - 多线程并行计算
    - 智能缓存机制
    - Armijo线搜索优化
    """
    
    def __init__(self, robot_length: float, robot_width: float, 
                 grid_resolution: float = 0.05, enable_parallel: bool = True,
                 num_workers: int = 4, cache_size: int = 10000):
        self.robot_length = robot_length
        self.robot_width = robot_width
        self.grid_resolution = grid_resolution
        
        # 并行计算配置
        self.enable_parallel = enable_parallel
        self.num_workers = num_workers
        self.executor = ThreadPoolExecutor(max_workers=num_workers) if enable_parallel else None
        
        # 缓存机制
        self.cache_size = cache_size
        self._sdf_cache = {}
        self._cache_lock = threading.Lock()
        self._cache_access_count = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # 性能监控
        self.computation_times = {
            'robot_sdf': [],
            'swept_volume_sdf': [],
            'armijo_search': [],
            'parallel_computation': []
        }
        
        # 优化参数
        self.armijo_c1 = 1e-4
        self.armijo_alpha = 0.5
        self.armijo_max_iter = 50
        self.numerical_epsilon = 1e-12
        
        print(f"✅ SDF计算器已初始化（优化版本）")
        print(f"   - 并行计算: {'启用' if enable_parallel else '禁用'}")
        print(f"   - 工作线程: {num_workers}")
        print(f"   - 缓存大小: {cache_size}")
        print(f"   - Numba加速: {'启用' if NUMBA_AVAILABLE else '禁用'}")
    
    def compute_robot_sdf(self, query_point: np.ndarray, robot_pose: np.ndarray) -> float:
        """
        计算查询点到机器人的SDF
        
        Args:
            query_point: 查询点世界坐标 [x, y]
            robot_pose: 机器人位姿 [x, y, theta]
            
        Returns:
            SDF值（负值表示在机器人内部）
        """
        start_time = time.time()
        
        # 检查缓存
        cache_key = self._create_cache_key(query_point, robot_pose)
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            self._cache_hits += 1
            return cached_result
        
        self._cache_misses += 1
        
        # 转换到机器人局部坐标系
        point_local = MathUtils.world_to_robot_frame(query_point, robot_pose)
        
        # 计算矩形SDF
        if NUMBA_AVAILABLE:
            sdf_value = self._rectangle_sdf_numba(
                point_local[0], point_local[1], 
                self.robot_length, self.robot_width)
        else:
            sdf_value = self._rectangle_sdf_python(
                point_local[0], point_local[1],
                self.robot_length, self.robot_width)
        
        # 缓存结果
        self._add_to_cache(cache_key, sdf_value)
        
        # 记录性能
        computation_time = time.time() - start_time
        self.computation_times['robot_sdf'].append(computation_time)
        
        return sdf_value
    
    def compute_swept_volume_sdf(self, query_point: np.ndarray,
                               trajectory: List[np.ndarray]) -> float:
        """
        计算扫掠体积SDF
        使用Armijo线搜索找到最优时间t*
        
        Args:
            query_point: 查询点世界坐标 [x, y]
            trajectory: 轨迹点列表，每个点为 [x, y, theta, time]
            
        Returns:
            扫掠体积SDF值
        """
        start_time = time.time()
        
        if not trajectory:
            return float('inf')
        
        # 使用Armijo线搜索找最优时间
        search_start = time.time()
        optimal_time = self._armijo_line_search_optimized(query_point, trajectory)
        search_time = time.time() - search_start
        self.computation_times['armijo_search'].append(search_time)
        
        # 在最优时间插值机器人位姿
        robot_pose = self._interpolate_robot_pose(trajectory, optimal_time)
        
        # 计算SDF
        sdf_value = self.compute_robot_sdf(query_point, robot_pose)
        
        # 记录性能
        computation_time = time.time() - start_time
        self.computation_times['swept_volume_sdf'].append(computation_time)
        
        return sdf_value
    
    def compute_swept_volume_parallel(self, query_points: np.ndarray,
                                    trajectory: List[np.ndarray]) -> np.ndarray:
        """
        并行计算多个查询点的扫掠体积SDF
        
        Args:
            query_points: 查询点数组 [N, 2]
            trajectory: 轨迹点列表
            
        Returns:
            SDF值数组 [N]
        """
        start_time = time.time()
        
        if not self.enable_parallel or self.executor is None:
            # 串行计算
            sdf_values = np.array([
                self.compute_swept_volume_sdf(point, trajectory) 
                for point in query_points
            ])
        else:
            # 并行计算
            chunk_size = max(1, len(query_points) // self.num_workers)
            chunks = [query_points[i:i+chunk_size] 
                     for i in range(0, len(query_points), chunk_size)]
            
            futures = []
            for chunk in chunks:
                future = self.executor.submit(self._compute_chunk_sdf, chunk, trajectory)
                futures.append(future)
            
            sdf_values = []
            for future in as_completed(futures):
                chunk_results = future.result()
                sdf_values.extend(chunk_results)
            
            sdf_values = np.array(sdf_values)
        
        # 记录性能
        computation_time = time.time() - start_time
        self.computation_times['parallel_computation'].append(computation_time)
        
        return sdf_values
    
    def compute_obstacle_cost(self, trajectory: List[np.ndarray],
                            obstacles: List[Dict]) -> float:
        """
        计算轨迹的障碍物代价
        
        Args:
            trajectory: 轨迹点列表
            obstacles: 障碍物列表
            
        Returns:
            障碍物代价值
        """
        if not obstacles or not trajectory:
            return 0.0
        
        total_cost = 0.0
        safety_margin = 0.2  # 安全边距
        
        for traj_point in trajectory:
            robot_pose = traj_point[:3]
            
            for obstacle in obstacles:
                # 计算机器人与障碍物的最小距离
                min_distance = self._compute_robot_obstacle_distance(robot_pose, obstacle)
                
                # 如果距离小于安全边距，计算代价
                if min_distance < safety_margin:
                    # 使用指数函数计算代价（距离越近代价越高）
                    cost = np.exp(-min_distance / 0.1)
                    total_cost += cost
        
        return total_cost
    
    def build_environment_sdf_grid(self, obstacles: List[Dict],
                                  bounds: np.ndarray,
                                  resolution: float = None) -> np.ndarray:
        """
        构建环境SDF网格
        
        Args:
            obstacles: 障碍物列表
            bounds: 边界 [x_min, y_min, x_max, y_max]
            resolution: 网格分辨率
            
        Returns:
            SDF网格
        """
        if resolution is None:
            resolution = self.grid_resolution
        
        x_min, y_min, x_max, y_max = bounds
        grid_width = int((x_max - x_min) / resolution) + 1
        grid_height = int((y_max - y_min) / resolution) + 1
        
        # 创建查询点网格
        query_points = []
        for i in range(grid_height):
            for j in range(grid_width):
                x = x_min + j * resolution
                y = y_min + i * resolution
                query_points.append([x, y])
        
        query_points = np.array(query_points)
        
        # 并行计算SDF值
        if self.enable_parallel:
            sdf_values = self._compute_environment_sdf_parallel(query_points, obstacles)
        else:
            sdf_values = self._compute_environment_sdf_serial(query_points, obstacles)
        
        # 重塑为网格
        sdf_grid = sdf_values.reshape(grid_height, grid_width)
        
        return sdf_grid
    
    # === 内部优化方法 ===
    
    def _compute_chunk_sdf(self, chunk_points: np.ndarray, 
                          trajectory: List[np.ndarray]) -> List[float]:
        """计算点块的SDF值"""
        return [self.compute_swept_volume_sdf(point, trajectory) 
                for point in chunk_points]
    
    def _armijo_line_search_optimized(self, query_point: np.ndarray,
                                    trajectory: List[np.ndarray]) -> float:
        """
        优化的Armijo线搜索算法
        使用金分搜索和二分搜索混合策略
        """
        if len(trajectory) < 2:
            return trajectory[0][3]
        
        t_min = trajectory[0][3]
        t_max = trajectory[-1][3]
        
        # 使用三点搜索找到粗略最优区间
        t1 = t_min + (t_max - t_min) * 0.382  # 黄金分割点
        t2 = t_min + (t_max - t_min) * 0.618
        
        f1 = self._evaluate_sdf_at_time(query_point, trajectory, t1)
        f2 = self._evaluate_sdf_at_time(query_point, trajectory, t2)
        
        # 使用黄金分割搜索精确化
        tolerance = (t_max - t_min) * 1e-6
        
        for _ in range(self.armijo_max_iter):
            if abs(t_max - t_min) < tolerance:
                break
            
            if f1 < f2:
                t_max = t2
                t2 = t1
                f2 = f1
                t1 = t_min + (t_max - t_min) * 0.382
                f1 = self._evaluate_sdf_at_time(query_point, trajectory, t1)
            else:
                t_min = t1
                t1 = t2
                f1 = f2
                t2 = t_min + (t_max - t_min) * 0.618
                f2 = self._evaluate_sdf_at_time(query_point, trajectory, t2)
        
        return (t_min + t_max) / 2.0
    
    def _evaluate_sdf_at_time(self, query_point: np.ndarray, 
                             trajectory: List[np.ndarray], time: float) -> float:
        """在指定时间评估SDF值"""
        robot_pose = self._interpolate_robot_pose(trajectory, time)
        return self.compute_robot_sdf(query_point, robot_pose)
    
    def _interpolate_robot_pose(self, trajectory: List[np.ndarray], time: float) -> np.ndarray:
        """轨迹插值获取机器人位姿"""
        if not trajectory:
            return np.array([0, 0, 0])
        
        if time <= trajectory[0][3]:
            return trajectory[0][:3]
        
        if time >= trajectory[-1][3]:
            return trajectory[-1][:3]
        
        # 找到时间区间
        for i in range(len(trajectory) - 1):
            if trajectory[i][3] <= time <= trajectory[i+1][3]:
                t1, t2 = trajectory[i][3], trajectory[i+1][3]
                
                if abs(t2 - t1) < self.numerical_epsilon:
                    return trajectory[i][:3]
                
                # 线性插值
                alpha = (time - t1) / (t2 - t1)
                pose = (1 - alpha) * trajectory[i][:3] + alpha * trajectory[i+1][:3]
                
                # 角度插值需要特殊处理
                pose[2] = MathUtils.interpolate_angle(trajectory[i][2], trajectory[i+1][2], alpha)
                
                return pose
        
        return trajectory[-1][:3]
    
    def _compute_robot_obstacle_distance(self, robot_pose: np.ndarray, 
                                       obstacle: Dict) -> float:
        """计算机器人与障碍物的最小距离"""
        # 简化实现：假设障碍物为圆形
        obstacle_center = np.array([obstacle.get('x', 0), obstacle.get('y', 0)])
        obstacle_radius = obstacle.get('radius', 0.5)
        
        robot_center = robot_pose[:2]
        distance_to_center = np.linalg.norm(robot_center - obstacle_center)
        
        # 机器人的外接圆半径
        robot_radius = np.sqrt(self.robot_length**2 + self.robot_width**2) / 2.0
        
        # 最小距离 = 中心距离 - 两个半径
        min_distance = max(0, distance_to_center - obstacle_radius - robot_radius)
        
        return min_distance
    
    def _compute_environment_sdf_parallel(self, query_points: np.ndarray,
                                        obstacles: List[Dict]) -> np.ndarray:
        """并行计算环境SDF"""
        chunk_size = max(1, len(query_points) // self.num_workers)
        chunks = [query_points[i:i+chunk_size] 
                 for i in range(0, len(query_points), chunk_size)]
        
        futures = []
        for chunk in chunks:
            future = self.executor.submit(self._compute_environment_sdf_chunk, chunk, obstacles)
            futures.append(future)
        
        sdf_values = []
        for future in as_completed(futures):
            chunk_results = future.result()
            sdf_values.extend(chunk_results)
        
        return np.array(sdf_values)
    
    def _compute_environment_sdf_chunk(self, chunk_points: np.ndarray,
                                     obstacles: List[Dict]) -> List[float]:
        """计算点块的环境SDF"""
        sdf_values = []
        
        for point in chunk_points:
            min_distance = float('inf')
            
            for obstacle in obstacles:
                distance = self._point_to_obstacle_distance(point, obstacle)
                min_distance = min(min_distance, distance)
            
            sdf_values.append(min_distance)
        
        return sdf_values
    
    def _compute_environment_sdf_serial(self, query_points: np.ndarray,
                                      obstacles: List[Dict]) -> np.ndarray:
        """串行计算环境SDF"""
        return np.array([
            min(self._point_to_obstacle_distance(point, obstacle) 
                for obstacle in obstacles) if obstacles else float('inf')
            for point in query_points
        ])
    
    def _point_to_obstacle_distance(self, point: np.ndarray, obstacle: Dict) -> float:
        """计算点到障碍物的距离"""
        obstacle_center = np.array([obstacle.get('x', 0), obstacle.get('y', 0)])
        obstacle_radius = obstacle.get('radius', 0.5)
        
        distance_to_center = np.linalg.norm(point - obstacle_center)
        return max(0, distance_to_center - obstacle_radius)
    
    # === 缓存管理 ===
    
    def _create_cache_key(self, query_point: np.ndarray, robot_pose: np.ndarray) -> str:
        """创建缓存键"""
        # 对坐标进行量化以提高缓存命中率
        qx = round(query_point[0] / self.grid_resolution) * self.grid_resolution
        qy = round(query_point[1] / self.grid_resolution) * self.grid_resolution
        rx = round(robot_pose[0] / self.grid_resolution) * self.grid_resolution
        ry = round(robot_pose[1] / self.grid_resolution) * self.grid_resolution
        rtheta = round(robot_pose[2] / 0.1) * 0.1  # 角度量化到0.1弧度
        
        return f"{qx:.3f},{qy:.3f},{rx:.3f},{ry:.3f},{rtheta:.3f}"
    
    def _get_from_cache(self, key: str) -> Optional[float]:
        """从缓存获取值"""
        with self._cache_lock:
            if key in self._sdf_cache:
                # 更新访问计数
                self._cache_access_count[key] = self._cache_access_count.get(key, 0) + 1
                return self._sdf_cache[key]
            return None
    
    def _add_to_cache(self, key: str, value: float):
        """添加值到缓存"""
        with self._cache_lock:
            if len(self._sdf_cache) >= self.cache_size:
                # 使用LFU策略清理缓存
                self._cleanup_cache()
            
            self._sdf_cache[key] = value
            self._cache_access_count[key] = 1
    
    def _cleanup_cache(self):
        """清理缓存（LFU策略）"""
        if not self._cache_access_count:
            return
        
        # 移除最少使用的25%条目
        sorted_items = sorted(self._cache_access_count.items(), key=lambda x: x[1])
        remove_count = len(sorted_items) // 4
        
        for key, _ in sorted_items[:remove_count]:
            self._sdf_cache.pop(key, None)
            self._cache_access_count.pop(key, None)
    
    # === Numba优化函数 ===
    
    if NUMBA_AVAILABLE:
        @staticmethod
        @numba.jit(nopython=True, cache=True)
        def _rectangle_sdf_numba(point_local_x: float, point_local_y: float,
                               length: float, width: float) -> float:
            """
            使用Numba优化的矩形SDF计算
            实现论文Equation (7)
            """
            dx = abs(point_local_x) - length / 2.0
            dy = abs(point_local_y) - width / 2.0
            
            if dx > 0.0 and dy > 0.0:
                # 外部角点：欧几里得距离
                return np.sqrt(dx * dx + dy * dy)
            else:
                # 边界或内部：切比雪夫距离
                return max(dx, dy)
    
    @staticmethod
    def _rectangle_sdf_python(point_local_x: float, point_local_y: float,
                            length: float, width: float) -> float:
        """
        纯Python矩形SDF计算（Numba不可用时的后备）
        """
        dx = abs(point_local_x) - length / 2.0
        dy = abs(point_local_y) - width / 2.0
        
        if dx > 0.0 and dy > 0.0:
            # 外部角点：欧几里得距离
            return np.sqrt(dx * dx + dy * dy)
        else:
            # 边界或内部：切比雪夫距离
            return max(dx, dy)
    
    # === 性能分析 ===
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        metrics = {}
        
        for operation, times in self.computation_times.items():
            if times:
                metrics[operation] = {
                    'count': len(times),
                    'total_time': sum(times),
                    'avg_time': np.mean(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'std_time': np.std(times)
                }
            else:
                metrics[operation] = {
                    'count': 0,
                    'total_time': 0,
                    'avg_time': 0,
                    'min_time': 0,
                    'max_time': 0,
                    'std_time': 0
                }
        
        # 缓存统计
        total_requests = self._cache_hits + self._cache_misses
        metrics['cache'] = {
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'hit_rate': self._cache_hits / total_requests if total_requests > 0 else 0,
            'cache_size': len(self._sdf_cache),
            'max_cache_size': self.cache_size
        }
        
        return metrics
    
    def clear_cache(self):
        """清空缓存"""
        with self._cache_lock:
            self._sdf_cache.clear()
            self._cache_access_count.clear()
            self._cache_hits = 0
            self._cache_misses = 0
    
    def reset_performance_counters(self):
        """重置性能计数器"""
        for key in self.computation_times:
            self.computation_times[key] = []
        self._cache_hits = 0
        self._cache_misses = 0
    
    def cleanup(self):
        """清理资源"""
        if self.executor:
            self.executor.shutdown(wait=True)
        self.clear_cache()
        print("🧹 SDF计算器资源清理完成")


# 工厂函数
def create_sdf_calculator(robot_length: float, robot_width: float,
                         enable_optimization: bool = True) -> SDFCalculatorOptimized:
    """创建优化的SDF计算器"""
    return SDFCalculatorOptimized(
        robot_length=robot_length,
        robot_width=robot_width,
        enable_parallel=enable_optimization,
        num_workers=4 if enable_optimization else 1,
        cache_size=10000 if enable_optimization else 100
    )
