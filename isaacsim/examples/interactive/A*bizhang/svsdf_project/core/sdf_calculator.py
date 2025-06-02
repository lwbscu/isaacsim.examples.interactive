# core/sdf_calculator.py
"""
签名距离场(SDF)计算器
实现快速SDF计算和扫掠体积分析
"""
import numpy as np
from typing import List, Tuple, Optional
import numba
from concurrent.futures import ThreadPoolExecutor
from utils.math_utils import MathUtils
from utils.config import config

class SDFCalculator:
    """
    高性能SDF计算器
    支持机器人形状SDF和扫掠体积SDF计算
    """
    
    def __init__(self, robot_length: float, robot_width: float, grid_resolution: float = 0.05):
        self.robot_length = robot_length
        self.robot_width = robot_width
        self.grid_resolution = grid_resolution
        
        # 性能优化参数
        self.enable_parallel = True
        self.num_workers = 4
    
    @numba.jit(nopython=True, cache=True)
    def _rectangle_sdf_numba(self, point_local_x: float, point_local_y: float,
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
    
    def compute_robot_sdf(self, query_point: np.ndarray, robot_pose: np.ndarray) -> float:
        """
        计算查询点到机器人的SDF
        
        Args:
            query_point: 查询点世界坐标 [x, y]
            robot_pose: 机器人位姿 [x, y, theta]
            
        Returns:
            SDF值（负值表示在机器人内部）
        """
        # 转换到机器人局部坐标系
        point_local = MathUtils.world_to_robot_frame(query_point, robot_pose)
        
        # 计算矩形SDF
        return self._rectangle_sdf_numba(point_local[0], point_local[1],
                                       self.robot_length, self.robot_width)
    
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
        if not trajectory:
            return float('inf')
        
        # 使用Armijo线搜索找最优时间
        optimal_time = self._armijo_line_search(query_point, trajectory)
        
        # 在最优时间插值机器人位姿
        robot_pose = self._interpolate_robot_pose(trajectory, optimal_time)
        
        # 计算SDF
        return self.compute_robot_sdf(query_point, robot_pose)
    
    def _armijo_line_search(self, query_point: np.ndarray,
                           trajectory: List[np.ndarray],
                           c1: float = 1e-4, alpha: float = 0.5,
                           max_iter: int = 50) -> float:
        """
        Armijo线搜索算法
        找到使SDF最小的时间t*
        """
        if len(trajectory) < 2:
            return trajectory[0][3]
        
        t_min = trajectory[0][3]
        t_max = trajectory[-1][3]
        
        # 初始猜测：中点时间
        t = (t_min + t_max) / 2.0
        step_size = (t_max - t_min) / 10.0
        
        best_t = t
        best_sdf = float('inf')
        
        # 简化的线搜索（实际可以使用更复杂的优化算法）
        for _ in range(max_iter):
            # 当前时间的SDF
            robot_pose = self._interpolate_robot_pose(trajectory, t)
            current_sdf = self.compute_robot_sdf(query_point, robot_pose)
            
            if current_sdf < best_sdf:
                best_sdf = current_sdf
                best_t = t
            
            # 计算梯度（数值微分）
            dt = 1e-6
            robot_pose_plus = self._interpolate_robot_pose(trajectory, t + dt)
            sdf_plus = self.compute_robot_sdf(query_point, robot_pose_plus)
            gradient = (sdf_plus - current_sdf) / dt
            
            # 梯度下降步骤
            new_t = t - step_size * gradient
            new_t = max(t_min, min(t_max, new_t))  # 约束在时间范围内
            
            # Armijo条件检查
            robot_pose_new = self._interpolate_robot_pose(trajectory, new_t)
            new_sdf = self.compute_robot_sdf(query_point, robot_pose_new)
            
            if new_sdf <= current_sdf + c1 * step_size * gradient * gradient:
                t = new_t
            else:
                step_size *= alpha  # 缩减步长
            
            # 收敛检查
            if step_size < 1e-8:
                break
        
        return best_t
    
    def _interpolate_robot_pose(self, trajectory: List[np.ndarray], time: float) -> np.ndarray:
        """轨迹插值获取机器人位姿"""
        if not trajectory:
            return np.zeros(3)
        
        if len(trajectory) == 1:
            return trajectory[0][:3]
        
        # 边界情况
        if time <= trajectory[0][3]:
            return trajectory[0][:3]
        if time >= trajectory[-1][3]:
            return trajectory[-1][:3]
        
        # 线性插值
        for i in range(len(trajectory) - 1):
            t0 = trajectory[i][3]
            t1 = trajectory[i + 1][3]
            
            if t0 <= time <= t1:
                alpha = (time - t0) / (t1 - t0)
                
                pose0 = trajectory[i][:3]
                pose1 = trajectory[i + 1][:3]
                
                # 位置线性插值
                result = np.zeros(3)
                result[:2] = (1.0 - alpha) * pose0[:2] + alpha * pose1[:2]
                
                # 角度插值（考虑周期性）
                theta0 = pose0[2]
                theta1 = pose1[2]
                dtheta = MathUtils.normalize_angle(theta1 - theta0)
                result[2] = MathUtils.normalize_angle(theta0 + alpha * dtheta)
                
                return result
        
        return trajectory[-1][:3]
    
    def compute_swept_volume(self, trajectory: List[np.ndarray], 
                           sampling_density: float = 0.1) -> float:
        """
        计算扫掠体积大小
        使用蒙特卡洛积分方法
        
        Args:
            trajectory: 轨迹点列表
            sampling_density: 采样密度
            
        Returns:
            扫掠体积大小
        """
        if not trajectory:
            return 0.0
        
        # 计算轨迹边界框
        positions = np.array([point[:2] for point in trajectory])
        x_min, y_min = np.min(positions, axis=0)
        x_max, y_max = np.max(positions, axis=0)
        
        # 添加机器人尺寸的边界
        margin = max(self.robot_length, self.robot_width) / 2.0 + 0.5
        x_min -= margin
        x_max += margin
        y_min -= margin
        y_max += margin
        
        # 蒙特卡洛采样
        area = (x_max - x_min) * (y_max - y_min)
        num_samples = int(area / (sampling_density ** 2))
        num_samples = max(1000, min(50000, num_samples))  # 限制采样数量
        
        # 生成随机采样点
        np.random.seed(42)  # 确保可重复性
        sample_x = np.random.uniform(x_min, x_max, num_samples)
        sample_y = np.random.uniform(y_min, y_max, num_samples)
        
        # 并行计算SDF
        inside_count = 0
        if self.enable_parallel:
            inside_count = self._parallel_swept_volume_sampling(
                sample_x, sample_y, trajectory)
        else:
            for i in range(num_samples):
                query_point = np.array([sample_x[i], sample_y[i]])
                sdf = self.compute_swept_volume_sdf(query_point, trajectory)
                if sdf <= 0:
                    inside_count += 1
        
        # 计算体积
        volume = area * inside_count / num_samples
        return volume
    
    def _parallel_swept_volume_sampling(self, sample_x: np.ndarray, 
                                       sample_y: np.ndarray,
                                       trajectory: List[np.ndarray]) -> int:
        """并行扫掠体积采样"""
        def worker(indices):
            count = 0
            for i in indices:
                query_point = np.array([sample_x[i], sample_y[i]])
                sdf = self.compute_swept_volume_sdf(query_point, trajectory)
                if sdf <= 0:
                    count += 1
            return count
        
        # 分割任务
        num_samples = len(sample_x)
        chunk_size = num_samples // self.num_workers
        chunks = []
        
        for i in range(self.num_workers):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < self.num_workers - 1 else num_samples
            chunks.append(range(start_idx, end_idx))
        
        # 并行执行
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(worker, chunk) for chunk in chunks]
            total_count = sum(future.result() for future in futures)
        
        return total_count
    
    def build_environment_sdf_grid(self, obstacles: List[dict],
                                  bounds: np.ndarray) -> np.ndarray:
        """
        构建环境障碍物SDF网格
        
        Args:
            obstacles: 障碍物列表
            bounds: 边界 [x_min, y_min, x_max, y_max]
            
        Returns:
            SDF网格
        """
        x_min, y_min, x_max, y_max = bounds
        
        # 计算网格大小
        grid_width = int((x_max - x_min) / self.grid_resolution) + 1
        grid_height = int((y_max - y_min) / self.grid_resolution) + 1
        
        sdf_grid = np.full((grid_height, grid_width), float('inf'))
        
        # 计算每个网格点的SDF
        for i in range(grid_height):
            for j in range(grid_width):
                world_x = x_min + j * self.grid_resolution
                world_y = y_min + i * self.grid_resolution
                query_point = np.array([world_x, world_y])
                
                min_sdf = float('inf')
                
                # 计算到所有障碍物的最小距离
                for obstacle in obstacles:
                    if obstacle['type'] == 'circle':
                        center = np.array(obstacle['center'])
                        radius = obstacle['radius']
                        distance = MathUtils.euclidean_distance(query_point, center)
                        sdf = distance - radius
                    elif obstacle['type'] == 'rectangle':
                        # 简化的矩形SDF
                        center = np.array(obstacle['center'])
                        size = obstacle['size']
                        local_point = query_point - center
                        sdf = self._rectangle_sdf_numba(local_point[0], local_point[1],
                                                      size[0], size[1])
                    else:
                        continue
                    
                    min_sdf = min(min_sdf, sdf)
                
                sdf_grid[i, j] = min_sdf
        
        return sdf_grid
    
    def compute_obstacle_cost(self, trajectory: List[np.ndarray],
                            obstacles: List[dict], safety_margin: float = 0.2) -> float:
        """
        计算轨迹的障碍物代价
        
        Args:
            trajectory: 轨迹点列表
            obstacles: 障碍物列表
            safety_margin: 安全距离阈值
            
        Returns:
            障碍物代价
        """
        total_cost = 0.0
        
        for traj_point in trajectory:
            robot_pose = traj_point[:3]
            
            # 计算机器人四个角点
            corners = self._get_robot_corners(robot_pose)
            
            # 检查每个角点到障碍物的距离
            for corner in corners:
                for obstacle in obstacles:
                    if obstacle['type'] == 'circle':
                        center = np.array(obstacle['center'])
                        radius = obstacle['radius']
                        distance = MathUtils.euclidean_distance(corner, center)
                        clearance = distance - radius
                    else:
                        continue  # 暂时只支持圆形障碍物
                    
                    # 安全距离违反惩罚
                    if clearance < safety_margin:
                        violation = safety_margin - clearance
                        total_cost += violation ** 3  # 三次惩罚函数
        
        return total_cost
    
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
    
    def compute_sdf_gradient(self, query_point: np.ndarray,
                           robot_pose: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """
        计算SDF梯度（数值微分）
        
        Args:
            query_point: 查询点
            robot_pose: 机器人位姿
            eps: 微分步长
            
        Returns:
            梯度向量 [∂SDF/∂x, ∂SDF/∂y]
        """
        sdf_center = self.compute_robot_sdf(query_point, robot_pose)
        
        # X方向梯度
        query_x_plus = query_point + np.array([eps, 0])
        sdf_x_plus = self.compute_robot_sdf(query_x_plus, robot_pose)
        grad_x = (sdf_x_plus - sdf_center) / eps
        
        # Y方向梯度
        query_y_plus = query_point + np.array([0, eps])
        sdf_y_plus = self.compute_robot_sdf(query_y_plus, robot_pose)
        grad_y = (sdf_y_plus - sdf_center) / eps
        
        return np.array([grad_x, grad_y])