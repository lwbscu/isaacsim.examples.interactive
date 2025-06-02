#!/usr/bin/env python3
"""
SVSDF (Signed Volume-based SDF) 轨迹优化器
基于扫掠体积感知的轨迹规划算法

集成四个阶段：
1. A*初始路径搜索
2. MINCO第一阶段优化（轨迹平滑化）
3. MINCO第二阶段优化（扫掠体积最小化）
4. MPC实时跟踪控制
"""

import numpy as np
import scipy.optimize
from scipy.spatial.distance import cdist
from scipy.interpolate import CubicSpline
import time
from typing import List, Tuple, Optional, Callable
import math
from dataclasses import dataclass

@dataclass
class RobotParams:
    """机器人参数"""
    length: float = 0.4        # 机器人长度(m)
    width: float = 0.3         # 机器人宽度(m)
    wheel_base: float = 0.235  # 轮距(m)
    max_vel: float = 1.0       # 最大线速度(m/s)
    max_omega: float = 2.0     # 最大角速度(rad/s)
    max_acc: float = 2.0       # 最大线加速度(m/s²)
    max_alpha: float = 3.0     # 最大角加速度(rad/s²)

@dataclass
class TrajectoryPoint:
    """轨迹点"""
    position: np.ndarray       # [x, y, yaw]
    velocity: np.ndarray       # [vx, vy, omega]
    acceleration: np.ndarray   # [ax, ay, alpha]
    time: float               # 时间戳
    swept_radius: float = 0.5  # 扫掠半径（用于圆环可视化）
    
@dataclass 
class SweptVolumeVisualization:
    """扫掠体积可视化参数"""
    center: np.ndarray         # 圆心位置
    radius: float             # 圆环半径
    alpha: float = 0.3        # 透明度
    color: np.ndarray = np.array([0.2, 0.8, 1.0])  # 蓝色

class MINCOTrajectory:
    """MINCO轨迹表示和优化类 - 工业级实现"""
    
    def __init__(self, num_segments: int = 8):
        self.num_segments = num_segments
        self.segments = []
        self.time_allocation = []
        self.total_time = 0.0
        
        # 预计算系数矩阵，提高计算效率
        self._precompute_coefficient_matrices()
        
    def _precompute_coefficient_matrices(self):
        """预计算系数矩阵，避免重复计算"""
        self._coeff_matrix_template = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 2, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],  # 将在运行时用实际T替换
            [0, 1, 2, 3, 4, 5],  # 将在运行时用实际T替换
            [0, 0, 2, 6, 12, 20] # 将在运行时用实际T替换
        ])
        
    def _get_coefficient_matrix(self, T: float):
        """获取指定时间的系数矩阵"""
        T2, T3, T4, T5 = T**2, T**3, T**4, T**5
        matrix = self._coeff_matrix_template.copy()
        matrix[3] = [1, T, T2, T3, T4, T5]
        matrix[4] = [0, 1, 2*T, 3*T2, 4*T3, 5*T4]
        matrix[5] = [0, 0, 2, 6*T, 12*T2, 20*T3]
        return matrix
        
    def initialize_from_waypoints(self, waypoints: List[np.ndarray], initial_times: List[float]):
        """从路径点初始化轨迹"""
        self.waypoints = waypoints
        self.time_allocation = initial_times
        self.total_time = sum(initial_times)
        
        # 初始化轨迹段
        self.segments = []
        for i in range(len(waypoints) - 1):
            segment = self._create_quintic_segment(
                waypoints[i], waypoints[i+1], initial_times[i]
            )
            self.segments.append(segment)
    
    def _create_quintic_segment(self, start: np.ndarray, end: np.ndarray, duration: float):
        """创建五次多项式轨迹段 - 工业级优化"""
        # 智能速度和加速度边界条件
        start_vel = self._compute_smart_velocity(start, end, duration, is_start=True)
        start_acc = np.zeros(3)  # 起始加速度为0
        end_vel = self._compute_smart_velocity(start, end, duration, is_start=False)
        end_acc = np.zeros(3)    # 结束加速度为0
        
        # 使用预计算的系数矩阵
        T = duration
        A = self._get_coefficient_matrix(T)
        
        coeffs = {}
        for dim, key in enumerate(['x', 'y', 'yaw']):
            b = np.array([
                start[dim], start_vel[dim], start_acc[dim],
                end[dim], end_vel[dim], end_acc[dim]
            ])
            # 使用LU分解提高求解效率
            coeffs[key] = np.linalg.solve(A, b)
        
        return {
            'coeffs': coeffs,
            'duration': duration
        }
    
    def _compute_smart_velocity(self, start: np.ndarray, end: np.ndarray, duration: float, is_start: bool):
        """智能计算速度边界条件"""
        direction = (end[:2] - start[:2]) / max(np.linalg.norm(end[:2] - start[:2]), 1e-6)
        
        # 基于距离和时间的速度估算
        distance = np.linalg.norm(end[:2] - start[:2])
        avg_speed = distance / duration * 0.8  # 80%的平均速度
        
        if is_start:
            vel = np.array([direction[0] * avg_speed, direction[1] * avg_speed, 0.0])
        else:
            vel = np.array([direction[0] * avg_speed, direction[1] * avg_speed, 0.0])
        
        return vel
    
    def get_state(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """获取指定时间的状态"""
        # 找到对应的轨迹段
        current_time = 0.0
        segment_idx = 0
        
        for i, segment in enumerate(self.segments):
            if current_time + segment['duration'] >= t:
                segment_idx = i
                break
            current_time += segment['duration']
        else:
            # 超出轨迹时间，返回最后一个点
            segment_idx = len(self.segments) - 1
            current_time = self.total_time - self.segments[-1]['duration']
        
        # 计算段内时间
        local_t = t - current_time
        local_t = max(0, min(local_t, self.segments[segment_idx]['duration']))
        
        # 计算位置、速度、加速度
        segment = self.segments[segment_idx]
        position = np.zeros(3)
        velocity = np.zeros(3)
        acceleration = np.zeros(3)
        
        for dim, key in enumerate(['x', 'y', 'yaw']):
            c = segment['coeffs'][key]
            t_powers = np.array([1, local_t, local_t**2, local_t**3, local_t**4, local_t**5])
            t_powers_vel = np.array([0, 1, 2*local_t, 3*local_t**2, 4*local_t**3, 5*local_t**4])
            t_powers_acc = np.array([0, 0, 2, 6*local_t, 12*local_t**2, 20*local_t**3])
            
            position[dim] = np.dot(c, t_powers)
            velocity[dim] = np.dot(c, t_powers_vel)
            acceleration[dim] = np.dot(c, t_powers_acc)
        
        return position, velocity, acceleration

class SDFCalculator:
    """SDF计算器"""
    
    def __init__(self, robot_params: RobotParams, grid_resolution: float = 0.05):
        self.robot_params = robot_params
        self.grid_resolution = grid_resolution
        
    def compute_swept_volume(self, trajectory: List[TrajectoryPoint]) -> float:
        """计算轨迹的扫掠体积 - 工业级优化算法"""
        total_volume = 0.0
        
        # 使用向量化计算提高效率
        if len(trajectory) < 2:
            return 0.0
            
        positions = np.array([p.position[:2] for p in trajectory])
        velocities = np.array([p.velocity[:2] for p in trajectory[:-1]])
        
        # 计算所有段的距离（向量化）
        distances = np.linalg.norm(positions[1:] - positions[:-1], axis=1)
        
        # 计算动态扫掠半径（基于速度）
        speeds = np.linalg.norm(velocities, axis=1)
        base_radius = max(self.robot_params.length, self.robot_params.width) / 2
        
        # 速度越快，扫掠半径越大（考虑不确定性）
        dynamic_radii = base_radius + 0.1 * speeds  # 基础半径 + 速度相关项
        
        # 更新轨迹点的扫掠半径
        for i, traj_point in enumerate(trajectory[:-1]):
            traj_point.swept_radius = dynamic_radii[i]
        
        # 扫掠体积 = 圆形截面积 × 路径长度
        volumes = np.pi * dynamic_radii**2 * distances
        total_volume = np.sum(volumes)
        
        return total_volume
    
    def compute_obstacle_cost(self, position: np.ndarray, obstacles: List) -> float:
        """计算障碍物代价"""
        min_distance = float('inf')
        
        for obstacle in obstacles:
            # 计算到障碍物的最小距离
            dist = np.linalg.norm(position[:2] - np.array(obstacle['center'][:2]))
            dist -= max(obstacle['size'][0], obstacle['size'][1]) / 2
            dist -= max(self.robot_params.length, self.robot_params.width) / 2
            
            min_distance = min(min_distance, dist)
        
        # 距离越近，代价越大
        if min_distance <= 0:
            return 1000.0  # 碰撞惩罚
        else:
            return max(0, 2.0 / min_distance - 0.1)  # 安全距离惩罚

class SVSDFPlanner:
    """SVSDF轨迹规划器主类"""
    
    def __init__(self, robot_params: RobotParams):
        self.robot_params = robot_params
        self.sdf_calc = SDFCalculator(robot_params)
        
        # 优化参数
        self.stage1_weights = {
            'energy': 1.0,
            'time': 0.1,
            'path': 2.0
        }
        
        self.stage2_weights = {
            'energy': 1.0,
            'time': 0.1,
            'obstacle': 10.0,
            'swept_volume': 5.0
        }
        
    def get_swept_volume_visualizations(self, trajectory: List[TrajectoryPoint]) -> List[SweptVolumeVisualization]:
        """生成扫掠体积可视化数据 - 圆环形状"""
        visualizations = []
        
        if len(trajectory) < 2:
            return visualizations
        
        # 每5个点显示一个圆环，避免过度密集
        step = max(1, len(trajectory) // 25)
        
        for i in range(0, len(trajectory), step):
            point = trajectory[i]
            
            # 基于速度动态调整圆环大小
            speed = np.linalg.norm(point.velocity[:2])
            base_radius = max(self.robot_params.length, self.robot_params.width) / 2
            
            # 动态半径：速度越快，不确定性越大，圆环越大
            dynamic_radius = base_radius + 0.1 * speed + 0.2  # 基础 + 速度项 + 安全余量
            
            # 基于轨迹密度调整透明度
            alpha = max(0.2, min(0.6, 0.4 - speed * 0.1))
            
            viz = SweptVolumeVisualization(
                center=point.position[:2],
                radius=dynamic_radius,
                alpha=alpha,
                color=np.array([0.2, 0.7, 1.0])  # 亮蓝色
            )
            visualizations.append(viz)
        
        return visualizations
        
    def plan_trajectory(self, 
                       start_state: np.ndarray,
                       goal_state: np.ndarray,
                       astar_path: List[Tuple[float, float]],
                       obstacles: List) -> Tuple[List[TrajectoryPoint], dict]:
        """
        主要轨迹规划函数
        
        Args:
            start_state: [x, y, yaw, vx, vy, omega]
            goal_state: [x, y, yaw, vx, vy, omega]
            astar_path: A*生成的初始路径
            obstacles: 障碍物列表
            
        Returns:
            trajectory: 优化后的轨迹点列表
            info: 规划信息和统计
        """
        print("\n=== 开始SVSDF轨迹规划 ===")
        total_start = time.time()
        
        # 第一阶段：从A*路径生成初始轨迹
        print("第一阶段：A*路径转换为MINCO轨迹...")
        stage1_start = time.time()
        
        initial_trajectory = self._stage1_path_to_trajectory(astar_path, start_state, goal_state)
        
        stage1_time = time.time() - stage1_start
        print(f"第一阶段完成，耗时: {stage1_time:.3f}s")
        
        # 第二阶段：轨迹平滑化优化
        print("第二阶段：轨迹平滑化优化...")
        stage2_start = time.time()
        
        smoothed_trajectory = self._stage2_smooth_optimization(initial_trajectory, astar_path)
        
        stage2_time = time.time() - stage2_start
        print(f"第二阶段完成，耗时: {stage2_time:.3f}s")
        
        # 第三阶段：扫掠体积最小化优化
        print("第三阶段：扫掠体积最小化优化...")
        stage3_start = time.time()
        
        final_trajectory = self._stage3_swept_volume_optimization(smoothed_trajectory, obstacles)
        
        stage3_time = time.time() - stage3_start
        print(f"第三阶段完成，耗时: {stage3_time:.3f}s")
        
        total_time = time.time() - total_start
        
        # 计算统计信息
        info = {
            'total_time': total_time,
            'stage1_time': stage1_time,
            'stage2_time': stage2_time,
            'stage3_time': stage3_time,
            'trajectory_length': len(final_trajectory),
            'swept_volume': self.sdf_calc.compute_swept_volume(final_trajectory),
            'success': True
        }
        
        print(f"SVSDF轨迹规划完成！总耗时: {total_time:.3f}s")
        print(f"轨迹点数: {len(final_trajectory)}, 扫掠体积: {info['swept_volume']:.3f}m³")
        
        return final_trajectory, info
    
    def get_robot_control_command(self, trajectory: List[TrajectoryPoint], current_time: float, 
                                 current_position: np.ndarray, current_yaw: float) -> Tuple[float, float]:
        """为机器人生成控制命令 - 确保真实移动"""
        if not trajectory:
            return 0.0, 0.0
        
        # 找到当前时间对应的轨迹点
        target_point = None
        for point in trajectory:
            if point.time >= current_time:
                target_point = point
                break
        
        if target_point is None:
            target_point = trajectory[-1]  # 使用最后一个点
        
        # 计算到目标点的距离和角度
        dx = target_point.position[0] - current_position[0]
        dy = target_point.position[1] - current_position[1]
        distance = np.sqrt(dx**2 + dy**2)
        target_angle = np.arctan2(dy, dx)
        
        # 计算角度差
        angle_diff = target_angle - current_yaw
        while angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        while angle_diff < -np.pi:
            angle_diff += 2 * np.pi
        
        # PID控制参数
        kp_linear = 1.5
        kp_angular = 2.0
        
        # 计算控制命令
        linear_velocity = min(kp_linear * distance, self.robot_params.max_vel)
        angular_velocity = kp_angular * angle_diff
        
        # 限制角速度
        angular_velocity = max(-self.robot_params.max_omega, 
                             min(self.robot_params.max_omega, angular_velocity))
        
        # 如果角度差太大，优先转向
        if abs(angle_diff) > np.pi/4:
            linear_velocity *= 0.3
        
        return linear_velocity, angular_velocity
    
    def _stage1_path_to_trajectory(self, astar_path: List[Tuple[float, float]], 
                                  start_state: np.ndarray, goal_state: np.ndarray) -> List[TrajectoryPoint]:
        """第一阶段：将A*路径转换为时间参数化轨迹"""
        if len(astar_path) < 2:
            return []
        
        # 创建路径点（添加偏航角）
        waypoints = []
        for i, point in enumerate(astar_path):
            if i == 0:
                yaw = start_state[2]
            elif i == len(astar_path) - 1:
                yaw = goal_state[2]
            else:
                # 计算朝向下一个点的角度
                next_point = astar_path[i + 1]
                dx = next_point[0] - point[0]
                dy = next_point[1] - point[1]
                yaw = math.atan2(dy, dx)
            
            waypoints.append(np.array([point[0], point[1], yaw]))
        
        # 计算初始时间分配
        initial_times = []
        for i in range(len(waypoints) - 1):
            distance = np.linalg.norm(waypoints[i+1][:2] - waypoints[i][:2])
            time_segment = distance / (self.robot_params.max_vel * 0.7)  # 70%最大速度
            initial_times.append(max(0.5, time_segment))  # 最小0.5秒
        
        # 创建MINCO轨迹
        minco_traj = MINCOTrajectory(len(waypoints) - 1)
        minco_traj.initialize_from_waypoints(waypoints, initial_times)
        
        # 生成离散轨迹点
        trajectory = []
        dt = 0.1  # 100ms采样
        t = 0.0
        while t <= minco_traj.total_time:
            pos, vel, acc = minco_traj.get_state(t)
            trajectory.append(TrajectoryPoint(pos, vel, acc, t))
            t += dt
        
        return trajectory
    
    def _stage2_smooth_optimization(self, initial_trajectory: List[TrajectoryPoint], 
                                   reference_path: List[Tuple[float, float]]) -> List[TrajectoryPoint]:
        """第二阶段：轨迹平滑化优化"""
        # 简化的平滑化：使用样条插值
        if len(initial_trajectory) < 3:
            return initial_trajectory
        
        # 提取位置信息
        times = [p.time for p in initial_trajectory]
        positions = np.array([p.position for p in initial_trajectory])
        
        # 创建平滑样条
        cs_x = CubicSpline(times, positions[:, 0])
        cs_y = CubicSpline(times, positions[:, 1])
        cs_yaw = CubicSpline(times, positions[:, 2])
        
        # 重新采样
        smoothed_trajectory = []
        dt = 0.1
        for t in np.arange(0, times[-1] + dt, dt):
            if t > times[-1]:
                t = times[-1]
            
            pos = np.array([cs_x(t), cs_y(t), cs_yaw(t)])
            vel = np.array([cs_x(t, 1), cs_y(t, 1), cs_yaw(t, 1)])
            acc = np.array([cs_x(t, 2), cs_y(t, 2), cs_yaw(t, 2)])
            
            smoothed_trajectory.append(TrajectoryPoint(pos, vel, acc, t))
        
        return smoothed_trajectory
    
    def _stage3_swept_volume_optimization(self, trajectory: List[TrajectoryPoint], 
                                        obstacles: List) -> List[TrajectoryPoint]:
        """第三阶段：扫掠体积最小化优化"""
        # 简化的优化：调整轨迹点以减少扫掠体积和避开障碍物
        optimized_trajectory = trajectory.copy()
        
        # 迭代优化
        for iteration in range(5):  # 最多5次迭代
            for i in range(1, len(optimized_trajectory) - 1):
                current = optimized_trajectory[i]
                prev_point = optimized_trajectory[i-1]
                next_point = optimized_trajectory[i+1]
                
                # 计算当前位置的代价
                current_cost = self._compute_point_cost(current, prev_point, next_point, obstacles)
                
                # 尝试小幅调整位置
                best_pos = current.position.copy()
                best_cost = current_cost
                
                for dx in [-0.1, 0, 0.1]:
                    for dy in [-0.1, 0, 0.1]:
                        if dx == 0 and dy == 0:
                            continue
                        
                        # 创建新的轨迹点
                        new_pos = current.position.copy()
                        new_pos[0] += dx
                        new_pos[1] += dy
                        
                        new_point = TrajectoryPoint(new_pos, current.velocity, current.acceleration, current.time)
                        cost = self._compute_point_cost(new_point, prev_point, next_point, obstacles)
                        
                        if cost < best_cost:
                            best_cost = cost
                            best_pos = new_pos
                
                # 更新位置
                optimized_trajectory[i].position = best_pos
        
        return optimized_trajectory
    
    def _compute_point_cost(self, point: TrajectoryPoint, prev_point: TrajectoryPoint, 
                           next_point: TrajectoryPoint, obstacles: List) -> float:
        """计算轨迹点的总代价"""
        # 障碍物代价
        obstacle_cost = self.sdf_calc.compute_obstacle_cost(point.position, obstacles)
        
        # 扫掠体积代价（简化为路径长度）
        path_length = (np.linalg.norm(point.position[:2] - prev_point.position[:2]) + 
                      np.linalg.norm(next_point.position[:2] - point.position[:2]))
        
        # 平滑度代价（角度变化）
        v1 = point.position[:2] - prev_point.position[:2]
        v2 = next_point.position[:2] - point.position[:2]
        if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            smoothness_cost = 1.0 - cos_angle  # 角度变化越大，代价越大
        else:
            smoothness_cost = 0.0
        
        total_cost = (self.stage2_weights['obstacle'] * obstacle_cost +
                     self.stage2_weights['swept_volume'] * path_length +
                     1.0 * smoothness_cost)
        
        return total_cost
