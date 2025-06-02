# core/minco_trajectory.py
"""
MINCO轨迹优化实现
实现论文中第二、三阶段的轨迹优化
"""
import numpy as np
from typing import List, Tuple, Optional, Callable
from scipy.optimize import minimize
from dataclasses import dataclass
from utils.math_utils import MathUtils, OptimizationUtils
from utils.config import config

@dataclass
class TrajectorySegment:
    """轨迹段数据结构"""
    coeffs_x: np.ndarray    # x方向多项式系数
    coeffs_y: np.ndarray    # y方向多项式系数
    coeffs_yaw: np.ndarray  # 偏航角多项式系数
    duration: float         # 段持续时间
    
    def evaluate_position(self, t: float) -> np.ndarray:
        """计算位置"""
        t_powers = np.array([1, t, t**2, t**3, t**4, t**5])
        x = np.dot(self.coeffs_x, t_powers)
        y = np.dot(self.coeffs_y, t_powers)
        yaw = np.dot(self.coeffs_yaw, t_powers)
        return np.array([x, y, yaw])
    
    def evaluate_velocity(self, t: float) -> np.ndarray:
        """计算速度"""
        t_powers = np.array([0, 1, 2*t, 3*t**2, 4*t**3, 5*t**4])
        vx = np.dot(self.coeffs_x, t_powers)
        vy = np.dot(self.coeffs_y, t_powers)
        vyaw = np.dot(self.coeffs_yaw, t_powers)
        return np.array([vx, vy, vyaw])
    
    def evaluate_acceleration(self, t: float) -> np.ndarray:
        """计算加速度"""
        t_powers = np.array([0, 0, 2, 6*t, 12*t**2, 20*t**3])
        ax = np.dot(self.coeffs_x, t_powers)
        ay = np.dot(self.coeffs_y, t_powers)
        ayaw = np.dot(self.coeffs_yaw, t_powers)
        return np.array([ax, ay, ayaw])

class MINCOTrajectory:
    """
    MINCO轨迹优化器
    实现最小控制能量的稀疏轨迹参数化
    """
    
    def __init__(self, num_segments: int = 8):
        self.num_segments = num_segments
        self.segments: List[TrajectorySegment] = []
        
        # 优化变量：控制点和时间分配
        self.control_points = np.zeros((num_segments - 1, 3))  # (N-1) x 3 控制点
        self.time_allocation = np.ones(num_segments)           # N 时间分配
        
        # 边界条件
        self.start_state = np.zeros((3, 3))  # [pos, vel, acc] at start
        self.end_state = np.zeros((3, 3))    # [pos, vel, acc] at end
        
    def initialize_from_waypoints(self, waypoints: List[np.ndarray], 
                                 initial_times: List[float]):
        """从路径点初始化轨迹"""
        if len(waypoints) != self.num_segments + 1:
            raise ValueError(f"需要 {self.num_segments + 1} 个路径点")
        
        if len(initial_times) != self.num_segments:
            raise ValueError(f"需要 {self.num_segments} 个时间段")
        
        # 设置控制点（使用路径点作为初始控制点）
        for i in range(self.num_segments - 1):
            self.control_points[i] = waypoints[i + 1]
        
        # 设置时间分配
        self.time_allocation = np.array(initial_times)
        
        # 设置边界条件
        self.start_state[0] = waypoints[0]     # 起点位置
        self.start_state[1] = np.zeros(3)      # 起点速度为零
        self.start_state[2] = np.zeros(3)      # 起点加速度为零
        
        self.end_state[0] = waypoints[-1]      # 终点位置
        self.end_state[1] = np.zeros(3)        # 终点速度为零
        self.end_state[2] = np.zeros(3)        # 终点加速度为零
        
        # 生成初始轨迹
        self._update_coefficients()
    
    def _update_coefficients(self):
        """根据控制点和时间分配更新多项式系数"""
        self.segments = []
        
        for seg_idx in range(self.num_segments):
            # 获取边界条件
            if seg_idx == 0:
                # 第一段：起点边界条件
                p0, v0, a0 = self.start_state[0], self.start_state[1], self.start_state[2]
            else:
                # 中间段：从前一段继承
                prev_seg = self.segments[seg_idx - 1]
                T_prev = prev_seg.duration
                p0 = prev_seg.evaluate_position(T_prev)
                v0 = prev_seg.evaluate_velocity(T_prev)
                a0 = prev_seg.evaluate_acceleration(T_prev)
            
            if seg_idx == self.num_segments - 1:
                # 最后一段：终点边界条件
                p1, v1, a1 = self.end_state[0], self.end_state[1], self.end_state[2]
            else:
                # 中间段：连接到控制点
                p1 = self.control_points[seg_idx]
                v1 = np.zeros(3)  # 简化：控制点处速度为零
                a1 = np.zeros(3)  # 简化：控制点处加速度为零
            
            # 当前段持续时间
            T = self.time_allocation[seg_idx]
            
            # 构建约束矩阵和向量
            # 6个约束：p(0), v(0), a(0), p(T), v(T), a(T)
            A = np.array([
                [1, 0, 0, 0, 0, 0],           # p(0)
                [0, 1, 0, 0, 0, 0],           # v(0)
                [0, 0, 2, 0, 0, 0],           # a(0)
                [1, T, T**2, T**3, T**4, T**5],   # p(T)
                [0, 1, 2*T, 3*T**2, 4*T**3, 5*T**4],  # v(T)
                [0, 0, 2, 6*T, 12*T**2, 20*T**3]      # a(T)
            ])
            
            # 分别求解 x, y, yaw 的系数
            coeffs_x = np.linalg.solve(A, np.array([p0[0], v0[0], a0[0], p1[0], v1[0], a1[0]]))
            coeffs_y = np.linalg.solve(A, np.array([p0[1], v0[1], a0[1], p1[1], v1[1], a1[1]]))
            coeffs_yaw = np.linalg.solve(A, np.array([p0[2], v0[2], a0[2], p1[2], v1[2], a1[2]]))
            
            # 创建轨迹段
            segment = TrajectorySegment(coeffs_x, coeffs_y, coeffs_yaw, T)
            self.segments.append(segment)
    
    def optimize_stage1(self, weight_energy: float, weight_time: float, 
                       weight_path: float, reference_path: List[np.ndarray]) -> bool:
        """
        第一阶段优化：平滑化轨迹
        最小化：J = W_E * J_E + W_T * J_T + W_P * J_P
        """
        print("开始MINCO第一阶段优化（平滑化）...")
        
        # 组装优化变量：[control_points_flat; time_allocation]
        dim_q = self.control_points.size
        dim_T = self.time_allocation.size
        
        x0 = np.concatenate([self.control_points.flatten(), self.time_allocation])
        
        def objective_function(x):
            # 解包优化变量
            control_points = x[:dim_q].reshape(self.num_segments - 1, 3)
            time_allocation = x[dim_q:dim_q + dim_T]
            
            # 临时更新
            self.control_points = control_points
            self.time_allocation = time_allocation
            self._update_coefficients()
            
            # 计算各项代价
            J_E = self._compute_energy_cost()
            J_T = self._compute_time_cost()
            J_P = self._compute_path_deviation_cost(reference_path)
            
            total_cost = weight_energy * J_E + weight_time * J_T + weight_path * J_P
            return total_cost
        
        # 约束：时间分配必须为正
        constraints = []
        for i in range(self.num_segments):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, idx=i: x[dim_q + idx] - 0.1  # 最小时间0.1秒
            })
        
        # 执行优化
        try:
            result = minimize(objective_function, x0, method='SLSQP',
                            constraints=constraints,
                            options={'maxiter': config.planning.max_opt_iterations,
                                   'ftol': config.planning.convergence_tolerance})
            
            if result.success:
                # 更新最优解
                optimal_control_points = result.x[:dim_q].reshape(self.num_segments - 1, 3)
                optimal_time_allocation = result.x[dim_q:dim_q + dim_T]
                
                self.control_points = optimal_control_points
                self.time_allocation = optimal_time_allocation
                self._update_coefficients()
                
                print(f"第一阶段优化成功，代价: {result.fun:.6f}")
                return True
            else:
                print(f"第一阶段优化失败: {result.message}")
                return False
                
        except Exception as e:
            print(f"第一阶段优化异常: {e}")
            return False
    
    def optimize_stage2(self, weight_energy: float, weight_time: float,
                       weight_obstacle: float, weight_swept_volume: float,
                       obstacle_cost_func: Callable,
                       swept_volume_cost_func: Callable) -> bool:
        """
        第二阶段优化：扫掠体积最小化
        最小化：J = W_E * J_E + W_T * J_T + W_ob * J_ob + W_sv * J_sv
        """
        print("开始MINCO第二阶段优化（扫掠体积最小化）...")
        
        dim_q = self.control_points.size
        dim_T = self.time_allocation.size
        
        x0 = np.concatenate([self.control_points.flatten(), self.time_allocation])
        
        def objective_function(x):
            # 解包优化变量
            control_points = x[:dim_q].reshape(self.num_segments - 1, 3)
            time_allocation = x[dim_q:dim_q + dim_T]
            
            # 临时更新
            old_control_points = self.control_points.copy()
            old_time_allocation = self.time_allocation.copy()
            
            self.control_points = control_points
            self.time_allocation = time_allocation
            self._update_coefficients()
            
            try:
                # 计算各项代价
                J_E = self._compute_energy_cost()
                J_T = self._compute_time_cost()
                J_ob = self._compute_obstacle_cost(obstacle_cost_func)
                J_sv = self._compute_swept_volume_cost(swept_volume_cost_func)
                
                total_cost = (weight_energy * J_E + weight_time * J_T + 
                            weight_obstacle * J_ob + weight_swept_volume * J_sv)
                
                return total_cost
                
            except Exception as e:
                # 恢复原始值
                self.control_points = old_control_points
                self.time_allocation = old_time_allocation
                self._update_coefficients()
                print(f"代价函数计算异常: {e}")
                return 1e6  # 返回大值
        
        # 约束
        constraints = []
        for i in range(self.num_segments):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, idx=i: x[dim_q + idx] - 0.1
            })
        
        # 执行优化
        try:
            result = minimize(objective_function, x0, method='SLSQP',
                            constraints=constraints,
                            options={'maxiter': config.planning.max_opt_iterations,
                                   'ftol': config.planning.convergence_tolerance})
            
            if result.success:
                # 更新最优解
                optimal_control_points = result.x[:dim_q].reshape(self.num_segments - 1, 3)
                optimal_time_allocation = result.x[dim_q:dim_q + dim_T]
                
                self.control_points = optimal_control_points
                self.time_allocation = optimal_time_allocation
                self._update_coefficients()
                
                print(f"第二阶段优化成功，代价: {result.fun:.6f}")
                return True
            else:
                print(f"第二阶段优化失败: {result.message}")
                return False
                
        except Exception as e:
            print(f"第二阶段优化异常: {e}")
            return False
    
    def _compute_energy_cost(self) -> float:
        """计算能量代价 J_E = ∫||u||²dt"""
        total_energy = 0.0
        
        for segment in self.segments:
            T = segment.duration
            
            # 计算加速度的积分（简化为数值积分）
            num_samples = 10
            dt = T / num_samples
            
            for i in range(num_samples):
                t = i * dt
                acc = segment.evaluate_acceleration(t)
                total_energy += np.dot(acc, acc) * dt
        
        return total_energy
    
    def _compute_time_cost(self) -> float:
        """计算时间代价 J_T = ∑T_i"""
        return np.sum(self.time_allocation)
    
    def _compute_path_deviation_cost(self, reference_path: List[np.ndarray]) -> float:
        """计算路径偏差代价 J_P = ∑||P_j - P_{ref,j}||²"""
        if not reference_path:
            return 0.0
        
        total_deviation = 0.0
        
        # 在控制点处计算偏差
        for i, control_point in enumerate(self.control_points):
            if i < len(reference_path) - 1:
                ref_point = reference_path[i + 1]
                deviation = np.linalg.norm(control_point[:2] - ref_point[:2])
                total_deviation += deviation ** 2
        
        return total_deviation
    
    def _compute_obstacle_cost(self, obstacle_cost_func: Callable) -> float:
        """计算障碍物代价"""
        total_cost = 0.0
        
        for segment in self.segments:
            T = segment.duration
            num_samples = 20  # 每段采样20个点
            dt = T / num_samples
            
            for i in range(num_samples):
                t = i * dt
                pos = segment.evaluate_position(t)
                vel = segment.evaluate_velocity(t)
                
                # 调用外部提供的障碍物代价函数
                cost = obstacle_cost_func(pos, vel)
                total_cost += cost * dt
        
        return total_cost
    
    def _compute_swept_volume_cost(self, swept_volume_cost_func: Callable) -> float:
        """计算扫掠体积代价"""
        try:
            return swept_volume_cost_func(self.segments)
        except Exception as e:
            print(f"扫掠体积代价计算异常: {e}")
            return 0.0
    
    def get_discretized_trajectory(self, dt: float = 0.1) -> Tuple[List[np.ndarray], 
                                                                 List[np.ndarray], 
                                                                 List[np.ndarray], 
                                                                 List[float]]:
        """获取离散化轨迹"""
        positions, velocities, accelerations, times = [], [], [], []
        
        current_time = 0.0
        
        for segment in self.segments:
            T = segment.duration
            num_samples = max(1, int(T / dt))
            segment_dt = T / num_samples
            
            for i in range(num_samples + 1):
                t = i * segment_dt
                if t > T:
                    t = T
                
                pos = segment.evaluate_position(t)
                vel = segment.evaluate_velocity(t)
                acc = segment.evaluate_acceleration(t)
                
                positions.append(pos)
                velocities.append(vel)
                accelerations.append(acc)
                times.append(current_time + t)
                
                if t >= T:
                    break
            
            current_time += T
        
        return positions, velocities, accelerations, times
    
    def get_total_time(self) -> float:
        """获取轨迹总时间"""
        return np.sum(self.time_allocation)
    
    def evaluate_at_time(self, global_time: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """在指定时间评估轨迹"""
        current_time = 0.0
        
        for segment in self.segments:
            if current_time <= global_time <= current_time + segment.duration:
                local_time = global_time - current_time
                pos = segment.evaluate_position(local_time)
                vel = segment.evaluate_velocity(local_time)
                acc = segment.evaluate_acceleration(local_time)
                return pos, vel, acc
            current_time += segment.duration
        
        # 如果超出范围，返回最后一个点
        if self.segments:
            last_segment = self.segments[-1]
            pos = last_segment.evaluate_position(last_segment.duration)
            vel = last_segment.evaluate_velocity(last_segment.duration)
            acc = last_segment.evaluate_acceleration(last_segment.duration)
            return pos, vel, acc
        
        return np.zeros(3), np.zeros(3), np.zeros(3)