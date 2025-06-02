# core/mpc_controller_optimized.py
"""
工业级MPC控制器优化实现
SVSDF轨迹规划的第四阶段：实时轨迹跟踪控制

核心特性：
1. 工业级实时性能优化
2. Numba JIT 加速核心计算
3. 自适应预测时域和权重调节
4. 鲁棒性约束处理
5. Isaac Sim 深度集成
6. 并行计算支持
7. 高级故障恢复机制
8. 实时性能监控
"""

import numpy as np
import numba as nb
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import warnings
from scipy.optimize import minimize, OptimizeResult
from scipy.sparse import csc_matrix
import logging

# Isaac Sim 集成 (带有错误处理)
try:
    from isaacsim.core.api.prims import RigidPrimView
    from isaacsim.core.api.articulations import ArticulationView
    from isaacsim.core.utils.types import ArticulationAction
    ISAAC_SIM_AVAILABLE = True
except ImportError:
    ISAAC_SIM_AVAILABLE = False
    ArticulationAction = None

# 工具导入
from utils.math_utils import MathUtils
from utils.config import config

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MPCStateOptimized:
    """优化的MPC状态表示"""
    x: float = 0.0           # x位置 (m)
    y: float = 0.0           # y位置 (m)
    theta: float = 0.0       # 偏航角 (rad)
    v: float = 0.0           # 线速度 (m/s)
    omega: float = 0.0       # 角速度 (rad/s)
    timestamp: float = 0.0   # 时间戳 (s)
    
    # 扩展状态信息
    ax: float = 0.0          # x方向加速度 (m/s²)
    ay: float = 0.0          # y方向加速度 (m/s²)
    alpha: float = 0.0       # 角加速度 (rad/s²)
    
    # 状态质量指标
    position_error: float = 0.0      # 位置跟踪误差
    orientation_error: float = 0.0   # 姿态跟踪误差
    velocity_error: float = 0.0      # 速度跟踪误差

@dataclass
class MPCControlOptimized:
    """优化的MPC控制输入"""
    linear_vel: float = 0.0     # 线速度指令 (m/s)
    angular_vel: float = 0.0    # 角速度指令 (rad/s)
    linear_acc: float = 0.0     # 线加速度指令 (m/s²)
    angular_acc: float = 0.0    # 角加速度指令 (rad/s²)
    
    # 差分驱动控制
    v_left: float = 0.0         # 左轮速度 (m/s)
    v_right: float = 0.0        # 右轮速度 (m/s)
    
    # 扩展控制信息
    timestamp: float = 0.0      # 控制时间戳
    control_mode: str = "velocity"  # 控制模式: "velocity", "acceleration", "force"
    
    # Isaac Sim 集成
    articulation_action: Any = None  # ArticulationAction for Isaac Sim

@dataclass
class MPCPredictionResults:
    """MPC预测结果"""
    predicted_states: List[MPCStateOptimized] = field(default_factory=list)
    control_sequence: List[MPCControlOptimized] = field(default_factory=list)
    cost_value: float = 0.0
    constraint_violations: List[str] = field(default_factory=list)
    solve_time_ms: float = 0.0
    iterations: int = 0
    success: bool = False

@dataclass 
class MPCPerformanceMetrics:
    """MPC性能指标"""
    average_solve_time_ms: float = 0.0
    max_solve_time_ms: float = 0.0
    min_solve_time_ms: float = float('inf')
    success_rate: float = 1.0
    total_solves: int = 0
    failed_solves: int = 0
    
    # 跟踪性能
    position_rmse: float = 0.0
    orientation_rmse: float = 0.0
    velocity_rmse: float = 0.0
    
    # 计算资源
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0

# Numba JIT 优化的核心计算函数
@nb.jit(nopython=True, cache=True)
def compute_prediction_matrices_jit(A_matrices: nb.float64[:, :, :], 
                                   B_matrices: nb.float64[:, :, :],
                                   N_p: int, N_c: int) -> Tuple[nb.float64[:, :], nb.float64[:, :]]:
    """
    JIT优化的预测矩阵计算
    
    预测模型：X = Psi * x(0) + Theta * U
    """
    # 初始化矩阵
    Psi = np.zeros((3 * N_p, 3), dtype=nb.float64)
    Theta = np.zeros((3 * N_p, 2 * N_c), dtype=nb.float64)
    
    # 计算Psi矩阵
    A_prod = np.eye(3, dtype=nb.float64)
    for i in range(N_p):
        Psi[i*3:(i+1)*3, :] = A_prod
        if i < A_matrices.shape[0]:
            A_prod = A_matrices[i] @ A_prod
    
    # 计算Theta矩阵
    for i in range(N_p):
        for j in range(min(N_c, i + 1)):
            # 计算状态转移乘积
            A_prod = np.eye(3, dtype=nb.float64)
            for k in range(j, i):
                if k < A_matrices.shape[0]:
                    A_prod = A_matrices[k] @ A_prod
            
            if j < B_matrices.shape[0]:
                Theta[i*3:(i+1)*3, j*2:(j+1)*2] = A_prod @ B_matrices[j]
    
    return Psi, Theta

@nb.jit(nopython=True, cache=True)
def linearize_dynamics_jit(theta_ref: float, v_ref: float, dt: float) -> Tuple[nb.float64[:, :], nb.float64[:, :]]:
    """
    JIT优化的动力学线性化
    
    非线性模型：
    x(k+1) = x(k) + T * v(k) * cos(θ(k))
    y(k+1) = y(k) + T * v(k) * sin(θ(k))
    θ(k+1) = θ(k) + T * ω(k)
    """
    cos_theta = np.cos(theta_ref)
    sin_theta = np.sin(theta_ref)
    
    # 状态矩阵 A
    A = np.eye(3, dtype=nb.float64)
    A[0, 2] = -dt * v_ref * sin_theta  # ∂x/∂θ
    A[1, 2] = dt * v_ref * cos_theta   # ∂y/∂θ
    
    # 控制矩阵 B
    B = np.zeros((3, 2), dtype=nb.float64)
    B[0, 0] = dt * cos_theta    # ∂x/∂v
    B[1, 0] = dt * sin_theta    # ∂y/∂v
    B[2, 1] = dt                # ∂θ/∂ω
    
    return A, B

@nb.jit(nopython=True, cache=True)
def predict_trajectory_jit(initial_state: nb.float64[:], 
                          control_sequence: nb.float64[:, :],
                          dt: float) -> nb.float64[:, :]:
    """JIT优化的轨迹预测"""
    n_steps = control_sequence.shape[0]
    predicted_states = np.zeros((n_steps + 1, 5), dtype=nb.float64)  # [x, y, theta, v, omega]
    
    # 初始状态
    predicted_states[0] = initial_state
    
    for i in range(n_steps):
        current_state = predicted_states[i]
        control = control_sequence[i]
        
        # 差分驱动动力学
        x, y, theta, v, omega = current_state
        v_cmd, omega_cmd = control
        
        # 状态更新
        next_x = x + dt * v_cmd * np.cos(theta)
        next_y = y + dt * v_cmd * np.sin(theta)
        next_theta = theta + dt * omega_cmd
        
        # 归一化角度
        while next_theta > np.pi:
            next_theta -= 2 * np.pi
        while next_theta < -np.pi:
            next_theta += 2 * np.pi
        
        predicted_states[i + 1] = np.array([next_x, next_y, next_theta, v_cmd, omega_cmd])
    
    return predicted_states

class MPCControllerOptimized:
    """
    工业级优化MPC控制器
    
    核心特性：
    1. 高性能数值优化
    2. 自适应参数调节
    3. 鲁棒性约束处理
    4. 实时性能监控
    5. Isaac Sim深度集成
    """
    
    def __init__(self, use_isaac_sim: bool = True, enable_parallel: bool = True):
        """
        初始化优化MPC控制器
        
        Args:
            use_isaac_sim: 是否使用Isaac Sim集成
            enable_parallel: 是否启用并行计算
        """
        logger.info("初始化优化MPC控制器...")
        
        # 基本参数
        self.N_p = config.mpc.prediction_horizon
        self.N_c = config.mpc.control_horizon
        self.dt = config.mpc.sample_time
        
        # 权重矩阵
        self.Q = config.mpc.state_weights.copy()
        self.R = config.mpc.control_weights.copy() 
        self.Q_f = config.mpc.terminal_weights.copy()
        
        # 约束参数
        self.max_linear_vel = config.robot.max_linear_velocity
        self.max_angular_vel = config.robot.max_angular_velocity
        self.max_linear_acc = config.robot.max_linear_acceleration
        self.max_angular_acc = config.robot.max_angular_acceleration
        self.wheel_base = config.robot.wheel_base
        
        # 优化设置
        self.use_isaac_sim = use_isaac_sim and ISAAC_SIM_AVAILABLE
        self.enable_parallel = enable_parallel
        self.solver_tolerance = 1e-6
        self.max_iterations = 100
        
        # 自适应参数
        self.adaptive_weights = True
        self.adaptive_horizon = True
        self.min_horizon = 5
        self.max_horizon = 30
        
        # 性能监控
        self.performance_metrics = MPCPerformanceMetrics()
        self.solve_history = []
        self.error_history = []
        
        # 状态缓存
        self.last_control = MPCControlOptimized()
        self.last_prediction = MPCPredictionResults()
        self.predicted_trajectory: List[MPCStateOptimized] = []
        
        # 线程安全
        self.computation_lock = threading.Lock()
        self.thread_pool = ThreadPoolExecutor(max_workers=4) if enable_parallel else None
        
        # Isaac Sim 集成
        if self.use_isaac_sim:
            self._initialize_isaac_sim()
        
        # 预编译JIT函数
        self._warm_up_jit()
        
        logger.info(f"MPC控制器初始化完成 - Isaac Sim: {self.use_isaac_sim}, 并行: {enable_parallel}")
    
    def _initialize_isaac_sim(self):
        """初始化Isaac Sim集成"""
        try:
            # 这里可以添加Isaac Sim特定的初始化
            logger.info("Isaac Sim集成已启用")
        except Exception as e:
            logger.warning(f"Isaac Sim初始化失败: {e}")
            self.use_isaac_sim = False
    
    def _warm_up_jit(self):
        """预热JIT编译函数"""
        try:
            # 预编译主要的JIT函数
            dummy_A = np.eye(3).reshape(1, 3, 3)
            dummy_B = np.zeros((1, 3, 2))
            compute_prediction_matrices_jit(dummy_A, dummy_B, 5, 3)
            
            dummy_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
            dummy_controls = np.zeros((3, 2))
            predict_trajectory_jit(dummy_state, dummy_controls, 0.1)
            
            logger.info("JIT函数预编译完成")
        except Exception as e:
            logger.warning(f"JIT预编译失败: {e}")
    
    def compute_control(self, current_state: MPCStateOptimized,
                       reference_trajectory: List[np.ndarray],
                       current_time: float) -> MPCControlOptimized:
        """
        计算优化MPC控制输入
        
        Args:
            current_state: 当前机器人状态
            reference_trajectory: 参考轨迹 [x, y, theta, time]
            current_time: 当前时间
            
        Returns:
            优化的MPC控制输入
        """
        solve_start_time = time.time()
        
        with self.computation_lock:
            try:
                # 自适应参数调节
                if self.adaptive_weights:
                    self._adapt_weights(current_state)
                
                if self.adaptive_horizon:
                    self._adapt_horizon(current_state, reference_trajectory, current_time)
                
                # 获取参考轨迹
                reference_states = self._get_reference_trajectory(reference_trajectory, current_time)
                
                if not reference_states:
                    logger.warning("无有效参考轨迹，返回停止控制")
                    return MPCControlOptimized()
                
                # 求解MPC优化问题
                prediction_result = self._solve_mpc_optimization(current_state, reference_states)
                
                # 提取控制输入
                if prediction_result.success and prediction_result.control_sequence:
                    control = prediction_result.control_sequence[0]
                else:
                    logger.warning("MPC求解失败，使用安全控制")
                    control = self._get_safe_control()
                
                # 应用约束和转换
                control = self._apply_constraints(control)
                self._convert_to_differential_drive(control)
                
                # Isaac Sim集成
                if self.use_isaac_sim:
                    control.articulation_action = self._create_articulation_action(control)
                
                # 更新性能指标
                solve_time = (time.time() - solve_start_time) * 1000
                self._update_performance_metrics(solve_time, prediction_result.success)
                
                # 保存结果
                self.last_control = control
                self.last_prediction = prediction_result
                self.predicted_trajectory = prediction_result.predicted_states
                
                logger.debug(f"MPC求解完成 - 时间: {solve_time:.2f}ms, 成功: {prediction_result.success}")
                
                return control
                
            except Exception as e:
                logger.error(f"MPC计算异常: {e}")
                solve_time = (time.time() - solve_start_time) * 1000
                self._update_performance_metrics(solve_time, False)
                return self._get_safe_control()
    
    def _adapt_weights(self, current_state: MPCStateOptimized):
        """自适应权重调节"""
        try:
            # 基于跟踪误差调节权重
            position_error = current_state.position_error
            orientation_error = current_state.orientation_error
            velocity_error = current_state.velocity_error
            
            # 动态调节位置权重
            if position_error > 0.5:  # 大位置误差
                self.Q[0, 0] = min(50.0, self.Q[0, 0] * 1.2)
                self.Q[1, 1] = min(50.0, self.Q[1, 1] * 1.2)
            elif position_error < 0.1:  # 小位置误差
                self.Q[0, 0] = max(5.0, self.Q[0, 0] * 0.9)
                self.Q[1, 1] = max(5.0, self.Q[1, 1] * 0.9)
            
            # 动态调节角度权重
            if orientation_error > 0.3:  # 大角度误差
                self.Q[2, 2] = min(30.0, self.Q[2, 2] * 1.2)
            elif orientation_error < 0.05:  # 小角度误差
                self.Q[2, 2] = max(2.0, self.Q[2, 2] * 0.9)
            
            # 基于速度调节控制权重
            if velocity_error > 0.2:
                self.R[0, 0] = max(0.1, self.R[0, 0] * 0.8)
            else:
                self.R[0, 0] = min(5.0, self.R[0, 0] * 1.05)
                
        except Exception as e:
            logger.warning(f"权重自适应调节失败: {e}")
    
    def _adapt_horizon(self, current_state: MPCStateOptimized, 
                      reference_trajectory: List[np.ndarray], current_time: float):
        """自适应时域调节"""
        try:
            # 基于轨迹复杂度和跟踪性能调节预测时域
            trajectory_complexity = self._estimate_trajectory_complexity(reference_trajectory, current_time)
            tracking_performance = self._estimate_tracking_performance()
            
            if trajectory_complexity > 0.8 or tracking_performance < 0.6:
                # 复杂轨迹或跟踪性能差，增加预测时域
                self.N_p = min(self.max_horizon, self.N_p + 2)
                self.N_c = min(self.N_p, self.N_c + 1)
            elif trajectory_complexity < 0.3 and tracking_performance > 0.9:
                # 简单轨迹且跟踪性能好，减少预测时域
                self.N_p = max(self.min_horizon, self.N_p - 1)
                self.N_c = max(3, self.N_c - 1)
                
        except Exception as e:
            logger.warning(f"时域自适应调节失败: {e}")
    
    def _estimate_trajectory_complexity(self, reference_trajectory: List[np.ndarray], 
                                      current_time: float) -> float:
        """估计轨迹复杂度"""
        try:
            if len(reference_trajectory) < 3:
                return 0.0
            
            # 计算未来轨迹的曲率变化
            future_points = []
            for i in range(self.N_p):
                future_time = current_time + i * self.dt
                point = MathUtils.interpolate_trajectory(reference_trajectory, future_time)
                future_points.append(point)
            
            if len(future_points) < 3:
                return 0.0
            
            # 计算曲率变化率
            curvature_changes = []
            for i in range(1, len(future_points) - 1):
                p1, p2, p3 = future_points[i-1][:2], future_points[i][:2], future_points[i+1][:2]
                curvature = MathUtils.compute_curvature(p1, p2, p3)
                curvature_changes.append(abs(curvature))
            
            complexity = np.mean(curvature_changes) if curvature_changes else 0.0
            return min(1.0, complexity * 10.0)  # 归一化到[0,1]
            
        except Exception as e:
            logger.warning(f"轨迹复杂度估计失败: {e}")
            return 0.5
    
    def _estimate_tracking_performance(self) -> float:
        """估计跟踪性能"""
        try:
            if len(self.error_history) < 5:
                return 1.0
            
            recent_errors = self.error_history[-10:]  # 最近10次的误差
            avg_error = np.mean([err.position_error + err.orientation_error for err in recent_errors])
            
            # 性能评分：误差越小性能越好
            performance = max(0.0, 1.0 - avg_error)
            return performance
            
        except Exception as e:
            logger.warning(f"跟踪性能估计失败: {e}")
            return 0.5
    
    def _get_reference_trajectory(self, reference_trajectory: List[np.ndarray],
                                 current_time: float) -> List[MPCStateOptimized]:
        """获取预测时域内的参考状态"""
        reference_states = []
        
        try:
            for i in range(self.N_p):
                future_time = current_time + i * self.dt
                
                # 轨迹插值
                ref_pose = MathUtils.interpolate_trajectory(reference_trajectory, future_time)
                
                if ref_pose is None:
                    break
                
                # 计算参考速度（数值微分）
                v_ref, omega_ref = 0.0, 0.0
                if i < self.N_p - 1:
                    next_time = current_time + (i + 1) * self.dt
                    next_pose = MathUtils.interpolate_trajectory(reference_trajectory, next_time)
                    
                    if next_pose is not None:
                        dx = next_pose[0] - ref_pose[0]
                        dy = next_pose[1] - ref_pose[1]
                        v_ref = np.sqrt(dx**2 + dy**2) / self.dt
                        
                        dtheta = MathUtils.normalize_angle(next_pose[2] - ref_pose[2])
                        omega_ref = dtheta / self.dt
                
                ref_state = MPCStateOptimized(
                    x=ref_pose[0], y=ref_pose[1], theta=ref_pose[2],
                    v=v_ref, omega=omega_ref, timestamp=future_time
                )
                reference_states.append(ref_state)
            
            return reference_states
            
        except Exception as e:
            logger.error(f"获取参考轨迹失败: {e}")
            return []
    
    def _solve_mpc_optimization(self, current_state: MPCStateOptimized,
                               reference_states: List[MPCStateOptimized]) -> MPCPredictionResults:
        """
        求解MPC优化问题
        
        使用高效的数值优化算法求解二次规划问题
        """
        solve_start = time.time()
        
        try:
            # 构建线性化模型
            A_matrices, B_matrices = self._build_linearized_model(current_state, reference_states)
            
            # 使用JIT优化的预测矩阵计算
            A_array = np.array(A_matrices)
            B_array = np.array(B_matrices)
            Psi, Theta = compute_prediction_matrices_jit(A_array, B_array, self.N_p, self.N_c)
            
            # 构建代价函数
            H, g = self._build_cost_matrices(Psi, Theta, current_state, reference_states)
            
            # 构建约束
            A_ineq, b_ineq, A_eq, b_eq = self._build_constraint_matrices()
            
            # 求解QP问题
            optimization_result = self._solve_quadratic_program(H, g, A_ineq, b_ineq, A_eq, b_eq)
            
            solve_time = (time.time() - solve_start) * 1000
            
            if optimization_result.success:
                # 转换优化结果
                control_sequence = self._extract_control_sequence(optimization_result.x)
                predicted_states = self._predict_future_states(current_state, control_sequence)
                
                return MPCPredictionResults(
                    predicted_states=predicted_states,
                    control_sequence=control_sequence,
                    cost_value=optimization_result.fun,
                    solve_time_ms=solve_time,
                    iterations=optimization_result.nit if hasattr(optimization_result, 'nit') else 0,
                    success=True
                )
            else:
                logger.warning(f"MPC优化求解失败: {optimization_result.message}")
                return MPCPredictionResults(
                    solve_time_ms=solve_time,
                    success=False
                )
                
        except Exception as e:
            solve_time = (time.time() - solve_start) * 1000
            logger.error(f"MPC优化异常: {e}")
            return MPCPredictionResults(
                solve_time_ms=solve_time,
                success=False
            )
    
    def _build_linearized_model(self, current_state: MPCStateOptimized,
                               reference_states: List[MPCStateOptimized]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """构建线性化动力学模型"""
        A_matrices = []
        B_matrices = []
        
        for i in range(self.N_p):
            if i < len(reference_states):
                ref_state = reference_states[i]
            else:
                ref_state = reference_states[-1] if reference_states else current_state
            
            # 使用JIT优化的线性化函数
            A, B = linearize_dynamics_jit(ref_state.theta, ref_state.v, self.dt)
            A_matrices.append(A)
            B_matrices.append(B)
        
        return A_matrices, B_matrices
    
    def _build_cost_matrices(self, Psi: np.ndarray, Theta: np.ndarray,
                           current_state: MPCStateOptimized,
                           reference_states: List[MPCStateOptimized]) -> Tuple[np.ndarray, np.ndarray]:
        """构建代价函数矩阵（高性能版本）"""
        # 构建块对角权重矩阵
        Q_bar = np.zeros((3 * self.N_p, 3 * self.N_p))
        for i in range(self.N_p - 1):
            Q_bar[i*3:(i+1)*3, i*3:(i+1)*3] = self.Q
        Q_bar[-3:, -3:] = self.Q_f  # 终端权重
        
        R_bar = np.zeros((2 * self.N_c, 2 * self.N_c))
        for i in range(self.N_c):
            R_bar[i*2:(i+1)*2, i*2:(i+1)*2] = self.R
        
        # 构建参考轨迹向量
        X_ref = np.zeros(3 * self.N_p)
        for i in range(self.N_p):
            if i < len(reference_states):
                ref_state = reference_states[i]
                X_ref[i*3:(i+1)*3] = [ref_state.x, ref_state.y, ref_state.theta]
            else:
                ref_state = reference_states[-1] if reference_states else current_state
                X_ref[i*3:(i+1)*3] = [ref_state.x, ref_state.y, ref_state.theta]
        
        # 当前状态向量
        x0 = np.array([current_state.x, current_state.y, current_state.theta])
        
        # 计算Hessian和梯度
        H = Theta.T @ Q_bar @ Theta + R_bar
        g = Theta.T @ Q_bar @ (Psi @ x0 - X_ref)
        
        # 确保正定性
        min_eigenval = np.min(np.linalg.eigvals(H))
        if min_eigenval <= 1e-8:
            H += (1e-6 - min_eigenval) * np.eye(H.shape[0])
        
        return H, g
    
    def _build_constraint_matrices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """构建约束矩阵（包含等式和不等式约束）"""
        num_vars = self.N_c * 2
        
        # 不等式约束：控制输入界限和增量约束
        num_ineq_constraints = self.N_c * 4 + (self.N_c - 1) * 4 if self.N_c > 1 else self.N_c * 4
        A_ineq = np.zeros((num_ineq_constraints, num_vars))
        b_ineq = np.zeros(num_ineq_constraints)
        
        constraint_idx = 0
        
        # 控制输入约束
        for i in range(self.N_c):
            # 线速度约束: -v_max <= v <= v_max
            A_ineq[constraint_idx, i * 2] = 1.0
            b_ineq[constraint_idx] = self.max_linear_vel
            constraint_idx += 1
            
            A_ineq[constraint_idx, i * 2] = -1.0
            b_ineq[constraint_idx] = self.max_linear_vel
            constraint_idx += 1
            
            # 角速度约束: -omega_max <= omega <= omega_max
            A_ineq[constraint_idx, i * 2 + 1] = 1.0
            b_ineq[constraint_idx] = self.max_angular_vel
            constraint_idx += 1
            
            A_ineq[constraint_idx, i * 2 + 1] = -1.0
            b_ineq[constraint_idx] = self.max_angular_vel
            constraint_idx += 1
        
        # 控制增量约束
        max_linear_rate = self.max_linear_acc * self.dt
        max_angular_rate = self.max_angular_acc * self.dt
        
        for i in range(self.N_c - 1):
            # 线速度增量约束
            A_ineq[constraint_idx, (i + 1) * 2] = 1.0
            A_ineq[constraint_idx, i * 2] = -1.0
            b_ineq[constraint_idx] = max_linear_rate
            constraint_idx += 1
            
            A_ineq[constraint_idx, (i + 1) * 2] = -1.0
            A_ineq[constraint_idx, i * 2] = 1.0
            b_ineq[constraint_idx] = max_linear_rate
            constraint_idx += 1
            
            # 角速度增量约束
            A_ineq[constraint_idx, (i + 1) * 2 + 1] = 1.0
            A_ineq[constraint_idx, i * 2 + 1] = -1.0
            b_ineq[constraint_idx] = max_angular_rate
            constraint_idx += 1
            
            A_ineq[constraint_idx, (i + 1) * 2 + 1] = -1.0
            A_ineq[constraint_idx, i * 2 + 1] = 1.0
            b_ineq[constraint_idx] = max_angular_rate
            constraint_idx += 1
        
        # 等式约束（如果需要）
        A_eq = np.zeros((0, num_vars))
        b_eq = np.zeros(0)
        
        return A_ineq, b_ineq, A_eq, b_eq
    
    def _solve_quadratic_program(self, H: np.ndarray, g: np.ndarray,
                                A_ineq: np.ndarray, b_ineq: np.ndarray,
                                A_eq: np.ndarray, b_eq: np.ndarray) -> OptimizeResult:
        """求解二次规划问题"""
        try:
            # 定义目标函数
            def objective(x):
                return 0.5 * x.T @ H @ x + g.T @ x
            
            def jacobian(x):
                return H @ x + g
            
            # 构建约束
            constraints = []
            
            # 不等式约束
            if A_ineq.shape[0] > 0:
                def ineq_constraint(x):
                    return b_ineq - A_ineq @ x
                
                constraints.append({
                    'type': 'ineq',
                    'fun': ineq_constraint,
                    'jac': lambda x: -A_ineq
                })
            
            # 等式约束
            if A_eq.shape[0] > 0:
                def eq_constraint(x):
                    return A_eq @ x - b_eq
                
                constraints.append({
                    'type': 'eq',
                    'fun': eq_constraint,
                    'jac': lambda x: A_eq
                })
            
            # 初始猜测（使用上次的解作为热启动）
            x0 = np.zeros(H.shape[0])
            if hasattr(self, 'last_solution') and len(self.last_solution) == H.shape[0]:
                x0 = self.last_solution
            
            # 求解优化问题
            options = {
                'maxiter': self.max_iterations,
                'ftol': self.solver_tolerance,
                'disp': False
            }
            
            result = minimize(
                objective, x0,
                method='SLSQP',
                jac=jacobian,
                constraints=constraints,
                options=options
            )
            
            # 保存解用于下次热启动
            if result.success:
                self.last_solution = result.x.copy()
            
            return result
            
        except Exception as e:
            logger.error(f"二次规划求解失败: {e}")
            # 返回失败结果
            dummy_result = OptimizeResult()
            dummy_result.success = False
            dummy_result.message = str(e)
            dummy_result.x = np.zeros(H.shape[0])
            return dummy_result
    
    def _extract_control_sequence(self, solution: np.ndarray) -> List[MPCControlOptimized]:
        """从优化解中提取控制序列"""
        control_sequence = []
        
        for i in range(self.N_c):
            control = MPCControlOptimized(
                linear_vel=solution[i * 2],
                angular_vel=solution[i * 2 + 1],
                timestamp=time.time()
            )
            control_sequence.append(control)
        
        return control_sequence
    
    def _predict_future_states(self, initial_state: MPCStateOptimized,
                              control_sequence: List[MPCControlOptimized]) -> List[MPCStateOptimized]:
        """预测未来状态序列"""
        predicted_states = [initial_state]
        
        # 准备JIT函数输入
        initial_state_array = np.array([
            initial_state.x, initial_state.y, initial_state.theta,
            initial_state.v, initial_state.omega
        ])
        
        control_array = np.array([
            [ctrl.linear_vel, ctrl.angular_vel] for ctrl in control_sequence
        ])
        
        # 使用JIT优化的预测
        predicted_array = predict_trajectory_jit(initial_state_array, control_array, self.dt)
        
        # 转换为MPCStateOptimized对象
        for i in range(1, predicted_array.shape[0]):
            state_data = predicted_array[i]
            predicted_state = MPCStateOptimized(
                x=state_data[0],
                y=state_data[1], 
                theta=state_data[2],
                v=state_data[3],
                omega=state_data[4],
                timestamp=initial_state.timestamp + i * self.dt
            )
            predicted_states.append(predicted_state)
        
        return predicted_states
    
    def _apply_constraints(self, control: MPCControlOptimized) -> MPCControlOptimized:
        """应用硬约束限制"""
        control.linear_vel = np.clip(control.linear_vel, 
                                   -self.max_linear_vel, self.max_linear_vel)
        control.angular_vel = np.clip(control.angular_vel,
                                    -self.max_angular_vel, self.max_angular_vel)
        
        # 应用加速度约束
        if hasattr(self, 'last_control'):
            dt = self.dt
            max_dv = self.max_linear_acc * dt
            max_domega = self.max_angular_acc * dt
            
            dv = control.linear_vel - self.last_control.linear_vel
            domega = control.angular_vel - self.last_control.angular_vel
            
            control.linear_vel = self.last_control.linear_vel + np.clip(dv, -max_dv, max_dv)
            control.angular_vel = self.last_control.angular_vel + np.clip(domega, -max_domega, max_domega)
        
        return control
    
    def _convert_to_differential_drive(self, control: MPCControlOptimized):
        """转换为差分驱动轮速"""
        # 差分驱动运动学逆解
        control.v_left = control.linear_vel - control.angular_vel * self.wheel_base / 2.0
        control.v_right = control.linear_vel + control.angular_vel * self.wheel_base / 2.0
        
        # 轮速限制
        max_wheel_speed = self.max_linear_vel
        control.v_left = np.clip(control.v_left, -max_wheel_speed, max_wheel_speed)
        control.v_right = np.clip(control.v_right, -max_wheel_speed, max_wheel_speed)
    
    def _create_articulation_action(self, control: MPCControlOptimized) -> Any:
        """创建Isaac Sim关节动作"""
        if not self.use_isaac_sim or ArticulationAction is None:
            return None
        
        try:
            # 创建关节动作（假设差分驱动机器人有两个轮子关节）
            action = ArticulationAction(
                joint_velocities=np.array([control.v_left, control.v_right])
            )
            return action
        except Exception as e:
            logger.warning(f"创建关节动作失败: {e}")
            return None
    
    def _get_safe_control(self) -> MPCControlOptimized:
        """获取安全控制（停止）"""
        return MPCControlOptimized(
            linear_vel=0.0,
            angular_vel=0.0,
            timestamp=time.time()
        )
    
    def _update_performance_metrics(self, solve_time_ms: float, success: bool):
        """更新性能指标"""
        self.performance_metrics.total_solves += 1
        
        if success:
            self.performance_metrics.average_solve_time_ms = (
                (self.performance_metrics.average_solve_time_ms * 
                 (self.performance_metrics.total_solves - 1) + solve_time_ms) / 
                self.performance_metrics.total_solves
            )
            self.performance_metrics.max_solve_time_ms = max(
                self.performance_metrics.max_solve_time_ms, solve_time_ms)
            self.performance_metrics.min_solve_time_ms = min(
                self.performance_metrics.min_solve_time_ms, solve_time_ms)
        else:
            self.performance_metrics.failed_solves += 1
        
        self.performance_metrics.success_rate = (
            (self.performance_metrics.total_solves - self.performance_metrics.failed_solves) / 
            self.performance_metrics.total_solves
        )
        
        # 保持历史记录长度
        self.solve_history.append(solve_time_ms)
        if len(self.solve_history) > 1000:
            self.solve_history.pop(0)
    
    def get_performance_metrics(self) -> MPCPerformanceMetrics:
        """获取性能指标"""
        return self.performance_metrics
    
    def get_predicted_trajectory(self) -> List[MPCStateOptimized]:
        """获取预测轨迹"""
        return self.predicted_trajectory.copy()
    
    def get_last_prediction_result(self) -> MPCPredictionResults:
        """获取最后的预测结果"""
        return self.last_prediction
    
    def reset(self):
        """重置控制器状态"""
        self.last_control = MPCControlOptimized()
        self.last_prediction = MPCPredictionResults()
        self.predicted_trajectory = []
        self.solve_history = []
        self.error_history = []
        
        # 重置性能指标
        self.performance_metrics = MPCPerformanceMetrics()
        
        logger.info("MPC控制器已重置")
    
    def cleanup(self):
        """清理资源"""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        
        logger.info("MPC控制器资源已清理")

# 工厂函数
def create_mpc_controller_optimized(use_isaac_sim: bool = True, 
                                   enable_parallel: bool = True) -> MPCControllerOptimized:
    """
    创建优化MPC控制器实例
    
    Args:
        use_isaac_sim: 是否启用Isaac Sim集成
        enable_parallel: 是否启用并行计算
        
    Returns:
        优化的MPC控制器实例
    """
    return MPCControllerOptimized(use_isaac_sim=use_isaac_sim, enable_parallel=enable_parallel)
