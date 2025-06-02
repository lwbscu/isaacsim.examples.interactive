# core/mpc_controller.py
"""
模型预测控制器(MPC)实现
SVSDF轨迹规划的第四阶段：实时跟踪控制
"""
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from scipy.optimize import minimize
from utils.math_utils import MathUtils, OptimizationUtils
from utils.config import config

@dataclass
class MPCState:
    """MPC状态"""
    x: float = 0.0      # x位置
    y: float = 0.0      # y位置  
    theta: float = 0.0  # 偏航角
    v: float = 0.0      # 线速度
    omega: float = 0.0  # 角速度

@dataclass
class MPCControl:
    """MPC控制输入"""
    linear_vel: float = 0.0   # 线速度指令
    angular_vel: float = 0.0  # 角速度指令
    v_left: float = 0.0       # 左轮速度
    v_right: float = 0.0      # 右轮速度

class MPCController:
    """
    模型预测控制器
    实现论文中第四阶段的实时轨迹跟踪控制
    
    系统模型（差分驱动）：
    x(k+1) = x(k) + T * v(k) * cos(θ(k))
    y(k+1) = y(k) + T * v(k) * sin(θ(k))
    θ(k+1) = θ(k) + T * ω(k)
    """
    
    def __init__(self):
        # 从配置加载参数
        self.N_p = config.mpc.prediction_horizon      # 预测时域
        self.N_c = config.mpc.control_horizon         # 控制时域
        self.dt = config.mpc.sample_time              # 采样时间
        
        # 权重矩阵
        self.Q = config.mpc.state_weights             # 状态权重
        self.R = config.mpc.control_weights           # 控制权重
        self.Q_f = config.mpc.terminal_weights        # 终端权重
        
        # 约束
        self.max_linear_vel = config.robot.max_linear_velocity
        self.max_angular_vel = config.robot.max_angular_velocity
        self.max_linear_acc = config.robot.max_linear_acceleration
        self.max_angular_acc = config.robot.max_angular_acceleration
        self.wheel_base = config.robot.wheel_base
        
        # 上一时刻的控制输入（用于控制增量约束）
        self.last_control = MPCControl()
        
        # 预测轨迹（用于可视化）
        self.predicted_trajectory: List[MPCState] = []
        
        # 性能监控
        self.computation_time = 0.0
        self.solve_success = True
    
    def compute_control(self, current_state: MPCState,
                       reference_trajectory: List[np.ndarray],
                       current_time: float) -> MPCControl:
        """
        计算MPC控制输入
        
        Args:
            current_state: 当前机器人状态
            reference_trajectory: 参考轨迹 [x, y, theta, time]
            current_time: 当前时间
            
        Returns:
            MPC控制输入
        """
        import time
        start_time = time.time()
        
        try:
            # 获取参考轨迹段
            reference_states = self._get_reference_trajectory(
                reference_trajectory, current_time)
            
            if not reference_states:
                # 如果没有参考轨迹，停止
                control = MPCControl()
                self._update_wheel_speeds(control)
                return control
            
            # 构建并求解QP问题
            control_sequence = self._solve_mpc_qp(current_state, reference_states)
            
            # 提取第一个控制输入（滚动时域原理）
            if control_sequence and len(control_sequence) > 0:
                control = control_sequence[0]
            else:
                control = MPCControl()
            
            # 约束检查
            control = self._apply_constraints(control)
            
            # 转换为轮速
            self._update_wheel_speeds(control)
            
            # 预测未来状态（用于可视化）
            self.predicted_trajectory = self._predict_trajectory(
                current_state, control_sequence)
            
            # 更新历史
            self.last_control = control
            self.solve_success = True
            
        except Exception as e:
            print(f"MPC求解异常: {e}")
            # 返回安全控制（停止）
            control = MPCControl()
            self._update_wheel_speeds(control)
            self.solve_success = False
        
        # 记录计算时间
        self.computation_time = (time.time() - start_time) * 1000  # ms
        
        return control
    
    def _get_reference_trajectory(self, reference_trajectory: List[np.ndarray],
                                current_time: float) -> List[MPCState]:
        """从参考轨迹中获取预测时域内的参考状态"""
        reference_states = []
        
        for i in range(self.N_p):
            future_time = current_time + i * self.dt
            
            # 在参考轨迹中插值
            ref_pose = MathUtils.interpolate_trajectory(reference_trajectory, future_time)
            
            # 计算参考速度（数值微分）
            if i < self.N_p - 1:
                next_time = current_time + (i + 1) * self.dt
                next_pose = MathUtils.interpolate_trajectory(reference_trajectory, next_time)
                
                # 线速度
                dx = next_pose[0] - ref_pose[0]
                dy = next_pose[1] - ref_pose[1]
                v_ref = np.sqrt(dx**2 + dy**2) / self.dt
                
                # 角速度
                dtheta = MathUtils.normalize_angle(next_pose[2] - ref_pose[2])
                omega_ref = dtheta / self.dt
            else:
                v_ref = 0.0
                omega_ref = 0.0
            
            ref_state = MPCState(
                x=ref_pose[0], y=ref_pose[1], theta=ref_pose[2],
                v=v_ref, omega=omega_ref
            )
            reference_states.append(ref_state)
        
        return reference_states
    
    def _solve_mpc_qp(self, current_state: MPCState,
                     reference_states: List[MPCState]) -> List[MPCControl]:
        """
        求解MPC二次规划问题
        
        问题形式：
        min 0.5 * U^T * H * U + g^T * U
        s.t. A_ineq * U <= b_ineq
        """
        # 构建优化变量：U = [u_0, u_1, ..., u_{N_c-1}]
        # 每个u_i = [v_i, omega_i]
        num_vars = self.N_c * 2
        
        # 线性化系统模型
        A_matrices, B_matrices = self._linearize_system_model(
            current_state, reference_states)
        
        # 构建预测矩阵
        Psi, Theta = self._build_prediction_matrices(A_matrices, B_matrices)
        
        # 构建代价函数矩阵
        H, g = self._build_cost_matrices(Psi, Theta, current_state, reference_states)
        
        # 构建约束矩阵
        A_ineq, b_ineq = self._build_constraint_matrices()
        
        # 求解QP
        try:
            U_opt = OptimizationUtils.quadratic_programming_solve(H, g, A_ineq, b_ineq)
            
            # 转换为控制序列
            control_sequence = []
            for i in range(self.N_c):
                control = MPCControl(
                    linear_vel=U_opt[i * 2],
                    angular_vel=U_opt[i * 2 + 1]
                )
                control_sequence.append(control)
            
            return control_sequence
            
        except Exception as e:
            print(f"QP求解失败: {e}")
            return []
    
    def _linearize_system_model(self, current_state: MPCState,
                               reference_states: List[MPCState]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        线性化系统模型
        
        非线性模型：
        x(k+1) = x(k) + T * v(k) * cos(θ(k))
        y(k+1) = y(k) + T * v(k) * sin(θ(k))
        θ(k+1) = θ(k) + T * ω(k)
        
        线性化：x(k+1) = A*x(k) + B*u(k) + C
        """
        A_matrices = []
        B_matrices = []
        
        for i in range(self.N_p):
            if i < len(reference_states):
                ref_state = reference_states[i]
            else:
                ref_state = reference_states[-1]
            
            theta_ref = ref_state.theta
            v_ref = ref_state.v
            
            # 状态矩阵 A
            A = np.eye(3)
            A[0, 2] = -self.dt * v_ref * np.sin(theta_ref)  # ∂x/∂θ
            A[1, 2] = self.dt * v_ref * np.cos(theta_ref)   # ∂y/∂θ
            
            # 控制矩阵 B
            B = np.zeros((3, 2))
            B[0, 0] = self.dt * np.cos(theta_ref)  # ∂x/∂v
            B[1, 0] = self.dt * np.sin(theta_ref)  # ∂y/∂v
            B[2, 1] = self.dt                      # ∂θ/∂ω
            
            A_matrices.append(A)
            B_matrices.append(B)
        
        return A_matrices, B_matrices
    
    def _build_prediction_matrices(self, A_matrices: List[np.ndarray],
                                 B_matrices: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        构建预测矩阵
        
        预测模型：X = Psi * x(0) + Theta * U
        """
        # Psi矩阵 (3*N_p x 3)
        Psi = np.zeros((3 * self.N_p, 3))
        
        # Theta矩阵 (3*N_p x 2*N_c)
        Theta = np.zeros((3 * self.N_p, 2 * self.N_c))
        
        # 填充Psi矩阵
        A_prod = np.eye(3)
        for i in range(self.N_p):
            Psi[i*3:(i+1)*3, :] = A_prod
            if i < len(A_matrices):
                A_prod = A_matrices[i] @ A_prod
        
        # 填充Theta矩阵
        for i in range(self.N_p):
            for j in range(min(self.N_c, i + 1)):
                # 计算从j到i的状态转移矩阵乘积
                A_prod = np.eye(3)
                for k in range(j, i):
                    if k < len(A_matrices):
                        A_prod = A_matrices[k] @ A_prod
                
                if j < len(B_matrices):
                    Theta[i*3:(i+1)*3, j*2:(j+1)*2] = A_prod @ B_matrices[j]
        
        return Psi, Theta
    
    def _build_cost_matrices(self, Psi: np.ndarray, Theta: np.ndarray,
                           current_state: MPCState,
                           reference_states: List[MPCState]) -> Tuple[np.ndarray, np.ndarray]:
        """
        构建代价函数矩阵
        
        代价函数：J = (X - X_ref)^T * Q_bar * (X - X_ref) + U^T * R_bar * U
        转化为：J = 0.5 * U^T * H * U + g^T * U + const
        """
        # 构建权重矩阵
        Q_bar = np.zeros((3 * self.N_p, 3 * self.N_p))
        for i in range(self.N_p - 1):
            Q_bar[i*3:(i+1)*3, i*3:(i+1)*3] = self.Q
        # 终端权重
        Q_bar[-3:, -3:] = self.Q_f
        
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
                # 使用最后一个参考状态
                ref_state = reference_states[-1]
                X_ref[i*3:(i+1)*3] = [ref_state.x, ref_state.y, ref_state.theta]
        
        # 当前状态向量
        x0 = np.array([current_state.x, current_state.y, current_state.theta])
        
        # 计算H和g矩阵
        H = Theta.T @ Q_bar @ Theta + R_bar
        g = Theta.T @ Q_bar @ (Psi @ x0 - X_ref)
        
        # 确保H是正定的
        try:
            np.linalg.cholesky(H)
        except np.linalg.LinAlgError:
            # 如果H不正定，添加正则化项
            H += 1e-6 * np.eye(H.shape[0])
        
        return H, g
    
    def _build_constraint_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        构建约束矩阵
        
        约束：
        1. 控制输入约束：u_min <= u <= u_max
        2. 控制增量约束：Δu_min <= Δu <= Δu_max
        """
        num_vars = self.N_c * 2
        
        # 控制输入约束数量：每个控制输入有上下界
        num_input_constraints = self.N_c * 4  # 每个u有4个约束（v上下界，ω上下界）
        
        # 控制增量约束数量
        num_rate_constraints = (self.N_c - 1) * 4 if self.N_c > 1 else 0
        
        total_constraints = num_input_constraints + num_rate_constraints
        
        A_ineq = np.zeros((total_constraints, num_vars))
        b_ineq = np.zeros(total_constraints)
        
        constraint_idx = 0
        
        # 1. 控制输入约束
        for i in range(self.N_c):
            # v <= v_max
            A_ineq[constraint_idx, i * 2] = 1.0
            b_ineq[constraint_idx] = self.max_linear_vel
            constraint_idx += 1
            
            # -v <= -v_min (即 v >= v_min)
            A_ineq[constraint_idx, i * 2] = -1.0
            b_ineq[constraint_idx] = self.max_linear_vel
            constraint_idx += 1
            
            # ω <= ω_max
            A_ineq[constraint_idx, i * 2 + 1] = 1.0
            b_ineq[constraint_idx] = self.max_angular_vel
            constraint_idx += 1
            
            # -ω <= -ω_min
            A_ineq[constraint_idx, i * 2 + 1] = -1.0
            b_ineq[constraint_idx] = self.max_angular_vel
            constraint_idx += 1
        
        # 2. 控制增量约束
        max_linear_rate = self.max_linear_acc * self.dt
        max_angular_rate = self.max_angular_acc * self.dt
        
        for i in range(self.N_c - 1):
            # Δv <= Δv_max
            A_ineq[constraint_idx, (i + 1) * 2] = 1.0
            A_ineq[constraint_idx, i * 2] = -1.0
            b_ineq[constraint_idx] = max_linear_rate
            constraint_idx += 1
            
            # -Δv <= -Δv_min
            A_ineq[constraint_idx, (i + 1) * 2] = -1.0
            A_ineq[constraint_idx, i * 2] = 1.0
            b_ineq[constraint_idx] = max_linear_rate
            constraint_idx += 1
            
            # Δω <= Δω_max
            A_ineq[constraint_idx, (i + 1) * 2 + 1] = 1.0
            A_ineq[constraint_idx, i * 2 + 1] = -1.0
            b_ineq[constraint_idx] = max_angular_rate
            constraint_idx += 1
            
            # -Δω <= -Δω_min
            A_ineq[constraint_idx, (i + 1) * 2 + 1] = -1.0
            A_ineq[constraint_idx, i * 2 + 1] = 1.0
            b_ineq[constraint_idx] = max_angular_rate
            constraint_idx += 1
        
        return A_ineq, b_ineq
    
    def _apply_constraints(self, control: MPCControl) -> MPCControl:
        """应用硬约束"""
        control.linear_vel = np.clip(control.linear_vel, 
                                   -self.max_linear_vel, self.max_linear_vel)
        control.angular_vel = np.clip(control.angular_vel,
                                    -self.max_angular_vel, self.max_angular_vel)
        return control
    
    def _update_wheel_speeds(self, control: MPCControl):
        """转换为差分驱动轮速"""
        # 差分驱动运动学逆解
        # v_left = v - ω*L/2
        # v_right = v + ω*L/2
        control.v_left = control.linear_vel - control.angular_vel * self.wheel_base / 2.0
        control.v_right = control.linear_vel + control.angular_vel * self.wheel_base / 2.0
    
    def _predict_trajectory(self, initial_state: MPCState,
                          control_sequence: List[MPCControl]) -> List[MPCState]:
        """预测未来轨迹"""
        predicted_states = [initial_state]
        current_state = initial_state
        
        for control in control_sequence:
            # 差分驱动动力学模型
            next_state = MPCState()
            next_state.x = current_state.x + self.dt * control.linear_vel * np.cos(current_state.theta)
            next_state.y = current_state.y + self.dt * control.linear_vel * np.sin(current_state.theta)
            next_state.theta = MathUtils.normalize_angle(current_state.theta + self.dt * control.angular_vel)
            next_state.v = control.linear_vel
            next_state.omega = control.angular_vel
            
            predicted_states.append(next_state)
            current_state = next_state
        
        return predicted_states
    
    def get_performance_metrics(self) -> dict:
        """获取性能指标"""
        return {
            'computation_time_ms': self.computation_time,
            'solve_success': self.solve_success,
            'prediction_horizon': self.N_p,
            'control_horizon': self.N_c
        }