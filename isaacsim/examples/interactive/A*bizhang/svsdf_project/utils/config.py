# utils/config.py
"""
SVSDF轨迹规划系统配置文件
"""
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

@dataclass
class RobotConfig:
    """机器人配置参数"""
    # 几何参数
    length: float = 0.6        # 机器人长度 (m)
    width: float = 0.4         # 机器人宽度 (m)
    height: float = 0.2        # 机器人高度 (m)
    wheel_base: float = 0.3    # 轮距 (m)
    wheel_radius: float = 0.05 # 车轮半径 (m)
    
    # 物理参数
    mass: float = 20.0         # 机器人质量 (kg)
    
    # 运动约束
    max_linear_velocity: float = 1.0    # 最大线速度 (m/s)
    max_angular_velocity: float = 1.5   # 最大角速度 (rad/s)
    max_linear_acceleration: float = 2.0 # 最大线加速度 (m/s²)
    max_angular_acceleration: float = 3.0 # 最大角加速度 (rad/s²)
    
    # 兼容性别名
    max_linear_vel: float = field(init=False)
    max_angular_vel: float = field(init=False)
    
    # 安全参数
    safety_margin: float = 0.2  # 安全距离 (m)
    
    def __post_init__(self):
        # 设置别名
        self.max_linear_vel = self.max_linear_velocity
        self.max_angular_vel = self.max_angular_velocity

@dataclass
class PlanningConfig:
    """轨迹规划配置参数"""
    # A*搜索参数
    grid_resolution: float = 0.1      # 网格分辨率 (m)
    heuristic_weight: float = 1.0     # 启发式权重
    max_iterations: int = 10000       # 最大迭代次数
    
    # MINCO优化参数
    num_segments: int = 8             # 轨迹段数量
    polynomial_order: int = 5         # 多项式阶数
    
    # 第一阶段权重
    stage1_weights: Dict[str, float] = field(default_factory=dict)
    # 第二阶段权重  
    stage2_weights: Dict[str, float] = field(default_factory=dict)
    
    # 优化器参数
    max_opt_iterations: int = 100     # 最大优化迭代次数
    convergence_tolerance: float = 1e-6 # 收敛容差
    
    def __post_init__(self):
        if self.stage1_weights is None:
            self.stage1_weights = {
                'energy': 1.0,
                'time': 1.0,
                'path_deviation': 100.0
            }
        if self.stage2_weights is None:
            self.stage2_weights = {
                'energy': 1.0,
                'time': 1.0,
                'obstacle': 10000.0,
                'swept_volume': 1000.0
            }

@dataclass
class MPCConfig:
    """MPC控制器配置参数"""
    # 时域参数
    prediction_horizon: int = 20      # 预测时域
    control_horizon: int = 10         # 控制时域
    sample_time: float = 0.1          # 采样时间 (s)
    
    # 权重矩阵
    state_weights: Optional[np.ndarray] = None  # 状态权重 Q
    control_weights: Optional[np.ndarray] = None # 控制权重 R
    terminal_weights: Optional[np.ndarray] = None # 终端权重 Qf
    
    def __post_init__(self):
        if self.state_weights is None:
            self.state_weights = np.diag([10.0, 10.0, 5.0])  # [x, y, theta]
        if self.control_weights is None:
            self.control_weights = np.diag([1.0, 1.0])       # [v, omega]
        if self.terminal_weights is None:
            self.terminal_weights = np.diag([20.0, 20.0, 10.0])

@dataclass
class VisualizationConfig:
    """可视化配置参数"""
    # 颜色设置
    trajectory_color: np.ndarray = field(default_factory=lambda: np.array([0.2, 0.6, 1.0]))      # 蓝色轨迹
    swept_volume_color: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.3, 0.3]))    # 红色扫掠体积
    robot_color: np.ndarray = field(default_factory=lambda: np.array([0.9, 0.7, 0.1]))          # 金色机器人
    obstacle_color: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.2, 0.8]))       # 紫色障碍物
    
    # 可视化参数
    trajectory_line_width: float = 3.0
    swept_volume_alpha: float = 0.3
    robot_trail_length: int = 50
    
    # 更新频率
    visualization_fps: int = 30

class GlobalConfig:
    """全局配置管理器"""
    def __init__(self):
        self.robot = RobotConfig()
        self.planning = PlanningConfig()
        self.mpc = MPCConfig()
        self.visualization = VisualizationConfig()
        
        # 仿真参数
        self.physics_dt = 1.0 / 60.0
        self.rendering_dt = 1.0 / 60.0
        
        # 世界边界
        self.world_bounds = np.array([-10.0, -10.0, 10.0, 10.0])  # [x_min, y_min, x_max, y_max]
        
        # 性能参数
        self.enable_gpu_acceleration = True
        self.max_planning_time = 5.0  # 最大规划时间 (s)
        self.enable_real_time_visualization = True

# 全局配置实例
config = GlobalConfig()