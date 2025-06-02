# robot/differential_robot.py
"""
差分驱动机器人模型
集成Isaac Sim物理仿真
"""
import numpy as np
from typing import Tuple, Optional, List
import warnings

# 尝试导入Isaac Sim模块，如果不可用则使用模拟版本
try:
    import omni
    from isaacsim.core.api.objects import DynamicCuboid
    from isaacsim.core.api.materials import PreviewSurface, PhysicsMaterial
    from isaacsim.robot.wheeled_robots import DifferentialController
    ISAAC_SIM_AVAILABLE = True
except ImportError:
    warnings.warn("Isaac Sim modules not available, using mock implementations")
    ISAAC_SIM_AVAILABLE = False
    
    # Mock classes for standalone testing
    class DynamicCuboid:
        def __init__(self, *args, **kwargs):
            pass
    
    class PreviewSurface:
        def __init__(self, *args, **kwargs):
            pass
    
    class PhysicsMaterial:
        def __init__(self, *args, **kwargs):
            pass
    
    class DifferentialController:
        def __init__(self, *args, **kwargs):
            pass

from utils.config import config
from core.mpc_controller import MPCState, MPCControl

class DifferentialRobot:
    """
    差分驱动机器人模型
    基于Isaac Sim的物理仿真，支持独立运行模式
    """
    
    def __init__(self, prim_path: str = "/World/Robot", name: str = "differential_robot"):
        self.prim_path = prim_path
        self.name = name
        
        # 机器人参数
        self.length = config.robot.length
        self.width = config.robot.width
        self.height = config.robot.height
        self.wheel_base = config.robot.wheel_base
        self.wheel_radius = config.robot.wheel_radius
        
        # 物理参数
        self.mass = config.robot.mass
        self.max_linear_vel = config.robot.max_linear_vel
        self.max_angular_vel = config.robot.max_angular_vel
        
        # 状态变量
        self.current_state = MPCState()
        self.current_state.x = 0.0
        self.current_state.y = 0.0
        self.current_state.theta = 0.0
        self.current_state.v = 0.0
        self.current_state.omega = 0.0
        
        # Isaac Sim相关
        self.robot_prim = None
        self.controller = None
        self.is_initialized = False
        
        if ISAAC_SIM_AVAILABLE:
            self._initialize_isaac_sim()
        else:
            print("Running in standalone mode without Isaac Sim")
            self.is_initialized = True
    
    def _initialize_isaac_sim(self):
        """初始化Isaac Sim组件"""
        try:
            # 创建机器人几何体
            self.robot_prim = DynamicCuboid(
                prim_path=self.prim_path,
                name=self.name,
                size=np.array([self.length, self.width, self.height]),
                color=np.array([0.2, 0.5, 0.8]),
                mass=self.mass
            )
            
            # 应用材质
            robot_material = PreviewSurface(
                prim_path=f"{self.prim_path}/material",
                name="robot_material"
            )
            robot_material.set_color(color=np.array([0.2, 0.5, 0.8]))
            robot_material.set_roughness(0.4)
            robot_material.set_metallic(0.0)
            self.robot_prim.apply_visual_material(robot_material)
            
            # 物理材质
            physics_material = PhysicsMaterial(
                prim_path=f"{self.prim_path}/physics_material",
                name="robot_physics"
            )
            physics_material.set_static_friction(0.5)
            physics_material.set_dynamic_friction(0.5)
            physics_material.set_restitution(0.0)
            self.robot_prim.apply_physics_material(physics_material)
            
            # 差分驱动控制器
            self.controller = DifferentialController(
                name="diff_controller",
                wheel_radius=self.wheel_radius,
                wheel_base=self.wheel_base
            )
            
            self.is_initialized = True
            print(f"机器人 '{self.name}' 在Isaac Sim中初始化成功")
            
        except Exception as e:
            print(f"Isaac Sim初始化失败: {e}")
            print("切换到独立运行模式")
            self.is_initialized = True
        
        # Isaac Sim组件
        self.robot_prim: Optional[DynamicCuboid] = None
        self.visual_material: Optional[PreviewSurface] = None
        self.physics_material: Optional[PhysicsMaterial] = None
        
        # 控制器
        self.controller = DifferentialController(
            name=f"{name}_controller",
            wheel_base=self.wheel_base,
            wheel_radius=self.wheel_radius
        )
        
        # 状态
        self.current_state = MPCState()
        self.target_control = MPCControl()
        
        # 可视化
        self.trail_positions = []
        self.max_trail_length = config.visualization.robot_trail_length
        
    def initialize(self, initial_position: np.ndarray = np.array([0, 0, 0.1]),
                  initial_orientation: np.ndarray = np.array([1, 0, 0, 0])):
        """初始化机器人"""
        print(f"初始化差分驱动机器人: {self.name}")
        
        # 创建机器人主体（立方体）
        self.robot_prim = DynamicCuboid(
            prim_path=self.prim_path,
            name=self.name,
            position=initial_position,
            orientation=initial_orientation,
            size=np.array([self.length, self.width, self.height]),
            color=config.visualization.robot_color
        )
        
        # 创建视觉材质
        self.visual_material = PreviewSurface(
            prim_path=f"{self.prim_path}/Looks/robot_material",
            name="robot_visual_material",
            color=config.visualization.robot_color,
            roughness=0.3,
            metallic=0.1
        )
        
        # 创建物理材质
        self.physics_material = PhysicsMaterial(
            prim_path=f"{self.prim_path}/PhysicsMaterials/robot_physics_material",
            name="robot_physics_material",
            static_friction=0.8,
            dynamic_friction=0.6,
            restitution=0.1
        )
        
        # 应用材质
        self.robot_prim.apply_visual_material(self.visual_material)
        self.robot_prim.apply_physics_material(self.physics_material)
        
        # 设置物理属性
        self.robot_prim.set_mass(20.0)  # 20kg
        
        # 初始化状态
        self.current_state.x = initial_position[0]
        self.current_state.y = initial_position[1]
        self.current_state.theta = 0.0
        
        print(f"机器人初始化完成，位置: {initial_position}")
    
    def update_state(self):
        """更新机器人状态"""
        if self.robot_prim is None:
            return
        
        # 获取当前位姿
        position, orientation = self.robot_prim.get_world_pose()
        
        # 更新位置
        self.current_state.x = position[0]
        self.current_state.y = position[1]
        
        # 转换四元数到欧拉角
        from scipy.spatial.transform import Rotation
        r = Rotation.from_quat([orientation[1], orientation[2], orientation[3], orientation[0]])
        euler = r.as_euler('xyz', degrees=False)
        self.current_state.theta = euler[2]  # Yaw角
        
        # 获取速度
        linear_velocity = self.robot_prim.get_linear_velocity()
        angular_velocity = self.robot_prim.get_angular_velocity()
        
        if linear_velocity is not None:
            # 计算机体坐标系下的线速度
            cos_theta = np.cos(self.current_state.theta)
            sin_theta = np.sin(self.current_state.theta)
            
            # 世界坐标系速度转机体坐标系
            v_body_x = cos_theta * linear_velocity[0] + sin_theta * linear_velocity[1]
            self.current_state.v = v_body_x
        
        if angular_velocity is not None:
            self.current_state.omega = angular_velocity[2]  # Z轴角速度
        
        # 更新轨迹记录
        self._update_trail()
    
    def apply_control(self, control: MPCControl):
        """应用控制输入"""
        if self.robot_prim is None:
            return
        
        self.target_control = control
        
        # 转换为差分驱动控制
        command = np.array([control.linear_vel, control.angular_vel])
        articulation_action = self.controller.forward(command)
        
        # 应用力控制（简化实现）
        # 在实际Isaac Sim中，这里应该设置轮子的关节速度
        # 这里使用简化的力控制方法
        
        # 计算推进力
        force_magnitude = control.linear_vel * 50.0  # 简化的力缩放
        
        # 计算力的方向
        cos_theta = np.cos(self.current_state.theta)
        sin_theta = np.sin(self.current_state.theta)
        
        force = np.array([
            force_magnitude * cos_theta,
            force_magnitude * sin_theta,
            0.0
        ])
        
        # 计算扭矩
        torque = np.array([0.0, 0.0, control.angular_vel * 10.0])  # 简化的扭矩缩放
        
        # 应用力和扭矩
        try:
            # Isaac Sim 的力控制接口
            self.robot_prim.apply_force(force, is_world_force=True)
            self.robot_prim.apply_torque(torque, is_world_torque=True)
        except Exception as e:
            print(f"应用控制力失败: {e}")
    
    def get_state(self) -> MPCState:
        """获取当前状态"""
        return self.current_state
    
    def set_target_pose(self, position: np.ndarray, orientation: Optional[np.ndarray] = None):
        """设置目标位姿"""
        if self.robot_prim is None:
            return
        
        if orientation is None:
            # 保持当前朝向
            _, current_orientation = self.robot_prim.get_world_pose()
            orientation = current_orientation
        
        self.robot_prim.set_world_pose(position, orientation)
    
    def get_bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取边界框"""
        half_length = self.length / 2.0
        half_width = self.width / 2.0
        half_height = self.height / 2.0
        
        # 局部坐标系边界框
        local_min = np.array([-half_length, -half_width, -half_height])
        local_max = np.array([half_length, half_width, half_height])
        
        # 转换到世界坐标系
        position, orientation = self.robot_prim.get_world_pose() if self.robot_prim else (np.zeros(3), np.array([1, 0, 0, 0]))
        
        # 简化：假设无旋转的边界框
        world_min = position + local_min
        world_max = position + local_max
        
        return world_min, world_max
    
    def get_corner_positions(self) -> np.ndarray:
        """获取机器人四个角点的世界坐标"""
        corners = []
        
        # 局部坐标系角点
        half_length = self.length / 2.0
        half_width = self.width / 2.0
        
        local_corners = [
            [-half_length, -half_width],
            [half_length, -half_width],
            [half_length, half_width],
            [-half_length, half_width]
        ]
        
        # 转换到世界坐标系
        for corner in local_corners:
            # 旋转
            cos_theta = np.cos(self.current_state.theta)
            sin_theta = np.sin(self.current_state.theta)
            
            rotated_x = cos_theta * corner[0] - sin_theta * corner[1]
            rotated_y = sin_theta * corner[0] + cos_theta * corner[1]
            
            # 平移
            world_corner = np.array([
                self.current_state.x + rotated_x,
                self.current_state.y + rotated_y
            ])
            corners.append(world_corner)
        
        return np.array(corners)
    
    def _update_trail(self):
        """更新轨迹记录"""
        current_pos = np.array([self.current_state.x, self.current_state.y])
        
        # 添加当前位置
        self.trail_positions.append(current_pos.copy())
        
        # 限制轨迹长度
        if len(self.trail_positions) > self.max_trail_length:
            self.trail_positions.pop(0)
    
    def get_trail_positions(self) -> List[np.ndarray]:
        """获取轨迹位置"""
        return self.trail_positions.copy()
    
    def reset(self, position: np.ndarray = np.array([0, 0, 0.1]),
             orientation: np.ndarray = np.array([1, 0, 0, 0])):
        """重置机器人状态"""
        if self.robot_prim is None:
            return
        
        # 重置位姿
        self.robot_prim.set_world_pose(position, orientation)
        
        # 重置速度
        self.robot_prim.set_linear_velocity(np.zeros(3))
        self.robot_prim.set_angular_velocity(np.zeros(3))
        
        # 重置状态
        self.current_state = MPCState()
        self.current_state.x = position[0]
        self.current_state.y = position[1]
        self.current_state.theta = 0.0
        
        # 清空轨迹
        self.trail_positions.clear()
        
        print(f"机器人已重置到位置: {position}")
    
    def cleanup(self):
        """清理资源"""
        if self.robot_prim is not None:
            # Isaac Sim的清理操作
            try:
                # 删除prim
                import omni.usd
                stage = omni.usd.get_context().get_stage()
                if stage:
                    prim = stage.GetPrimAtPath(self.prim_path)
                    if prim:
                        stage.RemovePrim(prim.GetPath())
            except Exception as e:
                print(f"清理机器人资源失败: {e}")
        
        self.robot_prim = None
        self.visual_material = None
        self.physics_material = None
        
        print(f"机器人 {self.name} 资源已清理")