# visualization/isaac_sim_visualizer.py
"""
Isaac Sim集成可视化器
实现美观的扫掠体积和轨迹可视化
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
import asyncio
import warnings

# 尝试导入Isaac Sim模块，如果不可用则使用模拟版本
try:
    import omni
    from omni.isaac.core.materials import PreviewSurface, OmniPBR
    from omni.isaac.core.objects import VisualCuboid, VisualSphere, VisualCylinder
    from omni.isaac.core.prims import GeometryPrim
    from omni.isaac.core.utils.prims import create_prim, get_prim_at_path
    from isaacsim.core.api.objects import DynamicSphere, FixedCuboid
    from pxr import Usd, UsdGeom, Gf, UsdShade
    import carb
    ISAAC_SIM_AVAILABLE = True
except ImportError:
    warnings.warn("Isaac Sim modules not available, using mock implementations")
    ISAAC_SIM_AVAILABLE = False
    
    # Mock classes for standalone testing
    class MockStage:
        pass
    
    class MockPrim:
        pass

from utils.config import config
from utils.math_utils import MathUtils

class IsaacSimVisualizer:
    """
    Isaac Sim高级可视化器
    提供美观的轨迹、扫掠体积和机器人状态可视化
    """
    
    def __init__(self, stage=None):
        if not ISAAC_SIM_AVAILABLE:
            print("Isaac Sim not available, visualizer running in mock mode")
            self.stage = MockStage()
            self.visualization_prims = {}
            self.materials = {}
            self.animation_timers = {}
            return
            
        self.stage = stage
        self.visualization_prims = {}
        self.materials = {}
        self.animation_timers = {}
        
        # 可视化设置
        self.trajectory_line_width = config.visualization.trajectory_line_width
        self.swept_volume_alpha = config.visualization.swept_volume_alpha
        self.enable_animations = config.enable_real_time_visualization
        
        # 颜色方案
        self.colors = {
            'trajectory': config.visualization.trajectory_color,
            'swept_volume': config.visualization.swept_volume_color,
            'robot': config.visualization.robot_color,
            'obstacle': config.visualization.obstacle_color,
            'grid': np.array([0.8, 0.8, 0.8]),
            'mpc_prediction': np.array([1.0, 0.8, 0.0]),
            'reference': np.array([0.0, 1.0, 0.0])
        }
        
        # 初始化材质
        self._create_materials()
        
        print("Isaac Sim可视化器已初始化")
    
    def _create_materials(self):
        """创建可视化材质"""
        # 轨迹材质
        self.materials['trajectory'] = PreviewSurface(
            prim_path="/World/Looks/trajectory_material",
            name="trajectory_material",
            color=self.colors['trajectory'],
            roughness=0.2,
            metallic=0.8
        )
        
        # 扫掠体积材质
        self.materials['swept_volume'] = PreviewSurface(
            prim_path="/World/Looks/swept_volume_material", 
            name="swept_volume_material",
            color=self.colors['swept_volume'],
            roughness=0.4,
            metallic=0.1
        )
        
        # 机器人材质
        self.materials['robot'] = OmniPBR(
            prim_path="/World/Looks/robot_material",
            name="robot_material",
            color=self.colors['robot'],
            roughness=0.3,
            metallic=0.7
        )
        
        # 障碍物材质
        self.materials['obstacle'] = PreviewSurface(
            prim_path="/World/Looks/obstacle_material",
            name="obstacle_material", 
            color=self.colors['obstacle'],
            roughness=0.6,
            metallic=0.0
        )
        
        # MPC预测材质
        self.materials['mpc_prediction'] = PreviewSurface(
            prim_path="/World/Looks/mpc_prediction_material",
            name="mpc_prediction_material",
            color=self.colors['mpc_prediction'],
            roughness=0.3,
            metallic=0.5
        )
    
    def create_trajectory_visualization(self, trajectory: List[np.ndarray],
                                     name: str = "trajectory") -> str:
        """
        创建轨迹可视化
        使用连续的圆柱体或球体表示轨迹路径
        """
        trajectory_group_path = f"/World/Visualizations/{name}"
        
        # 创建轨迹组
        create_prim(trajectory_group_path, "Xform")
        
        # 清理旧的轨迹
        self._clear_prim_children(trajectory_group_path)
        
        if len(trajectory) < 2:
            return trajectory_group_path
        
        # 创建轨迹点
        for i, point in enumerate(trajectory):
            point_path = f"{trajectory_group_path}/point_{i:04d}"
            
            # 创建小球表示轨迹点
            sphere = VisualSphere(
                prim_path=point_path,
                name=f"{name}_point_{i}",
                position=np.array([point[0], point[1], 0.05]),
                radius=0.02,
                color=self.colors['trajectory']
            )
            
            # 应用材质
            sphere.apply_visual_material(self.materials['trajectory'])
        
        # 创建轨迹线段
        for i in range(len(trajectory) - 1):
            line_path = f"{trajectory_group_path}/line_{i:04d}"
            
            # 计算线段参数
            start_pos = np.array([trajectory[i][0], trajectory[i][1], 0.05])
            end_pos = np.array([trajectory[i+1][0], trajectory[i+1][1], 0.05])
            
            # 线段中心和方向
            center = (start_pos + end_pos) / 2.0
            direction = end_pos - start_pos
            length = np.linalg.norm(direction)
            
            if length > 1e-6:
                # 计算旋转
                direction_normalized = direction / length
                # 默认圆柱体沿Z轴，需要旋转到XY平面
                default_dir = np.array([0, 0, 1])
                
                # 创建圆柱体表示线段
                cylinder = VisualCylinder(
                    prim_path=line_path,
                    name=f"{name}_line_{i}",
                    position=center,
                    radius=0.01,
                    height=length,
                    color=self.colors['trajectory']
                )
                
                # 应用材质
                cylinder.apply_visual_material(self.materials['trajectory'])
        
        print(f"轨迹可视化已创建: {trajectory_group_path}, 点数: {len(trajectory)}")
        return trajectory_group_path
    
    def create_swept_volume_visualization(self, boundary_points: List[np.ndarray],
                                        density_grid: Optional[np.ndarray] = None,
                                        grid_bounds: Optional[np.ndarray] = None,
                                        name: str = "swept_volume") -> str:
        """创建扫掠体积可视化"""
        swept_volume_path = f"/World/Visualizations/{name}"
        
        # 创建扫掠体积组
        create_prim(swept_volume_path, "Xform")
        self._clear_prim_children(swept_volume_path)
        
        if len(boundary_points) < 3:
            return swept_volume_path
        
        # 1. 创建边界可视化
        boundary_path = f"{swept_volume_path}/boundary"
        self._create_polygon_visualization(boundary_points, boundary_path, 
                                         self.colors['swept_volume'], 0.01)
        
        # 2. 创建填充区域
        fill_path = f"{swept_volume_path}/fill"
        self._create_filled_polygon(boundary_points, fill_path,
                                  self.colors['swept_volume'], self.swept_volume_alpha)
        
        # 3. 如果有密度网格，创建密度可视化
        if density_grid is not None and grid_bounds is not None:
            density_path = f"{swept_volume_path}/density"
            self._create_density_visualization(density_grid, grid_bounds, density_path)
        
        print(f"扫掠体积可视化已创建: {swept_volume_path}")
        return swept_volume_path
    
    def create_robot_trail_visualization(self, trail_positions: List[np.ndarray],
                                       name: str = "robot_trail") -> str:
        """创建机器人轨迹尾迹可视化"""
        trail_path = f"/World/Visualizations/{name}"
        
        # 创建尾迹组
        create_prim(trail_path, "Xform")
        self._clear_prim_children(trail_path)
        
        if len(trail_positions) < 2:
            return trail_path
        
        # 创建渐变的尾迹点
        for i, pos in enumerate(trail_positions):
            point_path = f"{trail_path}/trail_point_{i:04d}"
            
            # 计算透明度渐变
            alpha = (i + 1) / len(trail_positions)
            radius = 0.01 + 0.02 * alpha
            
            # 创建尾迹点
            sphere = VisualSphere(
                prim_path=point_path,
                name=f"{name}_point_{i}",
                position=np.array([pos[0], pos[1], 0.02]),
                radius=radius,
                color=self.colors['robot'] * alpha + np.array([1, 1, 1]) * (1 - alpha)
            )
        
        return trail_path
    
    def create_mpc_prediction_visualization(self, predicted_states: List,
                                          name: str = "mpc_prediction") -> str:
        """创建MPC预测轨迹可视化"""
        prediction_path = f"/World/Visualizations/{name}"
        
        # 创建预测组
        create_prim(prediction_path, "Xform")
        self._clear_prim_children(prediction_path)
        
        if not predicted_states:
            return prediction_path
        
        # 创建预测轨迹点
        for i, state in enumerate(predicted_states):
            point_path = f"{prediction_path}/pred_point_{i:04d}"
            
            # 透明度递减
            alpha = 1.0 - (i / len(predicted_states)) * 0.7
            
            sphere = VisualSphere(
                prim_path=point_path,
                name=f"{name}_point_{i}",
                position=np.array([state.x, state.y, 0.08]),
                radius=0.015,
                color=self.colors['mpc_prediction']
            )
            
            # 应用预测材质
            sphere.apply_visual_material(self.materials['mpc_prediction'])
        
        return prediction_path
    
    def create_obstacles_visualization(self, obstacles: List[Dict],
                                     name: str = "obstacles") -> str:
        """创建障碍物可视化"""
        obstacles_path = f"/World/Visualizations/{name}"
        
        # 创建障碍物组
        create_prim(obstacles_path, "Xform")
        self._clear_prim_children(obstacles_path)
        
        for i, obstacle in enumerate(obstacles):
            obs_path = f"{obstacles_path}/obstacle_{i:04d}"
            
            if obstacle['type'] == 'circle':
                # 圆形障碍物
                center = obstacle['center']
                radius = obstacle['radius']
                
                cylinder = FixedCuboid(  # 使用立方体近似圆形
                    prim_path=obs_path,
                    name=f"{name}_circle_{i}",
                    position=np.array([center[0], center[1], 0.5]),
                    size=np.array([radius*2, radius*2, 1.0]),
                    color=self.colors['obstacle']
                )
                
                # 应用障碍物材质
                cylinder.apply_visual_material(self.materials['obstacle'])
                
            elif obstacle['type'] == 'rectangle':
                # 矩形障碍物
                center = obstacle['center']
                size = obstacle['size']
                
                cuboid = FixedCuboid(
                    prim_path=obs_path,
                    name=f"{name}_rect_{i}",
                    position=np.array([center[0], center[1], 0.5]),
                    size=np.array([size[0], size[1], 1.0]),
                    color=self.colors['obstacle']
                )
                
                cuboid.apply_visual_material(self.materials['obstacle'])
        
        print(f"障碍物可视化已创建: {obstacles_path}, 数量: {len(obstacles)}")
        return obstacles_path
    
    def _create_polygon_visualization(self, points: List[np.ndarray], 
                                    prim_path: str, color: np.ndarray, 
                                    line_width: float):
        """创建多边形边界可视化"""
        create_prim(prim_path, "Xform")
        
        # 创建边界线段
        for i in range(len(points)):
            next_i = (i + 1) % len(points)
            line_path = f"{prim_path}/edge_{i:04d}"
            
            start_pos = np.array([points[i][0], points[i][1], 0.02])
            end_pos = np.array([points[next_i][0], points[next_i][1], 0.02])
            
            # 计算线段参数
            center = (start_pos + end_pos) / 2.0
            length = np.linalg.norm(end_pos - start_pos)
            
            if length > 1e-6:
                # 创建线段
                cylinder = VisualCylinder(
                    prim_path=line_path,
                    name=f"polygon_edge_{i}",
                    position=center,
                    radius=line_width,
                    height=length,
                    color=color
                )
    
    def _create_filled_polygon(self, points: List[np.ndarray], 
                             prim_path: str, color: np.ndarray, alpha: float):
        """创建填充多边形"""
        if len(points) < 3:
            return
        
        # 简化：使用多个小方块填充多边形内部
        # 计算边界框
        points_array = np.array(points)
        x_min, y_min = np.min(points_array, axis=0)
        x_max, y_max = np.max(points_array, axis=0)
        
        create_prim(prim_path, "Xform")
        
        # 网格填充
        grid_size = 0.1
        fill_count = 0
        
        x = x_min
        while x <= x_max:
            y = y_min
            while y <= y_max:
                test_point = np.array([x, y])
                
                # 检查点是否在多边形内
                if GeometryUtils.point_in_polygon(test_point, points):
                    fill_path = f"{prim_path}/fill_{fill_count:04d}"
                    
                    # 创建小立方体
                    cube = VisualCuboid(
                        prim_path=fill_path,
                        name=f"fill_cube_{fill_count}",
                        position=np.array([x, y, 0.01]),
                        size=np.array([grid_size*0.8, grid_size*0.8, 0.02]),
                        color=color
                    )
                    
                    fill_count += 1
                
                y += grid_size
            x += grid_size
    
    def _create_density_visualization(self, density_grid: np.ndarray,
                                    grid_bounds: np.ndarray, prim_path: str):
        """创建密度网格可视化"""
        create_prim(prim_path, "Xform")
        
        x_min, y_min, x_max, y_max = grid_bounds
        grid_height, grid_width = density_grid.shape
        
        grid_res_x = (x_max - x_min) / grid_width
        grid_res_y = (y_max - y_min) / grid_height
        
        max_density = np.max(density_grid)
        if max_density <= 0:
            return
        
        cube_count = 0
        
        for i in range(grid_height):
            for j in range(grid_width):
                density = density_grid[i, j]
                
                if density > 0:
                    # 位置
                    world_x = x_min + j * grid_res_x + grid_res_x / 2
                    world_y = y_min + i * grid_res_y + grid_res_y / 2
                    
                    # 高度和颜色基于密度
                    normalized_density = density / max_density
                    height = 0.1 + normalized_density * 0.3
                    
                    # 颜色渐变：蓝色到红色
                    color = np.array([normalized_density, 0.0, 1.0 - normalized_density])
                    
                    cube_path = f"{prim_path}/density_cube_{cube_count:04d}"
                    
                    cube = VisualCuboid(
                        prim_path=cube_path,
                        name=f"density_cube_{cube_count}",
                        position=np.array([world_x, world_y, height/2]),
                        size=np.array([grid_res_x*0.8, grid_res_y*0.8, height]),
                        color=color
                    )
                    
                    cube_count += 1
    
    def update_robot_visualization(self, robot_pose: np.ndarray, 
                                 robot_prim_path: str):
        """更新机器人可视化"""
        try:
            robot_prim = get_prim_at_path(robot_prim_path)
            if robot_prim and robot_prim.IsValid():
                # 更新位置和方向
                position = np.array([robot_pose[0], robot_pose[1], 0.1])
                
                # 转换角度到四元数
                from scipy.spatial.transform import Rotation
                r = Rotation.from_euler('z', robot_pose[2], degrees=False)
                quaternion = r.as_quat()  # [x, y, z, w]
                
                # Isaac Sim使用 [w, x, y, z] 格式
                orientation = np.array([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])
                
                # 设置变换
                xform = UsdGeom.Xformable(robot_prim)
                if xform:
                    transform_matrix = Gf.Matrix4d()
                    transform_matrix.SetTranslate(Gf.Vec3d(position[0], position[1], position[2]))
                    
                    # 设置旋转
                    rotation_matrix = Gf.Matrix3d(r.as_matrix().T)  # 转置因为Isaac Sim使用列主序
                    transform_matrix.SetRotateOnly(rotation_matrix)
                    
                    xform.GetTransformOp().Set(transform_matrix)
        
        except Exception as e:
            print(f"更新机器人可视化失败: {e}")
    
    def _clear_prim_children(self, prim_path: str):
        """清理prim的所有子节点"""
        try:
            prim = get_prim_at_path(prim_path)
            if prim and prim.IsValid():
                for child in prim.GetChildren():
                    self.stage.RemovePrim(child.GetPath())
        except Exception as e:
            print(f"清理prim子节点失败: {e}")
    
    def animate_trajectory_execution(self, trajectory: List[np.ndarray],
                                   robot_prim_path: str, duration: float = None):
        """动画展示轨迹执行"""
        if not self.enable_animations:
            return
        
        if not trajectory:
            return
        
        # 计算动画持续时间
        if duration is None:
            duration = trajectory[-1][3] - trajectory[0][3] if len(trajectory) > 1 else 5.0
        
        # 启动动画协程
        asyncio.create_task(self._animate_trajectory_coroutine(
            trajectory, robot_prim_path, duration))
    
    async def _animate_trajectory_coroutine(self, trajectory: List[np.ndarray],
                                          robot_prim_path: str, duration: float):
        """轨迹动画协程"""
        start_time = trajectory[0][3] if trajectory else 0.0
        fps = config.visualization.visualization_fps
        frame_time = 1.0 / fps
        
        for frame in range(int(duration * fps)):
            current_time = start_time + frame * frame_time
            
            # 插值获取当前机器人位姿
            robot_pose = MathUtils.interpolate_trajectory(trajectory, current_time)
            
            # 更新机器人可视化
            self.update_robot_visualization(robot_pose, robot_prim_path)
            
            # 等待下一帧
            await asyncio.sleep(frame_time)
    
    def create_performance_display(self, performance_data: Dict,
                                 name: str = "performance_display") -> str:
        """创建性能指标显示"""
        # 在Isaac Sim中创建文本显示性能数据
        # 这里简化实现，实际可以使用UI框架
        
        display_path = f"/World/Visualizations/{name}"
        create_prim(display_path, "Xform")
        
        print("=== 性能指标 ===")
        for key, value in performance_data.items():
            if isinstance(value, (int, float)):
                print(f"{key}: {value:.3f}")
            else:
                print(f"{key}: {value}")
        
        return display_path
    
    def cleanup_visualizations(self):
        """清理所有可视化"""
        viz_path = "/World/Visualizations"
        try:
            viz_prim = get_prim_at_path(viz_path)
            if viz_prim and viz_prim.IsValid():
                self.stage.RemovePrim(viz_prim.GetPath())
        except Exception as e:
            print(f"清理可视化失败: {e}")
        
        # 清理材质
        for material in self.materials.values():
            try:
                if hasattr(material, 'prim_path'):
                    material_prim = get_prim_at_path(material.prim_path)
                    if material_prim and material_prim.IsValid():
                        self.stage.RemovePrim(material_prim.GetPath())
            except Exception as e:
                print(f"清理材质失败: {e}")
        
        self.materials.clear()
        self.visualization_prims.clear()
        
        print("可视化已清理")

from utils.math_utils import GeometryUtils  # 确保导入