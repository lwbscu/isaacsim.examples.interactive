#!/usr/bin/env python3
"""
SVSDF增强的A*路径规划 - 基于现有框架的实现
"""

# 从 isaacsim 的交互式示例基础模块导入 BaseSample 类
from isaacsim.examples.interactive.base_sample import BaseSample
import numpy as np
import math
from queue import PriorityQueue
import time
from typing import List, Tuple, Optional, Callable
from scipy.spatial.transform import Rotation as R

# 导入Isaac Sim相关模块
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.robot.wheeled_robots import DifferentialController
from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import UsdGeom, Gf, Usd
import isaacsim.core.utils.prims as prim_utils

# 导入SVSDF轨迹规划器
from svsdf_planner import SVSDFPlanner, RobotParams, TrajectoryPoint

class SimpleAStarPlanner:
    """简化版A*路径规划器"""
    
    def __init__(self, grid_size=150, cell_size=0.2):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.grid = np.zeros((grid_size, grid_size), dtype=np.int32)
        print(f"Grid initialized with size {grid_size}x{grid_size}, cell size {cell_size}")
        
    def world_to_grid(self, world_pos):
        """世界坐标转网格坐标"""
        offset = self.grid_size * self.cell_size / 2
        grid_x = int((world_pos[0] + offset) / self.cell_size)
        grid_y = int((world_pos[1] + offset) / self.cell_size)
        grid_x = max(0, min(grid_x, self.grid_size - 1))
        grid_y = max(0, min(grid_y, self.grid_size - 1))
        return (grid_x, grid_y)
    
    def grid_to_world(self, grid_pos):
        """网格坐标转世界坐标"""
        offset = self.grid_size * self.cell_size / 2
        world_x = grid_pos[0] * self.cell_size - offset + self.cell_size/2
        world_y = grid_pos[1] * self.cell_size - offset + self.cell_size/2
        return (world_x, world_y)
    
    def add_obstacle(self, center, size):
        """添加障碍物到网格"""
        center_grid = self.world_to_grid(center)
        radius_x = int(size[0] / (2 * self.cell_size)) + 2
        radius_y = int(size[1] / (2 * self.cell_size)) + 2
        
        count = 0
        for i in range(max(0, center_grid[1] - radius_y), 
                      min(self.grid_size, center_grid[1] + radius_y + 1)):
            for j in range(max(0, center_grid[0] - radius_x), 
                          min(self.grid_size, center_grid[0] + radius_x + 1)):
                self.grid[i, j] = 1
                count += 1
        print(f"Added obstacle at {center}, grid center {center_grid}, marked {count} grid cells")
        
    def heuristic(self, a, b):
        """欧几里得距离启发式"""
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def find_path(self, start_world, goal_world):
        """A*寻路"""
        start = self.world_to_grid(start_world)
        goal = self.world_to_grid(goal_world)
        
        print(f"Planning path from {start} to {goal}")
        
        # 检查起始点和目标点是否在障碍物内
        if self.grid[start[1], start[0]] == 1:
            print(f"Error: Start position {start} is in obstacle!")
            # 尝试找到附近的自由空间
            for radius in range(1, 10):
                for dy in range(-radius, radius+1):
                    for dx in range(-radius, radius+1):
                        new_start = (start[0] + dx, start[1] + dy)
                        if (0 <= new_start[0] < self.grid_size and 
                            0 <= new_start[1] < self.grid_size and 
                            self.grid[new_start[1], new_start[0]] == 0):
                            print(f"Found free space near start: {new_start}")
                            start = new_start
                            break
                    else:
                        continue
                    break
                else:
                    continue
                break
            else:
                print("No free space found near start position!")
                return []
                
        if self.grid[goal[1], goal[0]] == 1:
            print(f"Error: Goal position {goal} is in obstacle!")
            # 尝试找到附近的自由空间
            for radius in range(1, 10):
                for dy in range(-radius, radius+1):
                    for dx in range(-radius, radius+1):
                        new_goal = (goal[0] + dx, goal[1] + dy)
                        if (0 <= new_goal[0] < self.grid_size and 
                            0 <= new_goal[1] < self.grid_size and 
                            self.grid[new_goal[1], new_goal[0]] == 0):
                            print(f"Found free space near goal: {new_goal}")
                            goal = new_goal
                            break
                    else:
                        continue
                    break
                else:
                    continue
                break
            else:
                print("No free space found near goal position!")
                return []
        
        open_set = PriorityQueue()
        open_set.put((0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        visited = 0
        
        while not open_set.empty():
            current = open_set.get()[1]
            visited += 1
            
            if current == goal:
                print(f"Path found! Visited {visited} nodes")
                # 重建路径
                path = []
                while current in came_from:
                    world_pos = self.grid_to_world(current)
                    path.append([world_pos[0], world_pos[1], 0])
                    current = came_from[current]
                path.append([self.grid_to_world(start)[0], self.grid_to_world(start)[1], 0])
                path.reverse()
                print(f"Path length: {len(path)} waypoints")
                return path
            
            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)
                
                if (0 <= neighbor[0] < self.grid_size and 
                    0 <= neighbor[1] < self.grid_size and 
                    self.grid[neighbor[1], neighbor[0]] == 0):
                    
                    tentative_g = g_score[current] + self.heuristic(current, neighbor)
                    
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                        open_set.put((f_score[neighbor], neighbor))
        
        print(f"No path found after visiting {visited} nodes")
        return []

class SVSDFNavigationSample(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        self._world_settings["physics_dt"] = 1.0 / 60.0
        self._world_settings["rendering_dt"] = 1.0 / 60.0
        return

    def setup_scene(self) -> None:
        """设置场景"""
        world = self.get_world()
        
        # 添加地面
        world.scene.add_default_ground_plane()
        
        # 加载create_3机器人
        robot_usd_path = "/home/lwb/isaacsim_assets/Assets/Isaac/4.5/Isaac/Robots/iRobot/create_4.usd"
        self.robot_prim_path = "/World/create_3"
        
        # 添加机器人到场景
        add_reference_to_stage(robot_usd_path, self.robot_prim_path)
        
        # 获取机器人prim和它的transform
        self.robot_prim = world.stage.GetPrimAtPath(self.robot_prim_path)
        self.robot_xform = UsdGeom.Xformable(self.robot_prim)
        
        # 创建差分控制器
        self.controller = DifferentialController(
            name="diff_controller",
            wheel_radius=0.0508,
            wheel_base=0.235,
            max_linear_speed=0.5,
            max_angular_speed=1.5
        )
        
        # 路径规划器
        self.planner = SimpleAStarPlanner()
        
        # SVSDF轨迹规划器
        robot_params = RobotParams(
            length=0.35,      # Create-3机器人长度
            width=0.33,       # Create-3机器人宽度  
            wheel_base=0.235, # Create-3轮距
            max_vel=0.5,      # 最大线速度
            max_omega=1.5,    # 最大角速度
            max_acc=2.0,      # 最大线加速度
            max_alpha=3.0     # 最大角加速度
        )
        self.svsdf_planner = SVSDFPlanner(robot_params)
        
        # 状态变量
        self.current_path = []
        self.current_trajectory = []
        self.waypoint_index = 0
        self.trajectory_index = 0
        self.start_pos = [-10, -10, 0.1]
        self.goal_pos = [10, 10, 0.1]
        self.state = "IDLE"
        self.use_svsdf = True
        self.trajectory_markers = []
        
        # 设置初始位置
        self.set_robot_pose(self.start_pos, 0.0)
        
        # 创建障碍物
        self.create_obstacles()
        
        # 创建目标立方体
        self.create_target_cube()
        
        print("🤖 SVSDF Enhanced Navigation System Ready!")
        print("Press 'P' to plan path, 'S' to toggle SVSDF mode")
        
        return
    
    def set_robot_pose(self, position, yaw_angle):
        """设置机器人位置和朝向"""
        # 创建旋转四元数
        rotation = R.from_euler('z', yaw_angle)
        quat = rotation.as_quat()  # [x, y, z, w]
        
        # 设置变换矩阵
        transform_matrix = Gf.Matrix4d()
        transform_matrix.SetTranslate(Gf.Vec3d(position[0], position[1], position[2]))
        transform_matrix.SetRotateOnly(Gf.Quatd(quat[3], quat[0], quat[1], quat[2]))  # w, x, y, z
        
        # 应用变换
        self.robot_xform.SetLocalTransformation(transform_matrix)
        print(f"Robot positioned at: {position[:2]}, yaw: {math.degrees(yaw_angle):.1f}°")
    
    def get_robot_pose(self):
        """获取机器人位置和朝向"""
        transform_matrix = self.robot_xform.GetLocalTransformation()
        translation = transform_matrix.ExtractTranslation()
        rotation_quat = transform_matrix.ExtractRotation().GetQuat()
        
        position = np.array([translation[0], translation[1], translation[2]])
        # 转换四元数格式: USD的Quatd(w,x,y,z) -> scipy的(x,y,z,w)
        rotation = np.array([rotation_quat.GetReal(), rotation_quat.GetImaginary()[0], 
                           rotation_quat.GetImaginary()[1], rotation_quat.GetImaginary()[2]])
        
        return position, rotation
    
    def get_robot_yaw(self):
        """获取机器人当前偏航角"""
        try:
            _, rotation = self.get_robot_pose()
            # 从四元数转换为欧拉角
            r = R.from_quat([rotation[1], rotation[2], rotation[3], rotation[0]])  # xyzw to wxyz
            euler = r.as_euler('xyz')
            return euler[2]  # yaw角
        except:
            return 0.0
    
    def create_obstacles(self):
        """创建障碍物"""
        world = self.get_world()
        
        # 静态障碍物定义
        obstacles = [
            {"pos": [3, 3, 0.5], "scale": [2, 2, 1]},
            {"pos": [-3, -3, 0.5], "scale": [2, 2, 1]},
            {"pos": [6, -2, 0.5], "scale": [1.5, 3, 1]},
            {"pos": [-4, 4, 0.5], "scale": [3, 1.5, 1]},
            {"pos": [0, 0, 0.5], "scale": [1, 4, 1]},
        ]
        
        # 创建障碍物
        for i, obs in enumerate(obstacles):
            obstacle = world.scene.add(
                FixedCuboid(
                    prim_path=f"/World/obstacle_{i}",
                    name=f"obstacle_{i}",
                    position=np.array(obs["pos"]),
                    scale=np.array(obs["scale"]),
                    color=np.array([0.8, 0.2, 0.2])
                )
            )
            
            self.planner.add_obstacle(obs["pos"], obs["scale"])
            print(f"Created obstacle {i} at {obs['pos']}")
        
        # 边界墙
        boundary_walls = [
            {"pos": [0, 13, 0.5], "scale": [26, 1, 1]},
            {"pos": [0, -13, 0.5], "scale": [26, 1, 1]},
            {"pos": [13, 0, 0.5], "scale": [1, 26, 1]},
            {"pos": [-13, 0, 0.5], "scale": [1, 26, 1]},
        ]
        
        for i, wall in enumerate(boundary_walls):
            boundary = world.scene.add(
                FixedCuboid(
                    prim_path=f"/World/boundary_{i}",
                    name=f"boundary_{i}",
                    position=np.array(wall["pos"]),
                    scale=np.array(wall["scale"]),
                    color=np.array([0.5, 0.5, 0.5])
                )
            )
            self.planner.add_obstacle(wall["pos"], wall["scale"])
    
    def create_target_cube(self):
        """创建目标立方体"""
        world = self.get_world()
        try:
            self.target_cube = world.scene.add(
                FixedCuboid(
                    prim_path="/World/target_cube",
                    name="target_cube",
                    position=np.array([self.goal_pos[0], self.goal_pos[1], 0.2]),
                    scale=np.array([0.5, 0.5, 0.5]),
                    color=np.array([1.0, 1.0, 0.0])  # 黄色
                )
            )
            print("Created target cube!")
        except Exception as e:
            print(f"Failed to create target cube: {e}")
            self.target_cube = None
    
    def plan_path(self):
        """规划路径 - 集成SVSDF轨迹优化"""
        current_pos, current_rot = self.get_robot_pose()
        print(f"Planning path from {current_pos[:2]} to {self.goal_pos[:2]}")
        print(f"Using {'SVSDF trajectory optimization' if self.use_svsdf else 'simple A* planning'}")
        
        # 清除旧的可视化
        self.clear_path_markers()
        self.clear_trajectory_markers()
        
        # 第一步：使用A*生成初始路径
        astar_path = self.planner.find_path(
            [current_pos[0], current_pos[1]], 
            [self.goal_pos[0], self.goal_pos[1]]
        )
        
        if not astar_path:
            print("No A* path found!")
            return False
        
        # 将A*路径转换为简单的(x,y)元组列表
        simple_path = [(point[0], point[1]) for point in astar_path]
        
        if self.use_svsdf:
            # 第二步：使用SVSDF优化轨迹
            try:
                # 获取当前机器人状态
                current_yaw = self.get_robot_yaw()
                start_state = np.array([current_pos[0], current_pos[1], current_yaw, 0.0, 0.0, 0.0])
                
                # 计算目标朝向
                goal_yaw = math.atan2(self.goal_pos[1] - current_pos[1], 
                                    self.goal_pos[0] - current_pos[0])
                goal_state = np.array([self.goal_pos[0], self.goal_pos[1], goal_yaw, 0.0, 0.0, 0.0])
                
                # 获取障碍物信息
                obstacles = self.get_obstacle_info()
                
                # 运行SVSDF轨迹优化
                print("🚀 Running SVSDF trajectory optimization...")
                trajectory, info = self.svsdf_planner.plan_trajectory(
                    start_state, goal_state, simple_path, obstacles
                )
                
                if trajectory:
                    self.current_trajectory = trajectory
                    self.current_path = simple_path
                    self.trajectory_index = 0
                    
                    # 可视化
                    self.visualize_path(simple_path)
                    self.visualize_trajectory()
                    
                    print(f"✅ SVSDF trajectory planning successful!")
                    print(f"   - Trajectory points: {len(trajectory)}")
                    print(f"   - Swept volume: {info['swept_volume']:.3f}m³")
                    print(f"   - Total time: {info['total_time']:.3f}s")
                    return True
                else:
                    print("❌ SVSDF optimization failed, using A* path")
                    self.use_simple_path_following(simple_path)
                    return True
                    
            except Exception as e:
                print(f"❌ SVSDF planning error: {e}")
                print("Falling back to simple A* path following")
                self.use_simple_path_following(simple_path)
                return True
        else:
            # 使用简单的A*路径跟踪
            self.use_simple_path_following(simple_path)
            return True
    
    def use_simple_path_following(self, astar_path):
        """使用简单的A*路径跟踪"""
        self.current_path = astar_path
        self.current_trajectory = []
        self.waypoint_index = 0
        self.visualize_path(astar_path)
        print(f"Using simple A* path with {len(astar_path)} waypoints")
    
    def get_obstacle_info(self):
        """获取障碍物信息"""
        obstacles = []
        
        # 静态障碍物
        static_obstacles = [
            {"center": [3, 3, 0.5], "size": [2, 2, 1]},
            {"center": [-3, -3, 0.5], "size": [2, 2, 1]},
            {"center": [6, -2, 0.5], "size": [1.5, 3, 1]},
            {"center": [-4, 4, 0.5], "size": [3, 1.5, 1]},
            {"center": [0, 0, 0.5], "size": [1, 4, 1]},
        ]
        
        for obs in static_obstacles:
            obstacles.append({'center': obs['center'], 'size': obs['size']})
        
        # 边界墙
        boundary_walls = [
            {"center": [0, 13, 0.5], "size": [26, 1, 1]},
            {"center": [0, -13, 0.5], "size": [26, 1, 1]},
            {"center": [13, 0, 0.5], "size": [1, 26, 1]},
            {"center": [-13, 0, 0.5], "size": [1, 26, 1]},
        ]
        
        for wall in boundary_walls:
            obstacles.append({'center': wall['center'], 'size': wall['size']})
        
        return obstacles
    
    def clear_path_markers(self):
        """清除路径标记"""
        world = self.get_world()
        try:
            cleared_count = 0
            # 清除所有路径标记
            for i in range(500):
                marker_path = f"/World/path_marker_{i}"
                if world.stage.GetPrimAtPath(marker_path).IsValid():
                    world.stage.RemovePrim(marker_path)
                    cleared_count += 1
            
            # 清除场景中的对象引用
            if hasattr(self, 'path_markers'):
                for marker in self.path_markers:
                    try:
                        if marker and hasattr(marker, 'prim_path'):
                            world.scene.remove_object(marker.name)
                    except:
                        pass
                self.path_markers = []
            
            if cleared_count > 0:
                print(f"Cleared {cleared_count} path markers")
        except Exception as e:
            print(f"Warning: Could not clear path markers: {e}")
    
    def clear_trajectory_markers(self):
        """清除轨迹标记"""
        world = self.get_world()
        try:
            cleared_count = 0
            # 清除所有轨迹标记
            for i in range(500):
                marker_path = f"/World/trajectory_marker_{i}"
                if world.stage.GetPrimAtPath(marker_path).IsValid():
                    world.stage.RemovePrim(marker_path)
                    cleared_count += 1
            
            # 清除场景中的对象引用  
            if hasattr(self, 'trajectory_markers'):
                for marker in self.trajectory_markers:
                    try:
                        if marker and hasattr(marker, 'prim_path'):
                            world.scene.remove_object(marker.name)
                    except:
                        pass
                self.trajectory_markers = []
                
            if cleared_count > 0:
                print(f"Cleared {cleared_count} trajectory markers")
        except Exception as e:
            print(f"Warning: Could not clear trajectory markers: {e}")
    
    def visualize_path(self, path):
        """可视化A*路径 - 使用圆形标记"""
        if not path:
            return
        
        world = self.get_world()
        print(f"🎨 Visualizing A* path with {len(path)} waypoints (green spheres)")
        
        # 确保有path_markers列表
        if not hasattr(self, 'path_markers'):
            self.path_markers = []
        
        try:
            step = max(1, len(path) // 20)  # 最多显示20个标记
            created_count = 0
            
            for i in range(0, len(path), step):
                point = path[i]
                marker_name = f"path_marker_{i}_{int(time.time() * 1000)}"  # 添加时间戳避免重复
                marker_path = f"/World/{marker_name}"
                
                try:
                    # 使用FixedCuboid但设置为更小的尺寸模拟圆形
                    path_marker = world.scene.add(
                        FixedCuboid(
                            prim_path=marker_path,
                            name=marker_name,
                            position=np.array([point[0], point[1], 2.0]),
                            scale=np.array([0.3, 0.3, 0.1]),  # 扁平的形状
                            color=np.array([0.0, 1.0, 0.0])  # 绿色
                        )
                    )
                    self.path_markers.append(path_marker)
                    created_count += 1
                except Exception as e:
                    print(f"Failed to create path marker {i}: {e}")
            
            print(f"✅ Created {created_count} green path markers")
        except Exception as e:
            print(f"Error visualizing path: {e}")
    
    def visualize_trajectory(self):
        """可视化SVSDF轨迹 - 使用蓝色圆环显示扫掠体积"""
        if not self.current_trajectory:
            return
        
        world = self.get_world()
        print(f"🎨 Visualizing SVSDF trajectory with swept volume (blue circles)")
        
        # 确保有trajectory_markers列表
        if not hasattr(self, 'trajectory_markers'):
            self.trajectory_markers = []
        
        try:
            step = max(1, len(self.current_trajectory) // 25)  # 最多25个标记
            created_count = 0
            
            for i in range(0, len(self.current_trajectory), step):
                traj_point = self.current_trajectory[i]
                marker_name = f"traj_marker_{i}_{int(time.time() * 1000)}"
                marker_path = f"/World/{marker_name}"
                
                try:
                    # 使用较大的扁平立方体表示扫掠体积
                    robot_radius = max(0.35, 0.33) / 2 + 0.2  # 机器人半径 + 安全余量
                    
                    traj_marker = world.scene.add(
                        FixedCuboid(
                            prim_path=marker_path,
                            name=marker_name,
                            position=np.array([traj_point.position[0], traj_point.position[1], 0.05]),
                            scale=np.array([robot_radius*2, robot_radius*2, 0.05]),  # 扁平的正方形表示扫掠区域
                            color=np.array([0.2, 0.5, 1.0])  # 蓝色
                        )
                    )
                    self.trajectory_markers.append(traj_marker)
                    created_count += 1
                except Exception as e:
                    print(f"Failed to create trajectory marker {i}: {e}")
            
            print(f"✅ Created {created_count} blue trajectory markers showing swept volume")
        except Exception as e:
            print(f"Error visualizing trajectory: {e}")
    
    def world_cleanup(self):
        """清理世界"""
        return
