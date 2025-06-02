#!/usr/bin/env python3
"""
A*算法避障项目 - 交互式版本
支持实时目标选择和拖拽功能
"""

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import carb
import omni
import omni.appwindow
import omni.ui as ui
import omni.usd
import os
import numpy as np
import math
from queue import PriorityQueue
import time
from scipy.spatial.transform import Rotation as R

# Isaac Sim imports
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.robot.wheeled_robots import DifferentialController
from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import UsdGeom, Gf, Usd
import isaacsim.core.utils.prims as prim_utils

# SVSDF轨迹规划器导入
from svsdf_planner import SVSDFPlanner, RobotParams, TrajectoryPoint

# 设置资源路径
asset_root = "/home/lwb/isaacsim_assets/Assets/Isaac/4.5"
carb.settings.get_settings().set("/persistent/isaac/asset_root/default", asset_root)

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

class InteractiveAvoidanceRobot:
    """交互式避障机器人类"""
    
    def __init__(self, world):
        self.world = world
        
        # 加载create_3机器人
        robot_usd_path = "/home/lwb/isaacsim_assets/Assets/Isaac/4.5/Isaac/Robots/iRobot/create_4.usd"
        self.robot_prim_path = "/World/create_3"
        
        # 添加机器人到场景
        add_reference_to_stage(robot_usd_path, self.robot_prim_path)
        
        # 获取机器人prim和它的transform
        self.robot_prim = self.world.stage.GetPrimAtPath(self.robot_prim_path)
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
        
        # 轨迹跟踪变量
        self.current_trajectory = []
        self.trajectory_index = 0
        self.use_svsdf = True  # 是否使用SVSDF轨迹优化
        self.trajectory_markers = []  # 轨迹可视化标记
        
        # 状态变量
        self.current_path = []
        self.waypoint_index = 0
        self.start_pos = [-10, -10, 0.1]
        self.goal_pos = [10, 10, 0.1]
        self.state = "IDLE"
        
        # 运动状态
        self.current_position = np.array(self.start_pos)
        self.current_orientation = 0.0
        
        # 交互控制
        self.auto_navigation = False
        self.target_cube = None
        self.goal_changed = False
        
        # 输入处理
        self._appwindow = None
        self._input = None
        self._keyboard = None
        self._sub_keyboard = None
        
        # 键盘映射
        self._input_keyboard_mapping = {
            # 目标移动
            "NUMPAD_8": [0, 2.0],    # 向前移动目标
            "UP": [0, 2.0],
            "NUMPAD_2": [0, -2.0],   # 向后移动目标
            "DOWN": [0, -2.0],
            "NUMPAD_4": [-2.0, 0],   # 向左移动目标
            "LEFT": [-2.0, 0],
            "NUMPAD_6": [2.0, 0],    # 向右移动目标
            "RIGHT": [2.0, 0],
            # 控制键
            "SPACE": "toggle_auto",   # 开始/停止自动导航
            "R": "replan",           # 重新规划路径
            "T": "new_target",       # 设置新目标
            "S": "toggle_svsdf",     # 切换SVSDF/A*规划模式
            "V": "visualize_trajectory", # 可视化当前轨迹
        }
        
        # 设置初始位置
        self.set_robot_pose(self.start_pos, 0.0)
        
        # 初始化输入系统
        self.setup_input_handling()
    
    def setup_input_handling(self):
        """设置输入处理"""
        try:
            self._appwindow = omni.appwindow.get_default_app_window()
            self._input = carb.input.acquire_input_interface()
            self._keyboard = self._appwindow.get_keyboard()
            self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._sub_keyboard_event)
            print("Input handling initialized successfully")
        except Exception as e:
            print(f"Failed to setup input handling: {e}")
    
    def _sub_keyboard_event(self, event, *args, **kwargs) -> bool:
        """键盘事件处理"""
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            key_name = event.input.name
            
            if key_name in self._input_keyboard_mapping:
                action = self._input_keyboard_mapping[key_name]
                
                if isinstance(action, list):  # 移动目标
                    self.move_target(action[0], action[1])
                elif action == "toggle_auto":
                    self.toggle_auto_navigation()
                elif action == "replan":
                    self.request_replan()
                elif action == "new_target":
                    self.set_random_target()
                elif action == "toggle_svsdf":
                    self.toggle_svsdf_mode()
                elif action == "visualize_trajectory":
                    self.visualize_current_trajectory()
                    
        return True
    
    def move_target(self, dx, dy):
        """移动目标位置"""
        self.goal_pos[0] += dx
        self.goal_pos[1] += dy
        
        # 限制目标在合理范围内
        self.goal_pos[0] = max(-12, min(12, self.goal_pos[0]))
        self.goal_pos[1] = max(-12, min(12, self.goal_pos[1]))
        
        print(f"Target moved to: {self.goal_pos[:2]}")
        self.goal_changed = True
        
        # 更新目标立方体位置
        self.update_target_cube_position()
    
    def update_target_cube_position(self):
        """更新目标立方体的位置 - 使用USD直接操作避免物理后端问题"""
        if self.target_cube:
            try:
                # 直接使用USD操作，避免物理后端问题
                target_prim_path = "/World/target_cube"
                target_prim = self.world.stage.GetPrimAtPath(target_prim_path)
                
                if target_prim.IsValid():
                    xform = UsdGeom.Xformable(target_prim)
                    # 清除现有变换
                    xform.ClearXformOpOrder()
                    # 设置新位置
                    translate_op = xform.AddTranslateOp()
                    translate_op.Set(Gf.Vec3d(self.goal_pos[0], self.goal_pos[1], 0.2))
                    print(f"Target cube updated to position: {self.goal_pos[:2]}")
                else:
                    print("Warning: Target cube prim not found")
            except Exception as e:
                print(f"Failed to update target cube position: {e}")
                # 如果更新失败，尝试重新创建目标立方体
                self.recreate_target_cube()
    
    def toggle_auto_navigation(self):
        """切换自动导航模式"""
        self.auto_navigation = not self.auto_navigation
        if self.auto_navigation:
            print("Auto navigation ENABLED - Robot will follow the target")
            self.state = "PLANNING"
        else:
            print("Auto navigation DISABLED - Use arrow keys to move target, SPACE to start")
            self.state = "IDLE"
    
    def request_replan(self):
        """请求重新规划路径"""
        if self.auto_navigation:
            print("Replanning path...")
            self.state = "PLANNING"
            self.goal_changed = False
    
    def set_random_target(self):
        """设置随机目标位置"""
        # 在合理范围内生成随机目标
        self.goal_pos[0] = np.random.uniform(-10, 10)
        self.goal_pos[1] = np.random.uniform(-10, 10)
        print(f"New random target: {self.goal_pos[:2]}")
        self.goal_changed = True
        self.update_target_cube_position()
        
        if self.auto_navigation:
            self.state = "PLANNING"
    
    def set_robot_pose(self, position, yaw):
        """设置机器人位置和朝向"""
        if self.robot_prim:
            # 清除现有的XForm操作
            self.robot_xform.ClearXformOpOrder()
            
            # 设置平移
            translate_op = self.robot_xform.AddTranslateOp()
            translate_op.Set(Gf.Vec3d(position[0], position[1], position[2]))
            
            # 设置旋转
            rotate_op = self.robot_xform.AddRotateZOp()
            rotate_op.Set(math.degrees(yaw))
            
            # 更新当前状态
            self.current_position = np.array(position)
            self.current_orientation = yaw
    
    def get_robot_pose(self):
        """获取机器人当前位置"""
        return self.current_position.copy(), self.current_orientation
    
    def recreate_target_cube(self):
        """重新创建目标立方体"""
        try:
            # 删除旧的目标立方体
            target_prim_path = "/World/target_cube"
            if self.world.stage.GetPrimAtPath(target_prim_path).IsValid():
                self.world.stage.RemovePrim(target_prim_path)
            
            # 创建新的目标立方体，使用FixedCuboid避免物理问题
            self.target_cube = self.world.scene.add(
                FixedCuboid(
                    prim_path="/World/target_cube",
                    name="target_cube",
                    position=np.array([self.goal_pos[0], self.goal_pos[1], 0.2]),
                    scale=np.array([0.5, 0.5, 0.5]),
                    color=np.array([1.0, 1.0, 0.0])  # 黄色
                )
            )
            print("Target cube recreated successfully")
        except Exception as e:
            print(f"Failed to recreate target cube: {e}")

    def create_obstacles(self):
        """创建障碍物"""
        obstacles = [
            {"pos": [0, 0, 0.5], "scale": [2, 2, 1]},
            {"pos": [5, 0, 0.5], "scale": [1, 6, 1]},
            {"pos": [-5, 0, 0.5], "scale": [1, 6, 1]},
            {"pos": [0, 5, 0.5], "scale": [6, 1, 1]},
            {"pos": [0, -5, 0.5], "scale": [6, 1, 1]},
        ]
        
        # 创建障碍物
        for i, obs in enumerate(obstacles):
            obstacle = self.world.scene.add(
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
            boundary = self.world.scene.add(
                FixedCuboid(
                    prim_path=f"/World/boundary_{i}",
                    name=f"boundary_{i}",
                    position=np.array(wall["pos"]),
                    scale=np.array(wall["scale"]),
                    color=np.array([0.5, 0.5, 0.5])
                )
            )
            self.planner.add_obstacle(wall["pos"], wall["scale"])
        
        # 创建可交互的目标立方体 - 使用FixedCuboid避免物理问题
        try:
            self.target_cube = self.world.scene.add(
                FixedCuboid(
                    prim_path="/World/target_cube",
                    name="target_cube",
                    position=np.array([self.goal_pos[0], self.goal_pos[1], 0.2]),
                    scale=np.array([0.5, 0.5, 0.5]),
                    color=np.array([1.0, 1.0, 0.0])  # 黄色
                )
            )
            print("Created interactive target cube - use arrow keys to move it!")
        except Exception as e:
            print(f"Failed to create target cube: {e}")
            self.target_cube = None
    
    def plan_path(self):
        """规划路径 - 集成SVSDF轨迹优化"""
        current_pos, current_rot = self.get_robot_pose()
        print(f"Planning path from {current_pos[:2]} to {self.goal_pos[:2]}")
        print(f"Using {'SVSDF trajectory optimization' if self.use_svsdf else 'simple A* planning'}")
        
        # 先清除旧的路径可视化
        self.clear_path_markers()
        self.clear_trajectory_markers()
        
        # 第一步：使用A*生成初始路径
        astar_path = self.planner.find_path(
            [current_pos[0], current_pos[1]], 
            [self.goal_pos[0], self.goal_pos[1]]
        )
        
        if not astar_path:
            print("No A* path found!")
            self.state = "IDLE"
            return False
        
        # 将A*路径转换为简单的(x,y)元组列表
        simple_path = [(point[0], point[1]) for point in astar_path]
        
        if self.use_svsdf:
            # 第二步：使用SVSDF优化轨迹
            try:
                # 获取当前机器人状态
                current_yaw = self.get_robot_yaw()
                start_state = np.array([current_pos[0], current_pos[1], current_yaw, 0.0, 0.0, 0.0])
                
                # 计算目标朝向（朝向目标点）
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
                    self.current_path = simple_path  # 保留A*路径用于fallback
                    self.trajectory_index = 0
                    
                    # 可视化
                    self.visualize_path()  # A*路径（绿色）
                    self.visualize_trajectory()  # SVSDF轨迹（蓝色）
                    
                    print(f"✅ SVSDF trajectory planning successful!")
                    print(f"   - Trajectory points: {len(trajectory)}")
                    print(f"   - Swept volume: {info['swept_volume']:.3f}m³")
                    print(f"   - Total time: {info['total_time']:.3f}s")
                    return True
                else:
                    print("❌ SVSDF optimization failed, falling back to A* path")
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
        self.current_trajectory = []  # 清空轨迹
        self.waypoint_index = 0
        self.visualize_path()
        print(f"Using simple A* path with {len(astar_path)} waypoints")
    
    def visualize_path(self):
        """可视化路径 - 使用FixedCuboid避免物理系统冲突"""
        if not self.current_path:
            print("No path to visualize")
            return
            
        print(f"🎨 Visualizing path with {len(self.current_path)} waypoints")
        
        # 使用FixedCuboid标记路径，避免物理冲突
        try:
            # 显示完整路径，每个路径点都标记
            path_length = len(self.current_path)
            created_count = 0
            
            for i in range(path_length):
                point = self.current_path[i]
                marker_path = f"/World/path_marker_{i}"
                
                try:
                    # 使用FixedCuboid创建路径标记，提高高度到3
                    path_marker = self.world.scene.add(
                        FixedCuboid(
                            prim_path=marker_path,
                            name=f"path_marker_{i}",
                            position=np.array([point[0], point[1], 3.0]),  # 高度提高到3
                            scale=np.array([0.15, 0.15, 0.15]),
                            color=np.array([0.0, 1.0, 0.0])  # 绿色
                        )
                    )
                    created_count += 1
                except Exception as marker_error:
                    print(f"Failed to create marker {i}: {marker_error}")
            
            print(f"✅ Path visualization created with {created_count}/{path_length} markers for complete path")
                
        except Exception as e:
            print(f"❌ Error: Could not visualize path: {e}")
            import traceback
            traceback.print_exc()
    
    def clear_path_markers(self):
        """清除路径标记"""
        try:
            cleared_count = 0
            # 清除所有可能的路径标记，增加范围以适应完整路径显示
            for i in range(500):  # 增加清除范围，适应更多路径点
                marker_path = f"/World/path_marker_{i}"
                if self.world.stage.GetPrimAtPath(marker_path).IsValid():
                    self.world.stage.RemovePrim(marker_path)
                    cleared_count += 1
            if cleared_count > 0:
                print(f"Cleared {cleared_count} old path markers")
        except Exception as e:
            print(f"Warning: Could not clear path markers: {e}")
    
    def toggle_svsdf_mode(self):
        """切换SVSDF/A*规划模式"""
        self.use_svsdf = not self.use_svsdf
        mode_name = "SVSDF trajectory optimization" if self.use_svsdf else "Simple A* planning"
        print(f"🔄 Switched to {mode_name} mode")
        
        # 如果正在导航，重新规划
        if self.auto_navigation:
            self.request_replan()
    
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
    
    def get_obstacle_info(self):
        """获取障碍物信息，用于SVSDF规划"""
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
            obstacles.append({
                'center': obs['center'],
                'size': obs['size']
            })
        
        # 边界墙
        boundary_walls = [
            {"center": [0, 13, 0.5], "size": [26, 1, 1]},
            {"center": [0, -13, 0.5], "size": [26, 1, 1]},
            {"center": [13, 0, 0.5], "size": [1, 26, 1]},
            {"center": [-13, 0, 0.5], "size": [1, 26, 1]},
        ]
        
        for wall in boundary_walls:
            obstacles.append({
                'center': wall['center'],
                'size': wall['size']
            })
        
        return obstacles
    
    def clear_trajectory_markers(self):
        """清除轨迹可视化标记"""
        try:
            for marker in self.trajectory_markers:
                if marker and hasattr(marker, 'prim_path'):
                    prim_path = marker.prim_path
                    if self.world.stage.GetPrimAtPath(prim_path):
                        self.world.stage.RemovePrim(prim_path)
            self.trajectory_markers.clear()
            print(f"Cleared {len(self.trajectory_markers)} trajectory markers")
        except Exception as e:
            print(f"Error clearing trajectory markers: {e}")
    
    def visualize_trajectory(self):
        """可视化SVSDF优化后的轨迹"""
        if not self.current_trajectory:
            print("No trajectory to visualize")
            return
        
        print(f"🎨 Visualizing SVSDF trajectory with {len(self.current_trajectory)} points")
        
        try:
            # 清除旧的轨迹标记
            self.clear_trajectory_markers()
            
            # 使用蓝色标记显示轨迹，每5个点显示一个标记以避免过密
            step = max(1, len(self.current_trajectory) // 20)  # 最多显示20个标记
            created_count = 0
            
            for i in range(0, len(self.current_trajectory), step):
                traj_point = self.current_trajectory[i]
                marker_path = f"/World/trajectory_marker_{i}"
                
                try:
                    # 使用蓝色标记区分SVSDF轨迹
                    traj_marker = self.world.scene.add(
                        FixedCuboid(
                            prim_path=marker_path,
                            name=f"trajectory_marker_{i}",
                            position=np.array([traj_point.position[0], traj_point.position[1], 2.5]),
                            scale=np.array([0.2, 0.2, 0.2]),
                            color=np.array([0.0, 0.5, 1.0])  # 蓝色
                        )
                    )
                    self.trajectory_markers.append(traj_marker)
                    created_count += 1
                except Exception as e:
                    print(f"Failed to create trajectory marker {i}: {e}")
            
            print(f"✅ Created {created_count} trajectory markers (blue)")
        
        except Exception as e:
            print(f"❌ Error visualizing trajectory: {e}")
    
    def visualize_current_trajectory(self):
        """手动触发轨迹可视化"""
        if self.current_trajectory:
            self.visualize_trajectory()
            print("📊 Current trajectory visualization refreshed")
        else:
            print("❌ No current trajectory to visualize")
    
    def update(self):
        """更新机器人状态"""
        # 检查目标是否改变
        if self.goal_changed and self.auto_navigation:
            print("Target changed - replanning...")
            self.state = "PLANNING"
            self.goal_changed = False
        
        if self.state == "IDLE":
            return True
        
        elif self.state == "PLANNING":
            print("🎯 Planning new path...")
            if self.plan_path():
                self.state = "MOVING"
                print("✅ Path planned successfully - Starting navigation...")
            else:
                print("❌ Failed to find path!")
                self.state = "IDLE"
            return True
        
        elif self.state == "MOVING":
            return self.follow_path()
        
        elif self.state == "REACHED":
            print("🎉 Target reached! Waiting for new commands...")
            self.state = "IDLE"
            return True
        
        return True
    
    def follow_path(self):
        """跟随路径 - 支持SVSDF轨迹跟踪和简单A*路径跟踪"""
        if self.current_trajectory:
            # 使用SVSDF轨迹跟踪
            return self.follow_trajectory()
        else:
            # 使用简单A*路径跟踪
            return self.follow_simple_path()
    
    def follow_trajectory(self):
        """跟随SVSDF优化后的轨迹"""
        if self.trajectory_index >= len(self.current_trajectory):
            self.state = "REACHED"
            return True
        
        current_pos, _ = self.get_robot_pose()
        current_yaw = self.get_robot_yaw()
        
        # 获取当前轨迹点
        traj_point = self.current_trajectory[self.trajectory_index]
        target_pos = traj_point.position[:2]
        target_yaw = traj_point.position[2]
        
        # 计算到目标的距离和角度
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        distance = math.sqrt(dx**2 + dy**2)
        
        # 调试信息
        if self.trajectory_index % 10 == 0:
            print(f"🤖 Trajectory point {self.trajectory_index}/{len(self.current_trajectory)}: "
                  f"Pos: ({current_pos[0]:.2f}, {current_pos[1]:.2f}), "
                  f"Target: ({target_pos[0]:.2f}, {target_pos[1]:.2f}), "
                  f"Distance: {distance:.2f}m")
        
        # 控制逻辑
        if distance < 0.3:  # 到达当前轨迹点
            self.trajectory_index += 1
            if self.trajectory_index >= len(self.current_trajectory):
                self.state = "REACHED"
                return True
        
        # 计算控制指令
        target_angle = math.atan2(dy, dx)
        angle_diff = target_angle - current_yaw
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        # PID控制器参数（轨迹跟踪）
        linear_speed = min(0.4, distance * 1.5)  # 稍快的线速度
        angular_speed = angle_diff * 2.0  # 角速度控制
        
        # 限制速度
        angular_speed = max(-1.2, min(1.2, angular_speed))
        
        # 如果角度差太大，先转向
        if abs(angle_diff) > math.pi/3:
            linear_speed *= 0.3
        
        # 发送控制指令
        try:
            self.controller.forward(command=[linear_speed, angular_speed])
        except Exception as e:
            print(f"Control error: {e}")
        
        return True
    
    def follow_simple_path(self):
        """跟随简单A*路径"""
        if self.waypoint_index >= len(self.current_path):
            self.state = "REACHED"
            return True
        
        current_pos, current_yaw = self.get_robot_pose()
        target = self.current_path[self.waypoint_index]
        
        # 计算到目标的距离和角度
        dx = target[0] - current_pos[0]
        dy = target[1] - current_pos[1]
        distance = math.sqrt(dx**2 + dy**2)
        target_angle = math.atan2(dy, dx)
        
        # 计算角度差
        angle_diff = target_angle - current_yaw
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        # 调试信息 
        if self.waypoint_index % 5 == 0:
            print(f"🤖 Waypoint {self.waypoint_index}/{len(self.current_path)}: "
                  f"Pos: ({current_pos[0]:.2f}, {current_pos[1]:.2f}), "
                  f"Target: ({target[0]:.2f}, {target[1]:.2f}), "
                  f"Distance: {distance:.2f}m, Angle: {math.degrees(angle_diff):.1f}°")
        
        # 控制逻辑
        if distance < 0.4:  # 到达当前航点
            self.waypoint_index += 1
            print(f"✅ Reached waypoint {self.waypoint_index-1}, moving to next...")
            if self.waypoint_index >= len(self.current_path):
                self.state = "REACHED"
                return True
        
        # 计算控制命令 - 改进的控制策略
        if abs(angle_diff) > 0.15:  # 需要转向
            linear_vel = max(0.1, distance * 0.3)  # 转向时保持前进
            angular_vel = np.sign(angle_diff) * min(abs(angle_diff) * 3.0, 2.5)
        else:  # 前进
            linear_vel = min(distance * 1.2, 0.8)  # 增加线速度
            angular_vel = angle_diff * 1.5
        
        # 确保最小速度，避免机器人停滞
        if linear_vel < 0.08:
            linear_vel = 0.08
        
        # 应用控制 - 确保机器人移动
        dt = 1.0 / 60.0
        
        # 更新角度
        new_yaw = current_yaw + angular_vel * dt
        
        # 更新位置
        new_x = current_pos[0] + linear_vel * math.cos(new_yaw) * dt
        new_y = current_pos[1] + linear_vel * math.sin(new_yaw) * dt
        
        # 应用新位置
        self.set_robot_pose([new_x, new_y, current_pos[2]], new_yaw)
        
        # 每次移动都输出运动状态
        if self.waypoint_index % 10 == 0:
            print(f"🚗 Robot moving: v={linear_vel:.3f}m/s, ω={angular_vel:.3f}rad/s")
        
        return True

def main():
    """主函数"""
    # 创建世界
    world = World()
    
    # 添加地面
    world.scene.add_default_ground_plane()
    
    # 创建交互式机器人
    print("Creating interactive robot and obstacles...")
    robot = InteractiveAvoidanceRobot(world)
    robot.create_obstacles()
    
    # 显示控制说明
    print("\n" + "="*70)
    print("INTERACTIVE A* + SVSDF TRAJECTORY PLANNING CONTROLS:")
    print("="*70)
    print("Arrow Keys / NUMPAD: Move target position")
    print("SPACE: Toggle auto navigation ON/OFF")
    print("R: Force replan current path")
    print("T: Set random target position")
    print("S: Toggle SVSDF/A* planning mode")
    print("V: Visualize current trajectory")
    print("ESC: Exit simulation")
    print("="*70)
    print(f"Robot starting position: {robot.start_pos[:2]}")
    print(f"Target position: {robot.goal_pos[:2]}")
    print(f"Planning mode: {'SVSDF trajectory optimization' if robot.use_svsdf else 'Simple A* planning'}")
    print("Use SPACE to start auto navigation!")
    print("="*70 + "\n")
    
    step_count = 0
    
    # 添加物理回调
    def physics_step(step_size):
        nonlocal step_count
        step_count += 1
        
        # 每300步显示一次状态
        if step_count % 300 == 0:
            status = "AUTO" if robot.auto_navigation else "MANUAL"
            mode = "SVSDF" if robot.use_svsdf else "A*"
            current_pos, _ = robot.get_robot_pose()
            print(f"Step: {step_count}, Mode: {status}/{mode}, State: {robot.state}, "
                  f"Pos: ({current_pos[0]:.2f}, {current_pos[1]:.2f})")
            
            if robot.state == "MOVING":
                if robot.current_trajectory:
                    print(f"   🔵 SVSDF Trajectory: {robot.trajectory_index}/{len(robot.current_trajectory)}")
                elif robot.current_path:
                    print(f"   🟢 A* Path: {robot.waypoint_index}/{len(robot.current_path)}")
        
        robot.update()
    
    world.add_physics_callback("physics_step", physics_step)
    
    # 重置世界并开始仿真
    world.reset()
    
    print("Interactive simulation started!")
    
    # 仿真循环
    start_time = time.time()
    while simulation_app.is_running():
        try:
            world.step(render=True)
            
            # 安全退出机制
            if time.time() - start_time > 3600:  # 1小时后自动退出
                print("Maximum simulation time reached")
                break
                
        except Exception as e:
            print(f"Error during simulation: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # 清理并关闭仿真
    print("Closing simulation...")
    simulation_app.close()

if __name__ == "__main__":
    main()
