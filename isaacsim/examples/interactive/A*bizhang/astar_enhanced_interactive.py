#!/usr/bin/env python3
"""
A*算法避障项目 - 增强交互式版本
支持实时目标选择、拖拽功能和UI界面
"""

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import carb
import omni
import omni.appwindow
import omni.ui as ui
import omni.usd
import omni.timeline
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

class InteractiveUI:
    """交互式用户界面"""
    
    def __init__(self, robot):
        self.robot = robot
        self.window = None
        self.ui_elements = {}
        self.build_ui()
    
    def build_ui(self):
        """构建用户界面"""
        try:
            self.window = ui.Window("A* Interactive Controls", width=350, height=500)
            
            with self.window.frame:
                with ui.VStack(spacing=10):
                    # 标题
                    ui.Label("A* 交互式路径规划", height=30, style={"font_size": 18})
                    ui.Separator()
                    
                    # 控制按钮
                    with ui.CollapsableFrame("控制", collapsed=False):
                        with ui.VStack(spacing=5):
                            self.ui_elements["start_btn"] = ui.Button(
                                "开始导航", 
                                height=30,
                                clicked_fn=self._on_start_navigation
                            )
                            
                            self.ui_elements["stop_btn"] = ui.Button(
                                "停止导航", 
                                height=30,
                                clicked_fn=self._on_stop_navigation
                            )
                            
                            self.ui_elements["replan_btn"] = ui.Button(
                                "重新规划", 
                                height=30,
                                clicked_fn=self._on_replan
                            )
                            
                            self.ui_elements["random_target_btn"] = ui.Button(
                                "随机目标", 
                                height=30,
                                clicked_fn=self._on_random_target
                            )
                    
                    # 目标位置控制
                    with ui.CollapsableFrame("目标位置", collapsed=False):
                        with ui.VStack(spacing=5):
                            ui.Label("目标 X 坐标:")
                            self.ui_elements["target_x"] = ui.FloatSlider(
                                min=-12, max=12, step=0.5,
                                value=self.robot.goal_pos[0]
                            )
                            self.ui_elements["target_x"].model.add_value_changed_fn(self._on_target_x_changed)
                            
                            ui.Label("目标 Y 坐标:")
                            self.ui_elements["target_y"] = ui.FloatSlider(
                                min=-12, max=12, step=0.5,
                                value=self.robot.goal_pos[1]
                            )
                            self.ui_elements["target_y"].model.add_value_changed_fn(self._on_target_y_changed)
                    
                    # 状态信息
                    with ui.CollapsableFrame("状态信息", collapsed=False):
                        with ui.VStack(spacing=5):
                            self.ui_elements["status_label"] = ui.Label("状态: IDLE")
                            self.ui_elements["mode_label"] = ui.Label("模式: MANUAL")
                            self.ui_elements["robot_pos_label"] = ui.Label("机器人位置: (-10, -10)")
                            self.ui_elements["target_pos_label"] = ui.Label(f"目标位置: ({self.robot.goal_pos[0]:.1f}, {self.robot.goal_pos[1]:.1f})")
                    
                    # 键盘快捷键说明
                    with ui.CollapsableFrame("键盘快捷键", collapsed=True):
                        with ui.VStack(spacing=2):
                            ui.Label("方向键: 移动目标")
                            ui.Label("空格键: 开始/停止导航")
                            ui.Label("R键: 重新规划路径")
                            ui.Label("T键: 随机目标位置")
                            ui.Label("ESC键: 退出程序")
        except Exception as e:
            print(f"Failed to create UI: {e}")
    
    def update_ui(self):
        """更新UI显示"""
        try:
            if self.ui_elements.get("status_label"):
                self.ui_elements["status_label"].text = f"状态: {self.robot.state}"
            
            if self.ui_elements.get("mode_label"):
                mode = "AUTO" if self.robot.auto_navigation else "MANUAL"
                self.ui_elements["mode_label"].text = f"模式: {mode}"
            
            if self.ui_elements.get("robot_pos_label"):
                pos = self.robot.current_position
                self.ui_elements["robot_pos_label"].text = f"机器人位置: ({pos[0]:.1f}, {pos[1]:.1f})"
            
            if self.ui_elements.get("target_pos_label"):
                self.ui_elements["target_pos_label"].text = f"目标位置: ({self.robot.goal_pos[0]:.1f}, {self.robot.goal_pos[1]:.1f})"
        except Exception as e:
            print(f"Failed to update UI: {e}")
    
    def _on_start_navigation(self):
        """开始导航按钮回调"""
        self.robot.auto_navigation = True
        self.robot.state = "PLANNING"
        print("Auto navigation started from UI")
    
    def _on_stop_navigation(self):
        """停止导航按钮回调"""
        self.robot.auto_navigation = False
        self.robot.state = "IDLE"
        print("Auto navigation stopped from UI")
    
    def _on_replan(self):
        """重新规划按钮回调"""
        if self.robot.auto_navigation:
            self.robot.state = "PLANNING"
            print("Replanning requested from UI")
    
    def _on_random_target(self):
        """随机目标按钮回调"""
        self.robot.set_random_target()
        # 更新滑块值
        if self.ui_elements.get("target_x"):
            self.ui_elements["target_x"].model.set_value(self.robot.goal_pos[0])
        if self.ui_elements.get("target_y"):
            self.ui_elements["target_y"].model.set_value(self.robot.goal_pos[1])
        print("Random target set from UI")
    
    def _on_target_x_changed(self, model):
        """目标X坐标滑块回调"""
        self.robot.goal_pos[0] = model.get_value_as_float()
        self.robot.goal_changed = True
        self.robot.update_target_cube_position()
    
    def _on_target_y_changed(self, model):
        """目标Y坐标滑块回调"""
        self.robot.goal_pos[1] = model.get_value_as_float()
        self.robot.goal_changed = True
        self.robot.update_target_cube_position()

class InteractiveAvoidanceRobot:
    """增强交互式避障机器人类"""
    
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
        
        # 性能统计
        self.path_planning_time = 0.0
        self.total_distance_traveled = 0.0
        self.last_position = np.array(self.start_pos)
        
        # 输入处理
        self._appwindow = None
        self._input = None
        self._keyboard = None
        self._sub_keyboard = None
        
        # 键盘映射
        self._input_keyboard_mapping = {
            # 目标移动
            "NUMPAD_8": [0, 1.0],    # 向前移动目标
            "UP": [0, 1.0],
            "NUMPAD_2": [0, -1.0],   # 向后移动目标
            "DOWN": [0, -1.0],
            "NUMPAD_4": [-1.0, 0],   # 向左移动目标
            "LEFT": [-1.0, 0],
            "NUMPAD_6": [1.0, 0],    # 向右移动目标
            "RIGHT": [1.0, 0],
            # 控制键
            "SPACE": "toggle_auto",   # 开始/停止自动导航
            "R": "replan",           # 重新规划路径
            "T": "new_target",       # 设置新目标
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
        """更新目标立方体的位置"""
        if self.target_cube:
            try:
                self.target_cube.set_world_pose(
                    position=np.array([self.goal_pos[0], self.goal_pos[1], 0.2])
                )
            except Exception as e:
                print(f"Failed to update target cube position: {e}")
    
    def toggle_auto_navigation(self):
        """切换自动导航模式"""
        self.auto_navigation = not self.auto_navigation
        if self.auto_navigation:
            print("Auto navigation ENABLED - Robot will follow the target")
            self.state = "PLANNING"
        else:
            print("Auto navigation DISABLED - Use controls to move target")
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
        
        # 创建可交互的目标立方体
        self.target_cube = self.world.scene.add(
            DynamicCuboid(
                prim_path="/World/target_cube",
                name="target_cube",
                position=np.array([self.goal_pos[0], self.goal_pos[1], 0.2]),
                scale=np.array([0.6, 0.6, 0.6]),
                color=np.array([1.0, 1.0, 0.0])  # 黄色
            )
        )
        print("Created interactive target cube!")
    
    def plan_path(self):
        """规划路径"""
        start_time = time.time()
        
        current_pos, _ = self.get_robot_pose()
        print(f"Planning path from {current_pos[:2]} to {self.goal_pos[:2]}")
        
        self.current_path = self.planner.find_path(
            [current_pos[0], current_pos[1]], 
            [self.goal_pos[0], self.goal_pos[1]]
        )
        
        self.path_planning_time = time.time() - start_time
        
        if not self.current_path:
            print("No path found!")
            self.state = "IDLE"
            return False
        
        self.waypoint_index = 0
        self.visualize_path()
        print(f"Path planned with {len(self.current_path)} waypoints in {self.path_planning_time:.3f}s")
        return True
    
    def visualize_path(self):
        """可视化路径"""
        # 清除旧路径
        for i in range(300):
            prim_path = f"/World/waypoint_{i}"
            prim = self.world.stage.GetPrimAtPath(prim_path)
            if prim.IsValid():
                self.world.stage.RemovePrim(prim_path)
        
        # 清除旧的标记
        for marker in ["/World/goal_marker", "/World/start_marker"]:
            marker_prim = self.world.stage.GetPrimAtPath(marker)
            if marker_prim.IsValid():
                self.world.stage.RemovePrim(marker)
        
        # 添加起始点标记
        self.world.scene.add(
            DynamicCuboid(
                prim_path="/World/start_marker",
                name="start_marker",
                position=np.array([self.start_pos[0], self.start_pos[1], 0.2]),
                scale=np.array([0.3, 0.3, 0.3]),
                color=np.array([0, 1, 1])  # 青色
            )
        )
        
        # 添加路径点
        for i, point in enumerate(self.current_path):
            if i % 3 == 0:  # 每隔3个点显示一个
                self.world.scene.add(
                    DynamicCuboid(
                        prim_path=f"/World/waypoint_{i}",
                        name=f"waypoint_{i}",
                        position=np.array([point[0], point[1], 0.1]),
                        scale=np.array([0.15, 0.15, 0.15]),
                        color=np.array([0, 1, 0])  # 绿色
                    )
                )
    
    def update(self):
        """更新机器人状态"""
        # 计算移动距离
        distance_moved = np.linalg.norm(self.current_position[:2] - self.last_position[:2])
        self.total_distance_traveled += distance_moved
        self.last_position = self.current_position.copy()
        
        # 检查目标是否改变
        if self.goal_changed and self.auto_navigation:
            print("Target changed - replanning...")
            self.state = "PLANNING"
            self.goal_changed = False
        
        if self.state == "IDLE":
            return True
        
        elif self.state == "PLANNING":
            if self.plan_path():
                self.state = "MOVING"
                print("Starting navigation...")
            else:
                print("Failed to find path!")
                self.state = "IDLE"
            return True
        
        elif self.state == "MOVING":
            return self.follow_path()
        
        elif self.state == "REACHED":
            print(f"Target reached! Total distance: {self.total_distance_traveled:.2f}m")
            self.state = "IDLE"
            return True
        
        return True
    
    def follow_path(self):
        """跟随路径"""
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
        
        # 控制逻辑
        if distance < 0.4:  # 到达当前航点
            self.waypoint_index += 1
            if self.waypoint_index >= len(self.current_path):
                self.state = "REACHED"
                return True
        
        # 计算控制命令
        if abs(angle_diff) > 0.3:  # 需要转向
            linear_vel = 0.1
            angular_vel = np.sign(angle_diff) * min(abs(angle_diff) * 1.0, 1.2)
        else:  # 前进
            linear_vel = min(distance * 0.8, 0.5)
            angular_vel = angle_diff * 0.8
        
        # 应用控制
        command = np.array([linear_vel, angular_vel])
        action = self.controller.forward(command)
        
        # 使用简单的运动学模型更新位置
        dt = 1.0 / 60.0
        
        # 更新角度
        new_yaw = current_yaw + angular_vel * dt
        
        # 更新位置
        new_x = current_pos[0] + linear_vel * math.cos(new_yaw) * dt
        new_y = current_pos[1] + linear_vel * math.sin(new_yaw) * dt
        
        # 应用新位置
        self.set_robot_pose([new_x, new_y, current_pos[2]], new_yaw)
        
        return True

def main():
    """主函数"""
    # 创建世界
    world = World()
    
    # 添加地面
    world.scene.add_default_ground_plane()
    
    # 创建交互式机器人
    print("Creating enhanced interactive robot and obstacles...")
    robot = InteractiveAvoidanceRobot(world)
    robot.create_obstacles()
    
    # 创建UI界面
    ui_interface = InteractiveUI(robot)
    
    # 显示控制说明
    print("\n" + "="*70)
    print("ENHANCED INTERACTIVE A* PATHFINDING CONTROLS:")
    print("="*70)
    print("GUI Controls: Use the control window on the right")
    print("Arrow Keys / NUMPAD: Move target position")
    print("SPACE: Toggle auto navigation ON/OFF")
    print("R: Force replan current path")
    print("T: Set random target position")
    print("ESC: Exit simulation")
    print("Drag Support: Select target cube in Stage tree and drag it!")
    print("="*70)
    print(f"Robot starting position: {robot.start_pos[:2]}")
    print(f"Target position: {robot.goal_pos[:2]}")
    print("Use UI controls or SPACE to start auto navigation!")
    print("="*70 + "\n")
    
    step_count = 0
    ui_update_counter = 0
    
    # 添加物理回调
    def physics_step(step_size):
        nonlocal step_count, ui_update_counter
        step_count += 1
        ui_update_counter += 1
        
        # 每300步显示一次状态
        if step_count % 300 == 0:
            status = "AUTO" if robot.auto_navigation else "MANUAL"
            print(f"Step: {step_count}, Mode: {status}, State: {robot.state}, Distance: {robot.total_distance_traveled:.2f}m")
        
        # 每30步更新一次UI（约0.5秒）
        if ui_update_counter % 30 == 0:
            ui_interface.update_ui()
        
        robot.update()
    
    world.add_physics_callback("physics_step", physics_step)
    
    # 重置世界并开始仿真
    world.reset()
    
    print("Enhanced interactive simulation started!")
    
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
    print("Closing enhanced simulation...")
    simulation_app.close()

if __name__ == "__main__":
    main()
