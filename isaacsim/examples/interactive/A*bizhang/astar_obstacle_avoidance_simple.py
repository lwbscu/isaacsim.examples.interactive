#!/usr/bin/env python3
"""
A*算法避障项目 - 调试版
修复起始位置在障碍物内的问题
"""

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import carb
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
import omni.usd
import isaacsim.core.utils.prims as prim_utils

# 设置资源路径
asset_root = "/home/lwb/isaacsim_assets/Assets/Isaac/4.5"
carb.settings.get_settings().set("/persistent/isaac/asset_root/default", asset_root)

class SimpleAStarPlanner:
    """简化版A*路径规划器"""
    
    def __init__(self, grid_size=150, cell_size=0.2):  # 增大网格，减小单元尺寸
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
        # 减小膨胀半径
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
        print(f"Start world: {start_world}, Goal world: {goal_world}")
        
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
        return []  # 没有找到路径

class AvoidanceRobot:
    """避障机器人类"""
    
    def __init__(self, world):
        self.world = world
        
        # 加载create_3机器人
        robot_usd_path = "/home/lwb/isaacsim_assets/Assets/Isaac/4.5/Isaac/Robots/iRobot/create_3.usd"
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
        # 调整起始和目标位置，确保不在障碍物内
        self.start_pos = [-10, -10, 0.1]
        self.goal_pos = [10, 10, 0.1]
        self.state = "IDLE"  # 初始状态为IDLE
        
        # 运动状态
        self.current_position = np.array(self.start_pos)
        self.current_orientation = 0.0
        
        # 运行时间控制
        self.idle_start_time = None
        self.reached_time = None
        self.run_duration = 30  # 程序运行30秒
        
        # 设置初始位置
        self.set_robot_pose(self.start_pos, 0.0)
    
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
        """创建障碍物 - 优化布局确保有可行路径"""
        # 减小中心障碍物，增大通道
        obstacles = [
            {"pos": [0, 0, 0.5], "scale": [2, 2, 1]},      # 中心方块，变小
            {"pos": [5, 0, 0.5], "scale": [1, 6, 1]},      # 右侧垂直墙，缩短
            {"pos": [-5, 0, 0.5], "scale": [1, 6, 1]},     # 左侧垂直墙，缩短
            {"pos": [0, 5, 0.5], "scale": [6, 1, 1]},      # 上方水平墙，缩短
            {"pos": [0, -5, 0.5], "scale": [6, 1, 1]},     # 下方水平墙，缩短
        ]
        
        # 创建障碍物，使用 numpy 数组作为颜色
        for i, obs in enumerate(obstacles):
            obstacle = self.world.scene.add(
                FixedCuboid(
                    prim_path=f"/World/obstacle_{i}",
                    name=f"obstacle_{i}",
                    position=np.array(obs["pos"]),
                    scale=np.array(obs["scale"]),
                    color=np.array([0.8, 0.2, 0.2])  # 红色
                )
            )
            
            # 添加到规划器
            self.planner.add_obstacle(obs["pos"], obs["scale"])
            print(f"Created obstacle {i} at {obs['pos']}")
        
        # 减小边界墙，确保起始位置不被覆盖
        boundary_walls = [
            {"pos": [0, 13, 0.5], "scale": [26, 1, 1]},   # 上边界
            {"pos": [0, -13, 0.5], "scale": [26, 1, 1]},  # 下边界
            {"pos": [13, 0, 0.5], "scale": [1, 26, 1]},   # 右边界
            {"pos": [-13, 0, 0.5], "scale": [1, 26, 1]},  # 左边界
        ]
        
        for i, wall in enumerate(boundary_walls):
            boundary = self.world.scene.add(
                FixedCuboid(
                    prim_path=f"/World/boundary_{i}",
                    name=f"boundary_{i}",
                    position=np.array(wall["pos"]),
                    scale=np.array(wall["scale"]),
                    color=np.array([0.5, 0.5, 0.5])  # 灰色
                )
            )
            self.planner.add_obstacle(wall["pos"], wall["scale"])
        
        # 打印网格调试信息
        print("\nGrid debug info:")
        print(f"Grid size: {self.planner.grid_size}x{self.planner.grid_size}")
        print(f"Cell size: {self.planner.cell_size}")
        start_grid = self.planner.world_to_grid(self.start_pos)
        goal_grid = self.planner.world_to_grid(self.goal_pos)
        print(f"Start position {self.start_pos} -> grid {start_grid}")
        print(f"Goal position {self.goal_pos} -> grid {goal_grid}")
        print(f"Start grid value: {self.planner.grid[start_grid[1], start_grid[0]]}")
        print(f"Goal grid value: {self.planner.grid[goal_grid[1], goal_grid[0]]}")
    
    def plan_path(self):
        """规划路径"""
        current_pos, _ = self.get_robot_pose()
        print(f"Planning path from {current_pos} to {self.goal_pos}")
        
        self.current_path = self.planner.find_path(
            [current_pos[0], current_pos[1]], 
            [self.goal_pos[0], self.goal_pos[1]]
        )
        
        if not self.current_path:
            print("No path found! Trying alternate goal position...")
            # 尝试一个更容易到达的目标位置
            alternate_goals = [
                [8, 8, 0.1],
                [9, 9, 0.1],
                [7, 7, 0.1]
            ]
            
            for alt_goal in alternate_goals:
                self.current_path = self.planner.find_path(
                    [current_pos[0], current_pos[1]], 
                    [alt_goal[0], alt_goal[1]]
                )
                if self.current_path:
                    print(f"Found path to alternate goal {alt_goal}")
                    self.goal_pos = alt_goal
                    break
            
            if not self.current_path:
                print("Still no path found!")
                self.state = "IDLE"
                return False
        
        self.waypoint_index = 0
        self.visualize_path()
        print(f"Path planned with {len(self.current_path)} waypoints")
        return True
    
    def visualize_path(self):
        """可视化路径"""
        # 清除旧路径
        for i in range(300):  # 增加清除范围
            prim_path = f"/World/waypoint_{i}"
            prim = self.world.stage.GetPrimAtPath(prim_path)
            if prim.IsValid():
                self.world.stage.RemovePrim(prim_path)
        
        # 清除旧的目标点
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
        
        # 添加路径点，使用numpy数组作为位置和颜色
        for i, point in enumerate(self.current_path):
            if i % 2 == 0:  # 每隔2个点显示一个
                self.world.scene.add(
                    DynamicCuboid(
                        prim_path=f"/World/waypoint_{i}",
                        name=f"waypoint_{i}",
                        position=np.array([point[0], point[1], 0.1]),
                        scale=np.array([0.15, 0.15, 0.15]),
                        color=np.array([0, 1, 0])  # 绿色
                    )
                )
        
        # 添加目标点marker
        self.world.scene.add(
            DynamicCuboid(
                prim_path="/World/goal_marker",
                name="goal_marker",
                position=np.array([self.goal_pos[0], self.goal_pos[1], 0.2]),
                scale=np.array([0.3, 0.3, 0.3]),
                color=np.array([0, 0, 1])  # 蓝色
            )
        )
    
    def update(self):
        """更新机器人状态"""
        current_time = time.time()
        
        if self.state == "IDLE":
            if self.idle_start_time is None:
                self.idle_start_time = current_time
                print("Starting navigation in 3 seconds...")
            
            # 等待3秒后开始规划
            if current_time - self.idle_start_time > 3.0:
                print("Starting path planning...")
                self.state = "PLANNING"
            return True
        
        elif self.state == "PLANNING":
            if self.plan_path():
                self.state = "MOVING"
                print("Starting navigation...")
            else:
                print("Failed to find path. Will retry in 5 seconds...")
                self.idle_start_time = current_time
                self.state = "IDLE"
            return True
        
        elif self.state == "MOVING":
            return self.follow_path()
        
        elif self.state == "REACHED":
            if self.reached_time is None:
                self.reached_time = current_time
                print("Target reached! Simulation will continue for 10 more seconds...")
            
            # 到达目标后再运行10秒
            if current_time - self.reached_time > 10.0:
                print("Simulation ending...")
                return False
            return True
        
        elif self.state == "FAILED":
            print("Navigation failed! Simulation will continue to show the scene...")
            # 即使失败也继续运行，不要关闭仿真
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
            print(f"Reached waypoint {self.waypoint_index}/{len(self.current_path)}")
            if self.waypoint_index >= len(self.current_path):
                self.state = "REACHED"
                return True
        
        # 计算控制命令
        if abs(angle_diff) > 0.3:  # 需要转向
            linear_vel = 0.1
            angular_vel = np.sign(angle_diff) * min(abs(angle_diff) * 1.0, 1.2)
        else:  # 前进
            linear_vel = min(distance * 0.8, 0.4)
            angular_vel = angle_diff * 0.8
        
        # 应用控制
        command = np.array([linear_vel, angular_vel])
        action = self.controller.forward(command)
        
        # 使用简单的运动学模型更新位置
        dt = 1.0 / 60.0  # 假设60Hz更新频率
        
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
    
    # 创建机器人
    print("Creating robot and obstacles...")
    robot = AvoidanceRobot(world)
    robot.create_obstacles()
    
    # 显示信息
    print(f"Robot starting position: {robot.start_pos}")
    print(f"Target position: {robot.goal_pos}")
    print("Simulation will run for about 30 seconds")
    
    step_count = 0
    
    # 添加物理回调
    def physics_step(step_size):
        nonlocal step_count
        step_count += 1
        
        # 每100步显示一次状态
        if step_count % 100 == 0:
            print(f"Step: {step_count}, Robot state: {robot.state}")
        
        if not robot.update():
            # 导航完成，等待2秒后关闭
            print("Simulation completed!")
            time.sleep(2)
            simulation_app.close()
    
    world.add_physics_callback("physics_step", physics_step)
    
    # 重置世界并开始仿真
    world.reset()
    
    print("Simulation started - press ESC to exit anytime")
    
    # 仿真循环
    start_time = time.time()
    while simulation_app.is_running():
        try:
            world.step(render=True)

            # 安全退出机制 - 运行最多180秒
            if time.time() - start_time > 180:
                print("Maximum simulation time reached (60 seconds)")
                break
                
        except Exception as e:
            print(f"Error during simulation: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # 关闭仿真
    print("Closing simulation...")
    simulation_app.close()

if __name__ == "__main__":
    main()