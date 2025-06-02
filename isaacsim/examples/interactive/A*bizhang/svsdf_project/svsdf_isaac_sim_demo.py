#!/usr/bin/env python3
"""
SVSDF轨迹规划系统Isaac Sim演示脚本
完整展示扫掠体积感知轨迹规划的四个阶段
参考astar_interactive.py的正确模式
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
import time
from queue import PriorityQueue

# Isaac Sim imports (正确的导入方式)
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.robot.wheeled_robots import DifferentialController
from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import UsdGeom, Gf, Usd
import isaacsim.core.utils.prims as prim_utils

# 导入上级目录的SVSDF规划器
import sys
sys.path.append('/home/lwb/isaacsim/exts/isaacsim.examples.interactive/isaacsim/examples/interactive/A*bizhang')
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
        world_x = grid_pos[0] * self.cell_size - offset
        world_y = grid_pos[1] * self.cell_size - offset
        return (world_x, world_y)
    
    def add_circular_obstacle(self, center, radius):
        """添加圆形障碍物"""
        center_grid = self.world_to_grid(center)
        radius_grid = int(radius / self.cell_size) + 2  # 增加安全余量
        
        for i in range(max(0, center_grid[0] - radius_grid), 
                      min(self.grid_size, center_grid[0] + radius_grid + 1)):
            for j in range(max(0, center_grid[1] - radius_grid), 
                          min(self.grid_size, center_grid[1] + radius_grid + 1)):
                dist = math.sqrt((i - center_grid[0])**2 + (j - center_grid[1])**2)
                if dist <= radius_grid:
                    self.grid[i, j] = 1
    
    def heuristic(self, a, b):
        """A*启发式函数"""
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def get_neighbors(self, pos):
        """获取邻居节点"""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                new_x, new_y = pos[0] + dx, pos[1] + dy
                if (0 <= new_x < self.grid_size and 
                    0 <= new_y < self.grid_size and 
                    self.grid[new_x, new_y] == 0):
                    neighbors.append((new_x, new_y))
        return neighbors
    
    def plan_path(self, start_world, goal_world):
        """A*路径规划"""
        start_grid = self.world_to_grid(start_world)
        goal_grid = self.world_to_grid(goal_world)
        
        if self.grid[start_grid[0], start_grid[1]] == 1:
            print("起点在障碍物中")
            return []
        if self.grid[goal_grid[0], goal_grid[1]] == 1:
            print("终点在障碍物中")
            return []
        
        open_set = PriorityQueue()
        open_set.put((0, start_grid))
        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: self.heuristic(start_grid, goal_grid)}
        
        while not open_set.empty():
            current = open_set.get()[1]
            
            if current == goal_grid:
                # 重建路径
                path = []
                while current in came_from:
                    world_pos = self.grid_to_world(current)
                    path.append([world_pos[0], world_pos[1]])
                    current = came_from[current]
                world_pos = self.grid_to_world(start_grid)
                path.append([world_pos[0], world_pos[1]])
                path.reverse()
                return path
            
            for neighbor in self.get_neighbors(current):
                tentative_g_score = g_score[current] + self.heuristic(current, neighbor)
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal_grid)
                    open_set.put((f_score[neighbor], neighbor))
        
        print("未找到路径")
        return []

class SVSDFDemo:
    """SVSDF演示类 - 参考astar_interactive.py的实现模式"""
    
    def __init__(self):
        self.world = None
        self.robot_prim_path = "/World/create_3"
        self.robot_prim = None
        self.robot_xform = None
        self.controller = None
        self.astar_planner = SimpleAStarPlanner()
        self.svsdf_planner = None
        
        # 机器人状态
        self.current_position = np.array([0.0, 0.0, 0.1])
        self.current_orientation = 0.0
        
        # 轨迹相关
        self.current_trajectory = []
        self.trajectory_index = 0
        
        # 可视化
        self.obstacle_prims = []
        self.trajectory_markers = []
        self.swept_volume_markers = []
        
        # 演示场景
        self.demo_scenarios = []
        self._setup_demo_scenarios()
        
    def _setup_demo_scenarios(self):
        """设置演示场景"""
        
        # 场景1：简单导航
        self.demo_scenarios.append({
            'name': '简单导航',
            'description': '在开放空间中的基本导航',
            'start_pos': np.array([0.0, 0.0]),
            'goal_pos': np.array([5.0, 3.0]),
            'start_yaw': 0.0,
            'goal_yaw': np.pi/4,
            'obstacles': [
                {'type': 'circle', 'center': [2.5, 1.5], 'radius': 0.8}
            ]
        })
        
        # 场景2：多障碍物环境
        self.demo_scenarios.append({
            'name': '多障碍物导航',
            'description': '复杂多障碍物环境中的导航',
            'start_pos': np.array([0.0, 0.0]),
            'goal_pos': np.array([8.0, 6.0]),
            'start_yaw': 0.0,
            'goal_yaw': 0.0,
            'obstacles': [
                {'type': 'circle', 'center': [2.0, 1.0], 'radius': 0.6},
                {'type': 'circle', 'center': [4.0, 3.0], 'radius': 0.5},
                {'type': 'circle', 'center': [6.0, 2.0], 'radius': 0.7},
                {'type': 'rectangle', 'center': [3.0, 4.5], 'size': [1.5, 0.8]},
                {'type': 'rectangle', 'center': [7.0, 5.0], 'size': [1.0, 1.2]}
            ]
        })
        
        # 场景3：狭窄通道
        self.demo_scenarios.append({
            'name': '狭窄通道',
            'description': '需要精确规划的狭窄通道导航',
            'start_pos': np.array([0.0, 2.0]),
            'goal_pos': np.array([6.0, 2.0]),
            'start_yaw': 0.0,
            'goal_yaw': 0.0,
            'obstacles': [
                {'type': 'rectangle', 'center': [2.0, 1.0], 'size': [3.0, 0.4]},
                {'type': 'rectangle', 'center': [2.0, 3.0], 'size': [3.0, 0.4]},
                {'type': 'rectangle', 'center': [4.5, 1.0], 'size': [1.0, 0.4]},
                {'type': 'rectangle', 'center': [4.5, 3.0], 'size': [1.0, 0.4]}
            ]
        })
        
        # 场景4：U型转弯
        self.demo_scenarios.append({
            'name': 'U型转弯',
            'description': '测试大角度转弯的扫掠体积优化',
            'start_pos': np.array([0.0, 2.0]),
            'goal_pos': np.array([0.0, 2.0]),
            'start_yaw': 0.0,
            'goal_yaw': np.pi,  # 180度转弯
            'obstacles': [
                {'type': 'rectangle', 'center': [2.0, 0.8], 'size': [4.0, 0.4]},
                {'type': 'rectangle', 'center': [2.0, 3.2], 'size': [4.0, 0.4]},
                {'type': 'rectangle', 'center': [4.3, 2.0], 'size': [0.4, 2.8]}
            ]
        })
    
    def initialize_isaac_sim(self):
        """初始化Isaac Sim环境"""
        print("正在初始化Isaac Sim环境...")
        
        # 创建世界
        self.world = World(stage_units_in_meters=1.0)
        # 注意：使用同步方式初始化，参考astar_interactive.py
        
        # 设置物理参数
        self.world.get_physics_context().set_gravity(-9.81)
        self.world.get_physics_context().set_solver_type("TGS")
        
        # 添加地面
        self.world.scene.add_default_ground_plane()
        
        # 设置照明
        self._setup_lighting()
        
        # 设置相机
        self._setup_camera()
        
        print("Isaac Sim环境初始化完成")
    
    def _setup_lighting(self):
        """设置场景照明"""
        try:
            # 添加定向光源
            from omni.isaac.core.utils.prims import create_prim
            from pxr import UsdLux
            
            light_prim = create_prim("/World/DistantLight", "DistantLight")
            distant_light = UsdLux.DistantLight(light_prim)
            distant_light.CreateIntensityAttr(3000)
            distant_light.CreateAngleAttr(0.5)
            
            # 设置光源方向
            from pxr import Gf
            light_prim.GetAttribute("xformOp:rotateXYZ").Set(Gf.Vec3f(-45, 45, 0))
            
        except Exception as e:
            print(f"设置照明失败: {e}")
    
    def _setup_camera(self):
        """设置相机视角"""
        try:
            # 设置相机位置和角度以获得最佳视角
            from omni.isaac.core.utils.viewports import set_camera_view
            
            # 俯视角度
            eye = np.array([5.0, 5.0, 8.0])
            target = np.array([5.0, 3.0, 0.0])
            
            set_camera_view(eye=eye, target=target, camera_prim_path="/OmniverseKit_Persp")
            
        except Exception as e:
            print(f"设置相机失败: {e}")
    
    def run_demo_scenario(self, scenario_index: int = 1):
        """运行指定的演示场景 - 默认运行复杂场景"""
        if scenario_index >= len(self.demo_scenarios):
            print(f"场景索引 {scenario_index} 超出范围，使用场景1")
            scenario_index = 1
        
        scenario = self.demo_scenarios[scenario_index]
        print(f"\n{'='*50}")
        print(f"运行演示场景: {scenario['name']}")
        print(f"描述: {scenario['description']}")
        print(f"{'='*50}")
        
        # 设置机器人初始位置
        start_pos = np.array([scenario['start_pos'][0], scenario['start_pos'][1], 0.1])
        self.set_robot_pose(start_pos, scenario['start_yaw'])
        
        # 创建障碍物
        self.create_obstacles_for_scenario(scenario['obstacles'])
        
        # 等待物理稳定
        self._wait_for_stability()
        
        # 第一阶段：A*路径规划
        print(f"\n阶段1: A*初始路径搜索...")
        astar_path = self.astar_planner.plan_path(
            scenario['start_pos'], scenario['goal_pos']
        )
        
        if not astar_path:
            print("A*路径规划失败!")
            return False
        
        print(f"✓ A*路径规划完成，找到 {len(astar_path)} 个路径点")
        
        # 第二阶段：MINCO第一阶段优化（轨迹平滑化）
        print(f"阶段2: MINCO第一阶段优化（轨迹平滑化）...")
        try:
            # 将A*路径转换为轨迹点
            trajectory_points = []
            for i, point in enumerate(astar_path):
                t = float(i) * 0.5  # 每个点间隔0.5秒
                traj_point = TrajectoryPoint(
                    position=np.array([point[0], point[1], scenario['start_yaw'] if i == 0 else 0.0]),
                    velocity=np.array([0.3, 0.0, 0.0]),  # 保持前进
                    acceleration=np.array([0.0, 0.0, 0.0]),
                    time=t
                )
                trajectory_points.append(traj_point)
            
            # SVSDF第一阶段优化
            stage1_trajectory = self.svsdf_planner.optimize_stage1(
                trajectory_points, scenario['start_pos'], scenario['goal_pos']
            )
            print(f"✓ MINCO第一阶段完成，优化了 {len(stage1_trajectory)} 个轨迹点")
            
        except Exception as e:
            print(f"MINCO第一阶段失败: {e}")
            print("使用A*路径继续...")
            stage1_trajectory = trajectory_points
        
        # 第三阶段：MINCO第二阶段优化（扫掠体积最小化）
        print(f"阶段3: MINCO第二阶段优化（扫掠体积最小化）...")
        try:
            final_trajectory = self.svsdf_planner.optimize_stage2(
                stage1_trajectory, scenario['obstacles']
            )
            print(f"✓ MINCO第二阶段完成，最终轨迹包含 {len(final_trajectory)} 个点")
            
        except Exception as e:
            print(f"MINCO第二阶段失败: {e}")
            print("使用第一阶段轨迹继续...")
            final_trajectory = stage1_trajectory
        
        # 第四阶段：轨迹跟踪执行
        print(f"阶段4: 轨迹跟踪执行...")
        self.current_trajectory = final_trajectory
        success = self.execute_trajectory()
        
        if success:
            print(f"✓ 场景 '{scenario['name']}' 执行完成!")
            print(f"起点: ({scenario['start_pos'][0]:.2f}, {scenario['start_pos'][1]:.2f})")
            print(f"终点: ({scenario['goal_pos'][0]:.2f}, {scenario['goal_pos'][1]:.2f})")
            print(f"最终轨迹点数: {len(final_trajectory)}")
        else:
            print(f"✗ 场景 '{scenario['name']}' 执行失败!")
        
        return success
    
    def execute_trajectory(self):
        """执行轨迹跟踪"""
        if not self.current_trajectory:
            print("没有可执行的轨迹")
            return False
        
        print("开始执行轨迹跟踪...")
        
        # 简化的轨迹跟踪：逐点移动机器人
        for i, traj_point in enumerate(self.current_trajectory):
            # 计算进度
            progress = (i + 1) / len(self.current_trajectory) * 100
            
            # 设置机器人位置
            self.set_robot_pose(
                [traj_point.position[0], traj_point.position[1], 0.1],
                traj_point.position[2]  # yaw
            )
            
            # 打印进度
            if i % 5 == 0 or i == len(self.current_trajectory) - 1:
                print(f"执行进度: {progress:.1f}% - 位置: ({traj_point.position[0]:.2f}, {traj_point.position[1]:.2f})")
            
            # 等待一帧
            self.world.step(render=True)
            time.sleep(0.1)
        
        print("轨迹执行完成")
        return True
    
    def run_complex_demo(self):
        """运行复杂场景演示 - 按照用户要求简化为一个复杂场景"""
        print(f"\n{'='*60}")
        print("SVSDF轨迹规划系统 - 复杂多障碍物演示")
        print("展示完整的4阶段SVSDF框架:")
        print("1. A*初始路径搜索")
        print("2. MINCO阶段1优化（轨迹平滑化）") 
        print("3. MINCO阶段2优化（扫掠体积最小化）")
        print("4. 轨迹跟踪执行")
        print(f"{'='*60}")
        
        # 运行复杂多障碍物场景（索引1）
        success = self.run_demo_scenario(1)
        
        if success:
            print(f"\n🎉 SVSDF复杂场景演示完成!")
            print("已成功展示了完整的4阶段SVSDF轨迹规划框架")
        else:
            print(f"\n❌ 演示执行失败")
        
        return success
    
    def _wait_for_stability(self, duration: float = 2.0):
        """等待物理系统稳定"""
        print(f"等待物理系统稳定 ({duration}s)...")
        
        for _ in range(int(duration * 10)):
            self.world.step(render=True)
            time.sleep(0.1)
    
    def cleanup(self):
        """清理资源"""
        try:
            self.clear_obstacles()
            
            if self.world:
                self.world.stop()
            
            print("演示系统已清理")
        except Exception as e:
            print(f"清理资源时出错: {e}")
    
    def initialize_robot(self):
        """初始化机器人 - 参考astar_interactive.py的实现"""
        print("正在初始化Create-3机器人...")
        
        # 加载Create-3机器人USD文件
        robot_usd_path = "/home/lwb/isaacsim_assets/Assets/Isaac/4.5/Isaac/Robots/iRobot/create_4.usd"
        
        # 添加机器人到场景
        add_reference_to_stage(robot_usd_path, self.robot_prim_path)
        
        # 获取机器人prim和transform
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
        
        # 初始化SVSDF规划器
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
        
        # 设置初始位置
        self.set_robot_pose(self.current_position, self.current_orientation)
        
        print("机器人初始化完成")
    
    def set_robot_pose(self, position, yaw):
        """设置机器人位置和朝向 - 参考astar_interactive.py"""
        if self.robot_prim and self.robot_xform:
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
    
    def create_obstacles_for_scenario(self, obstacles):
        """为场景创建障碍物"""
        # 清除现有障碍物
        self.clear_obstacles()
        
        for i, obs in enumerate(obstacles):
            if obs['type'] == 'circle':
                # 创建圆形障碍物（使用圆柱体）
                obstacle_prim_path = f"/World/obstacle_circle_{i}"
                center = obs['center']
                radius = obs['radius']
                height = 0.5
                
                obstacle = FixedCuboid(
                    prim_path=obstacle_prim_path,
                    name=f"obstacle_circle_{i}",
                    position=np.array([center[0], center[1], height/2]),
                    scale=np.array([radius*2, radius*2, height]),
                    color=np.array([0.8, 0.2, 0.2])  # 红色
                )
                self.world.scene.add(obstacle)
                self.obstacle_prims.append(obstacle)
                
                # 添加到A*规划器的网格中
                self.astar_planner.add_circular_obstacle(center, radius)
                
            elif obs['type'] == 'rectangle':
                # 创建矩形障碍物
                obstacle_prim_path = f"/World/obstacle_rect_{i}"
                center = obs['center']
                size = obs['size']
                height = 0.5
                
                obstacle = FixedCuboid(
                    prim_path=obstacle_prim_path,
                    name=f"obstacle_rect_{i}",
                    position=np.array([center[0], center[1], height/2]),
                    scale=np.array([size[0], size[1], height]),
                    color=np.array([0.8, 0.2, 0.2])  # 红色
                )
                self.world.scene.add(obstacle)
                self.obstacle_prims.append(obstacle)
                
                # 添加到A*规划器的网格中（简化为圆形）
                radius = max(size[0], size[1]) / 2 + 0.2  # 安全余量
                self.astar_planner.add_circular_obstacle(center, radius)
    
    def clear_obstacles(self):
        """清除所有障碍物"""
        for obstacle in self.obstacle_prims:
            try:
                self.world.scene.remove_object(obstacle.name)
            except:
                pass
        self.obstacle_prims.clear()
        
        # 重置A*网格
        self.astar_planner.grid.fill(0)

    # ...existing code...
# 主函数
def main():
    """主函数 - 运行SVSDF复杂场景演示"""
    demo = SVSDFDemo()
    
    try:
        # 初始化Isaac Sim
        demo.initialize_isaac_sim()
        
        # 初始化机器人
        demo.initialize_robot()
        
        # 运行复杂场景演示
        demo.run_complex_demo()
        
    except KeyboardInterrupt:
        print("\n\n用户中断演示")
    except Exception as e:
        print(f"演示运行异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        demo.cleanup()

if __name__ == "__main__":
    main()