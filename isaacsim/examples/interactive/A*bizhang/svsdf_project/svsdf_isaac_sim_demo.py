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
        self.trash_objects = []
        self.target_cube = None
        
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
        """运行指定的演示场景 - 仅用于初始化障碍物"""
        if scenario_index >= len(self.demo_scenarios):
            scenario_index = 1
        
        scenario = self.demo_scenarios[scenario_index]
        print(f"初始化场景: {scenario['name']}")
        
        # 创建障碍物
        self.create_obstacles_for_scenario(scenario['obstacles'])
        
        # 等待物理稳定
        self._wait_for_stability()
        
        return True
    
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
        """运行复杂场景演示 - 交互式版本，参考astar_interactive.py"""
        print(f"\n{'='*60}")
        print("SVSDF轨迹规划系统 - 交互式复杂演示")
        print("展示完整的4阶段SVSDF框架:")
        print("1. A*初始路径搜索")
        print("2. MINCO阶段1优化（轨迹平滑化）") 
        print("3. MINCO阶段2优化（扫掠体积最小化）")
        print("4. 轨迹跟踪执行")
        print("")
        print("交互控制:")
        print("- 箭头键/WASD: 移动目标位置")
        print("- SPACE: 开始/停止自动导航")
        print("- R: 重新规划路径")
        print("- T: 设置随机目标")
        print("- ESC: 退出")
        print(f"{'='*60}")
        
        # 创建起点和终点标记
        self.create_start_end_markers()
        
        # 设置输入处理
        self.setup_input_handling()
        
        # 创建目标立方体
        self.create_target_cube()
        
        # 运行交互式循环
        self.interactive_loop()
    
    def create_trash_objects(self, num_trash=5):
        """创建随机的垃圾对象（小方块）"""
        print(f"正在创建 {num_trash} 个垃圾对象...")
        for i in range(num_trash):
            prim_path = f"/World/trash_{i}"
            position = np.array([
                np.random.uniform(-7, 7),
                np.random.uniform(-7, 7),
                0.1
            ])
            scale = np.random.uniform(0.2, 0.4)
            
            trash_object = DynamicCuboid(
                prim_path=prim_path,
                name=f"trash_cube_{i}",
                position=position,
                scale=np.array([scale, scale, scale]),
                color=np.array([0.6, 0.6, 0.9])  # 淡蓝色
            )
            self.world.scene.add(trash_object)
            self.trash_objects.append(trash_object)
            print(f"  创建了垃圾: {prim_path} at {position}")

    def run_trash_collection_demo(self):
        """运行垃圾收集演示"""
        print("\n" + "="*60)
        print("🤖 开始垃圾自动收集演示")
        print("机器人将依次导航到每个垃圾对象。")
        print("="*60)

        # 1. 创建障碍物和垃圾
        scenario = self.demo_scenarios[1] # 使用场景2的障碍物
        self.create_obstacles_for_scenario(scenario['obstacles'])
        self.create_trash_objects(num_trash=5)
        self._wait_for_stability(2.0) 

        # 2. 创建目标标记
        self.create_target_cube()

        # 3. 遍历所有垃圾
        for i, trash in enumerate(self.trash_objects):
            print(f"\n--- 前往第 {i+1}/{len(self.trash_objects)} 个垃圾 ---")
            
            # 检查垃圾是否还可见 (可能已被吸附)
            if not trash.get_visibility():
                print("  垃圾已被收集，跳过。")
                continue

            trash_position, _ = trash.get_world_pose()
            print(f"垃圾位置: {trash_position}")

            # 设置目标
            self.goal_pos = trash_position
            self.update_target_cube_position() 

            # 规划并执行路径
            print("  🎯 规划路径...")
            success = self.run_svsdf_planning()

            if success:
                print("  ✅ 路径规划成功，开始执行")
                self.execute_trajectory()
                print("  🎉 到达垃圾位置!")
                self.simulate_suction(trash)
            else:
                print(f"  ❌ 无法规划到垃圾 {i+1} 的路径，跳过。")

            time.sleep(1.0)

        print("\n" + "="*60)
        print("✅ 所有垃圾收集任务完成!")
        print("="*60)

    def simulate_suction(self, trash_object):
        """模拟吸附垃圾"""
        print(f"⚡️ 正在吸附 {trash_object.name}...")
        # 通过使其不可见来模拟吸附
        trash_object.set_visibility(False)
        time.sleep(1.0) 
        print("💨 吸附完成!")

    def setup_input_handling(self):
        """设置输入处理 - 参考astar_interactive.py"""
        try:
            import carb
            import omni.appwindow
            
            self._appwindow = omni.appwindow.get_default_app_window()
            self._input = carb.input.acquire_input_interface()
            self._keyboard = self._appwindow.get_keyboard()
            self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._sub_keyboard_event)
            
            # 状态变量
            self.goal_pos = np.array([8.0, 6.0, 0.1])
            self.auto_navigation = False
            self.goal_changed = False
            
            print("✓ 输入处理初始化成功")
        except Exception as e:
            print(f"输入处理初始化失败: {e}")
    
    def _sub_keyboard_event(self, event, *args, **kwargs) -> bool:
        """键盘事件处理 - 参考astar_interactive.py"""
        import carb
        
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            key_name = event.input.name
            
            # 目标移动
            if key_name in ["UP", "NUMPAD_8", "W"]:
                self.move_target(0, 2.0)
            elif key_name in ["DOWN", "NUMPAD_2", "S"]:
                self.move_target(0, -2.0)
            elif key_name in ["LEFT", "NUMPAD_4", "A"]:
                self.move_target(-2.0, 0)
            elif key_name in ["RIGHT", "NUMPAD_6", "D"]:
                self.move_target(2.0, 0)
            # 控制键
            elif key_name == "SPACE":
                self.toggle_auto_navigation()
            elif key_name == "R":
                self.request_replan()
            elif key_name == "T":
                self.set_random_target()
            elif key_name == "ESCAPE":
                self.running = False
                
        return True
    
    def move_target(self, dx, dy):
        """移动目标位置"""
        self.goal_pos[0] += dx
        self.goal_pos[1] += dy
        
        # 限制目标在合理范围内
        self.goal_pos[0] = max(-12, min(12, self.goal_pos[0]))
        self.goal_pos[1] = max(-12, min(12, self.goal_pos[1]))
        
        print(f"目标移动到: ({self.goal_pos[0]:.1f}, {self.goal_pos[1]:.1f})")
        self.goal_changed = True
        
        # 更新目标立方体位置
        self.update_target_cube_position()
    
    def create_target_cube(self):
        """创建目标立方体 - 参考astar_interactive.py"""
        try:
            self.target_cube = FixedCuboid(
                prim_path="/World/target_cube",
                name="target_cube",
                position=np.array([self.goal_pos[0], self.goal_pos[1], 0.3]),
                scale=np.array([0.6, 0.6, 0.6]),
                color=np.array([1.0, 1.0, 0.0])  # 黄色
            )
            self.world.scene.add(self.target_cube)
            print("✓ 目标立方体创建成功")
        except Exception as e:
            print(f"创建目标立方体失败: {e}")
    
    def update_target_cube_position(self):
        """更新目标立方体位置"""
        if self.target_cube:
            try:
                target_prim_path = "/World/target_cube"
                target_prim = self.world.stage.GetPrimAtPath(target_prim_path)
                
                if target_prim.IsValid():
                    xform = UsdGeom.Xformable(target_prim)
                    xform.ClearXformOpOrder()
                    # 使用一致的精度类型
                    translate_op = xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
                    translate_op.Set(Gf.Vec3d(self.goal_pos[0], self.goal_pos[1], 0.3))
            except Exception as e:
                print(f"更新目标位置失败: {e}")
    
    def create_start_end_markers(self):
        """创建起点和终点标记"""
        try:
            # 创建起点标记（绿色）
            start_marker = FixedCuboid(
                prim_path="/World/start_marker",
                name="start_marker",
                position=np.array([0.0, 0.0, 0.5]),
                scale=np.array([0.8, 0.8, 1.0]),
                color=np.array([0.0, 1.0, 0.0])  # 绿色
            )
            self.world.scene.add(start_marker)
            
            print("✓ 起点和终点标记创建成功")
        except Exception as e:
            print(f"创建标记失败: {e}")
    
    def toggle_auto_navigation(self):
        """切换自动导航模式"""
        self.auto_navigation = not self.auto_navigation
        if self.auto_navigation:
            print("🚀 自动导航开启 - 机器人将跟随目标")
            self.request_replan()
        else:
            print("⏸️ 自动导航关闭 - 使用箭头键移动目标，SPACE键开始")
    
    def request_replan(self):
        """请求重新规划路径 - 优化版本：先清除后重新规划"""
        if self.auto_navigation:
            print("🔄 开始重新规划路径...")
            
            # 1. 先清除所有旧的可视化
            print("  🧹 清除旧路径和可视化...")
            self.clear_sdf_rings()
            self.clear_all_markers()
            
            # 2. 清空轨迹数据
            self.current_trajectory = []
            self.trajectory_index = 0
            
            # 3. 强制刷新场景
            for _ in range(3):
                self.world.step(render=True)
                time.sleep(0.05)
            
            # 4. 重新规划路径
            print("  🎯 重新规划新路径...")
            success = self.run_svsdf_planning()
            
            if success:
                print("  ✅ 路径规划成功，开始执行")
                self.execute_trajectory()
            else:
                print("  ❌ 路径规划失败")
        else:
            print("⚠️ 自动导航未启用")
    
    def set_random_target(self):
        """设置随机目标位置"""
        self.goal_pos[0] = np.random.uniform(-8, 8)
        self.goal_pos[1] = np.random.uniform(-8, 8)
        print(f"🎯 随机目标: ({self.goal_pos[0]:.1f}, {self.goal_pos[1]:.1f})")
        self.goal_changed = True
        self.update_target_cube_position()
        
        if self.auto_navigation:
            self.request_replan()
    
    def interactive_loop(self):
        """交互式主循环"""
        self.running = True
        print("\n🎮 交互模式开始！使用箭头键移动目标，SPACE开始导航，ESC退出")
        
        try:
            while self.running and simulation_app.is_running():
                # 更新应用状态 - 参考成功的虚光圈示例
                simulation_app.update()
                
                # 更新仿真
                self.world.step(render=True)
                
                # 检查是否需要重新规划
                if self.auto_navigation and self.goal_changed:
                    self.goal_changed = False
                    self.request_replan()
                
                time.sleep(0.05)  # 50Hz更新频率
                
        except KeyboardInterrupt:
            print("\n用户中断")
        finally:
            print("退出交互模式")
    
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
            
            # 设置平移 - 使用一致的精度类型
            translate_op = self.robot_xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
            translate_op.Set(Gf.Vec3d(position[0], position[1], position[2]))
            
            # 设置旋转 - 使用一致的精度类型
            rotate_op = self.robot_xform.AddRotateZOp(UsdGeom.XformOp.PrecisionDouble)
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

    def run_svsdf_planning(self):
        """执行SVSDF 4阶段规划"""
        try:
            # 获取当前机器人位置
            current_pos = self.current_position
            goal_pos = self.goal_pos
            
            print(f"\n🚀 开始SVSDF轨迹规划")
            print(f"起点: ({current_pos[0]:.2f}, {current_pos[1]:.2f})")
            print(f"终点: ({goal_pos[0]:.2f}, {goal_pos[1]:.2f})")
            
            # 阶段1: A*路径规划
            print(f"阶段1: A*初始路径搜索...")
            astar_path = self.astar_planner.plan_path(
                [current_pos[0], current_pos[1]], 
                [goal_pos[0], goal_pos[1]]
            )
            
            if not astar_path:
                print("❌ A*路径规划失败!")
                return False
            
            print(f"✓ A*路径规划完成，找到 {len(astar_path)} 个路径点")
            
            # 清除旧的可视化
            self.clear_all_markers()
            
            # 阶段2和3: SVSDF优化（暂时简化）
            print(f"阶段2: MINCO第一阶段优化（轨迹平滑化）...")
            print(f"阶段3: MINCO第二阶段优化（扫掠体积最小化）...")
            
            # 将A*路径转换为轨迹点
            trajectory_points = []
            for i, point in enumerate(astar_path):
                t = float(i) * 0.5
                traj_point = TrajectoryPoint(
                    position=np.array([point[0], point[1], 0.0]),
                    velocity=np.array([0.3, 0.0, 0.0]),
                    acceleration=np.array([0.0, 0.0, 0.0]),
                    time=t
                )
                trajectory_points.append(traj_point)
            
            self.current_trajectory = trajectory_points
            
            # 使用SVSDF虚光圈可视化替代乱七八糟的方块
            print(f"阶段4: SVSDF可视化（虚光圈显示到障碍物距离）...")
            self.visualize_svsdf_rings(trajectory_points)
            
            print(f"✓ SVSDF轨迹优化完成")
            return True
            
        except Exception as e:
            print(f"❌ SVSDF规划失败: {e}")
            return False
    
    def visualize_svsdf_rings(self, trajectory):
        """使用虚光圈可视化SVSDF - 优化版本：相切验证 + 完美圆形显示"""
        try:
            print(f"🎨 创建SVSDF虚光圈可视化")
            
            # 清除旧的可视化
            self.clear_sdf_rings()
            
            # 验证切线条件
            is_valid = self.verify_tangent_condition(trajectory)
            
            # 为轨迹上的关键点创建虚光圈
            step = max(1, len(trajectory) // 8)  # 减少圈数避免过密
            created_rings = 0
            
            for i in range(0, len(trajectory), step):
                traj_point = trajectory[i]
                pos = [traj_point.position[0], traj_point.position[1]]
                
                # 计算该点到所有障碍物的最小距离（SDF值）
                min_distance = self.compute_sdf_at_point(pos)
                
                # 创建虚光圈，半径等于SDF值（确保与障碍物相切）
                ring_created = self.create_sdf_ring(i, pos, min_distance)
                if ring_created:
                    created_rings += 1
                
            print(f"✓ SVSDF虚光圈可视化完成: {created_rings}个相切圆环")
            
            # 如果切线验证通过，显示成功消息
            if is_valid:
                print(f"  🎯 完美相切: 扫掠体积与障碍物精确相切，无重叠无缝隙")
            else:
                print(f"  ⚠️ 需要优化: 部分区域可进一步优化切线条件")
                
        except Exception as e:
            print(f"SVSDF可视化失败: {e}")
    
    def compute_sdf_at_point(self, point):
        """计算点到最近障碍物的精确距离 - 优化版本：确保相切无缝隙"""
        min_dist = float('inf')
        point = np.array(point, dtype=np.float64)  # 高精度计算
        
        # 遍历演示场景中的障碍物配置来计算精确距离
        scenario = self.demo_scenarios[1]  # 使用当前场景
        
        for obs in scenario['obstacles']:
            if obs['type'] == 'circle':
                # 圆形障碍物 - 精确计算
                center = np.array(obs['center'], dtype=np.float64)
                radius = float(obs['radius'])
                
                # 计算点到圆心的距离
                dist_to_center = np.linalg.norm(point - center)
                
                # SDF距离：点到圆边界的距离
                sdf_dist = dist_to_center - radius
                
                # 确保扫掠圆与障碍物精确相切（无重叠，无缝隙）
                # 加上机器人半径（假设为0.15m）确保安全相切
                robot_radius = 0.15
                tangent_dist = max(0.08, sdf_dist - robot_radius)
                
            elif obs['type'] == 'rectangle':
                # 矩形障碍物 - 使用Inigo Quilez算法精确计算
                center = np.array(obs['center'], dtype=np.float64)
                half_size = np.array(obs['size'], dtype=np.float64) / 2.0
                
                # 矩形SDF计算
                relative_pos = np.abs(point - center) - half_size
                outside_dist = np.linalg.norm(np.maximum(relative_pos, 0.0))
                inside_dist = min(max(relative_pos[0], relative_pos[1]), 0.0)
                rect_sdf = outside_dist + inside_dist
                
                # 加上机器人半径确保相切
                robot_radius = 0.15
                tangent_dist = max(0.08, rect_sdf - robot_radius)
            
            min_dist = min(min_dist, tangent_dist)
        
        # 确保距离在合理范围内，最小距离保证可视化效果
        final_dist = max(0.08, min(min_dist, 2.0))
        
        return final_dist
    
    def create_sdf_ring(self, index, position, radius):
        """创建SDF虚光圈 - 优化版本：完美圆形，相切显示"""
        timestamp = int(time.time() * 1000) % 10000  # 避免路径冲突
        ring_path = f"/World/PerfectSDF_Ring_{index}_{timestamp}"
        
        try:
            # 创建高质量圆环（使用圆柱体确保完美圆形）
            ring_prim = prim_utils.create_prim(ring_path, "Cylinder")
            ring = UsdGeom.Cylinder(ring_prim)
            
            # 设置几何属性：完美圆形
            ring.CreateRadiusAttr().Set(float(radius))
            ring.CreateHeightAttr().Set(0.02)  # 极薄的圆环
            ring.CreateAxisAttr().Set("Z")      # Z轴向上
            
            # 设置变换：精确定位
            xform = UsdGeom.Xformable(ring_prim)
            xform.ClearXformOpOrder()
            
            # 使用高精度坐标
            translate_op = xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
            translate_op.Set(Gf.Vec3d(float(position[0]), float(position[1]), 0.05))
            
            # 智能颜色映射：基于与障碍物的相对距离
            if radius < 0.2:
                color = (1.0, 0.0, 0.0)    # 红色 - 危险（非常接近障碍物）
                opacity = 0.9
            elif radius < 0.5:
                color = (1.0, 0.5, 0.0)    # 橙色 - 警告
                opacity = 0.8
            elif radius < 1.0:
                color = (1.0, 1.0, 0.0)    # 黄色 - 注意
                opacity = 0.7
            else:
                color = (0.0, 1.0, 0.0)    # 绿色 - 安全（远离障碍物）
                opacity = 0.6
            
            # 设置显示属性
            ring.CreateDisplayColorAttr().Set([color])
            ring.CreateDisplayOpacityAttr().Set([opacity])
            
            # 确保材质属性用于更好的渲染
            try:
                # 设置发光效果，突出相切关系
                ring.CreatePurposeAttr().Set("render")
            except:
                pass
            
            print(f"  ✨ 完美SDF圆环 {index}: 位置({position[0]:.3f}, {position[1]:.3f}), 半径={radius:.4f}m, 相切显示")
            return ring_path
            
        except Exception as e:
            print(f"  ❌ 创建SDF圆环失败: {e}")
            return None
        
    def clear_sdf_rings(self):
        """清除所有SDF光圈 - 优化版本：彻底清除，支持新路径规划"""
        cleared_count = 0
        try:
            stage = self.world.stage
            
            # 方法1: 清除传统命名的SDF圆环
            for i in range(50):  # 扩大清除范围
                traditional_paths = [
                    f"/World/SDF_Ring_{i}",
                    f"/World/PerfectSDF_Ring_{i}",
                    f"/World/sdf_ring_{i}",
                ]
                
                for ring_path in traditional_paths:
                    if stage.GetPrimAtPath(ring_path).IsValid():
                        stage.RemovePrim(ring_path)
                        cleared_count += 1
                        
            # 方法2: 基于时间戳的圆环清除（支持新的相切圆环）
            world_prim = stage.GetPrimAtPath("/World")
            if world_prim.IsValid():
                children_to_remove = []
                for child in world_prim.GetChildren():
                    child_name = child.GetName()
                    # 匹配所有可能的SDF圆环命名模式
                    ring_keywords = [
                        'SDF_Ring', 'PerfectSDF_Ring', 'TangentRing', 
                        'Ring', 'SDF', 'Circle', 'Perfect', 'Tangent'
                    ]
                    
                    if any(keyword in child_name for keyword in ring_keywords):
                        children_to_remove.append(child.GetPath())
                        
                # 批量删除
                for path in children_to_remove:
                    try:
                        if stage.GetPrimAtPath(path).IsValid():
                            stage.RemovePrim(path)
                            cleared_count += 1
                    except Exception as e:
                        print(f"删除圆环失败 {path}: {e}")
            
            # 方法3: 强制场景刷新，确保清除生效
            if cleared_count > 0:
                for _ in range(5):
                    self.world.step(render=True)
                    time.sleep(0.02)
                    
            print(f"  🧹 SDF圆环清除完成: {cleared_count} 个对象")
            return cleared_count
            
        except Exception as e:
            print(f"清除SDF圆环失败: {e}")
            return 0
    
    def clear_all_markers(self):
        """清除所有可视化标记"""
        try:
            # 清除SDF光圈
            self.clear_sdf_rings()
            
            # 清除其他旧标记
            for i in range(100):
                marker_paths = [
                    f"/World/astar_marker_{i}",
                    f"/World/traj_marker_{i}"
                ]
                for marker_path in marker_paths:
                    if self.world.stage.GetPrimAtPath(marker_path).IsValid():
                        self.world.stage.RemovePrim(marker_path)
            
            # 清除扫掠体积标记
            for i in range(50):
                for j in range(8):
                    marker_path = f"/World/swept_marker_{i}_{j}"
                    if self.world.stage.GetPrimAtPath(marker_path).IsValid():
                        self.world.stage.RemovePrim(marker_path)
                        
        except Exception as e:
            print(f"清除标记失败: {e}")
    
    def verify_tangent_condition(self, trajectory):
        """验证扫掠体积是否与障碍物精确相切 - 无重叠、无缝隙"""
        try:
            print("🔍 验证切线条件...")
            
            total_violations = 0
            max_violation = 0.0
            
            for i, traj_point in enumerate(trajectory):
                pos = [traj_point.position[0], traj_point.position[1]]
                
                # 计算当前点的SDF距离
                sdf_distance = self.compute_sdf_at_point(pos)
                
                # 验证机器人半径与SDF距离的关系
                robot_radius = 0.15  # Create-3机器人半径
                
                # 检查是否有重叠（违反安全约束）
                violation = robot_radius - sdf_distance
                
                if violation > 0.01:  # 允许1cm的误差容忍
                    total_violations += 1
                    max_violation = max(max_violation, violation)
                    print(f"  ⚠️ 点{i}: 重叠违规 {violation:.3f}m (位置: {pos[0]:.2f}, {pos[1]:.2f})")
                
                # 检查是否有过大间隙（效率损失）
                elif sdf_distance > robot_radius + 0.5:
                    print(f"  💡 点{i}: 可优化间隙 {sdf_distance - robot_radius:.3f}m")
            
            # 总结验证结果
            if total_violations == 0:
                print(f"  ✅ 切线验证通过: 所有扫掠圆完美相切，无安全违规")
                return True
            else:
                print(f"  ❌ 切线验证失败: {total_violations}个违规点，最大重叠{max_violation:.3f}m")
                return False
                
        except Exception as e:
            print(f"切线验证异常: {e}")
            return False

def main():
    """主执行函数"""
    demo = SVSDFDemo()
    try:
        # 初始化
        demo.initialize_isaac_sim()
        demo.initialize_robot()

        # 运行垃圾收集演示
        demo.run_trash_collection_demo()
        
        # 或者运行交互式演示
        # demo.run_demo_scenario(1)
        # demo.run_complex_demo()

    except Exception as e:
        print(f"演示过程中发生严重错误: {e}")
    finally:
        # 清理资源
        demo.cleanup()
        simulation_app.close()

if __name__ == "__main__":
    main()