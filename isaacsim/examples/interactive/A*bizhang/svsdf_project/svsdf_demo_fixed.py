#!/usr/bin/env python3
"""
SVSDF轨迹规划系统Isaac Sim演示脚本 - 修正版
基于astar_interactive.py的成功模式
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
import asyncio
from queue import PriorityQueue
import time
from scipy.spatial.transform import Rotation as R

# Isaac Sim imports (正确的导入方式，参考astar_interactive.py)
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.robot.wheeled_robots import DifferentialController
from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import UsdGeom, Gf, Usd
import isaacsim.core.utils.prims as prim_utils

# 导入SVSDF规划器
import sys
sys.path.append('/home/lwb/isaacsim/exts/isaacsim.examples.interactive/isaacsim/examples/interactive/A*bizhang')
from svsdf_planner import SVSDFPlanner, RobotParams, TrajectoryPoint

# 设置资源路径
asset_root = "/home/lwb/isaacsim_assets/Assets/Isaac/4.5"
carb.settings.get_settings().set("/persistent/isaac/asset_root/default", asset_root)

class SVSDFDemo:
    """SVSDF演示类 - 基于astar_interactive.py模式"""
    
    def __init__(self):
        # 基本设置
        self.world = None
        self.robot_prim_path = "/World/create_3"
        self.robot_prim = None
        self.robot_xform = None
        self.controller = None
        self.svsdf_planner = None
        
        # 演示场景
        self.demo_scenarios = []
        self._setup_demo_scenarios()
        
        # 状态变量
        self.current_scenario = 0
        self.current_trajectory = []
        self.trajectory_index = 0
        
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
        })
        
        # 场景2：障碍物导航
        self.demo_scenarios.append({
            'name': '障碍物导航',
            'description': '复杂障碍物环境中的导航',
            'start_pos': np.array([0.0, 0.0]),
            'goal_pos': np.array([8.0, 6.0]),
            'start_yaw': 0.0,
            'goal_yaw': 0.0,
        })
    
    def initialize_world(self):
        """初始化Isaac Sim世界"""
        print("正在初始化Isaac Sim世界...")
        
        # 创建世界
        self.world = World(stage_units_in_meters=1.0)
        
        # 添加地面
        self.world.scene.add_default_ground_plane()
        
        # 创建机器人
        self._create_robot()
        
        # 创建障碍物
        self._create_obstacles()
        
        # 初始化SVSDF规划器
        self._initialize_svsdf_planner()
        
        print("Isaac Sim世界初始化完成")
    
    def _create_robot(self):
        """创建机器人"""
        try:
            # 加载create_3机器人
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
            
            # 设置初始位置
            self.set_robot_pose([0, 0, 0.1], 0.0)
            
            print("机器人创建成功")
            
        except Exception as e:
            print(f"创建机器人失败: {e}")
    
    def _create_obstacles(self):
        """创建障碍物"""
        try:
            # 简单障碍物配置
            obstacles = [
                {"pos": [3, 3, 0.5], "scale": [2, 2, 1]},
                {"pos": [-3, -3, 0.5], "scale": [2, 2, 1]},
                {"pos": [6, -2, 0.5], "scale": [1.5, 3, 1]},
                {"pos": [-4, 4, 0.5], "scale": [3, 1.5, 1]},
                {"pos": [0, 0, 0.5], "scale": [1, 4, 1]},
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
                print(f"创建障碍物 {i}")
            
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
            
            print("障碍物创建完成")
            
        except Exception as e:
            print(f"创建障碍物失败: {e}")
    
    def _initialize_svsdf_planner(self):
        """初始化SVSDF规划器"""
        try:
            # 创建机器人参数
            robot_params = RobotParams(
                length=0.35,      # Create-3机器人长度
                width=0.33,       # Create-3机器人宽度  
                wheel_base=0.235, # Create-3轮距
                max_vel=0.5,      # 最大线速度
                max_omega=1.5,    # 最大角速度
                max_acc=2.0,      # 最大线加速度
                max_alpha=3.0     # 最大角加速度
            )
            
            # 创建SVSDF规划器
            self.svsdf_planner = SVSDFPlanner(robot_params)
            
            print("SVSDF规划器初始化完成")
            
        except Exception as e:
            print(f"SVSDF规划器初始化失败: {e}")
    
    def set_robot_pose(self, position, yaw):
        """设置机器人位置和朝向"""
        if self.robot_prim:
            try:
                # 清除现有的XForm操作
                self.robot_xform.ClearXformOpOrder()
                
                # 设置平移
                translate_op = self.robot_xform.AddTranslateOp()
                translate_op.Set(Gf.Vec3d(position[0], position[1], position[2]))
                
                # 设置旋转
                rotate_op = self.robot_xform.AddRotateZOp()
                rotate_op.Set(math.degrees(yaw))
                
                print(f"机器人位置设置为: {position[:2]}, 朝向: {math.degrees(yaw):.1f}°")
                
            except Exception as e:
                print(f"设置机器人位置失败: {e}")
    
    def run_demo_scenario(self, scenario_index=0):
        """运行指定的演示场景"""
        if scenario_index >= len(self.demo_scenarios):
            print(f"场景索引 {scenario_index} 超出范围")
            return
        
        scenario = self.demo_scenarios[scenario_index]
        print(f"\n{'='*50}")
        print(f"运行演示场景: {scenario['name']}")
        print(f"描述: {scenario['description']}")
        print(f"{'='*50}")
        
        # 设置机器人初始位置
        initial_pos = np.array([scenario['start_pos'][0], scenario['start_pos'][1], 0.1])
        self.set_robot_pose(initial_pos, scenario['start_yaw'])
        
        # 创建目标标记
        self._create_goal_marker(scenario['goal_pos'])
        
        print(f"场景设置完成")
        print(f"起点: {scenario['start_pos']}")
        print(f"终点: {scenario['goal_pos']}")
        print(f"按空格键开始SVSDF轨迹规划...")
    
    def _create_goal_marker(self, goal_pos):
        """创建目标标记"""
        try:
            # 删除旧的目标标记
            goal_prim_path = "/World/goal_marker"
            if self.world.stage.GetPrimAtPath(goal_prim_path).IsValid():
                self.world.stage.RemovePrim(goal_prim_path)
            
            # 创建新的目标标记
            goal_marker = self.world.scene.add(
                FixedCuboid(
                    prim_path=goal_prim_path,
                    name="goal_marker",
                    position=np.array([goal_pos[0], goal_pos[1], 0.2]),
                    scale=np.array([0.5, 0.5, 0.5]),
                    color=np.array([1.0, 1.0, 0.0])  # 黄色
                )
            )
            print(f"目标标记创建在: {goal_pos}")
            
        except Exception as e:
            print(f"创建目标标记失败: {e}")
    
    def interactive_demo(self):
        """交互式演示"""
        print(f"\n{'='*50}")
        print("SVSDF轨迹规划演示系统")
        print(f"{'='*50}")
        
        print("可用场景:")
        for i, scenario in enumerate(self.demo_scenarios):
            print(f"  {i+1}. {scenario['name']} - {scenario['description']}")
        
        print(f"\n控制:")
        print(f"  数字键 1-{len(self.demo_scenarios)}: 选择场景")
        print(f"  空格键: 开始SVSDF轨迹规划")
        print(f"  ESC: 退出")
        
        # 开始第一个场景
        self.run_demo_scenario(0)

# 主函数
def main():
    """主函数"""
    try:
        print("启动SVSDF演示系统...")
        
        # 创建演示实例
        demo = SVSDFDemo()
        
        # 初始化世界
        demo.initialize_world()
        
        # 开始交互式演示
        demo.interactive_demo()
        
        # 保持程序运行
        while simulation_app.is_running():
            demo.world.step(render=True)
            
    except KeyboardInterrupt:
        print("\n用户中断，退出程序")
    except Exception as e:
        print(f"程序运行错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("清理资源...")
        if 'demo' in locals() and demo.world:
            demo.world.stop()
        simulation_app.close()

if __name__ == "__main__":
    main()
