# svsdf_isaac_sim_demo.py
"""
SVSDF轨迹规划系统Isaac Sim演示脚本
完整展示扫掠体积感知轨迹规划的四个阶段
"""
import numpy as np
import asyncio
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.extensions import enable_extension

# 导入我们的模块
from core.svsdf_planner import SVSDFPlanner
from utils.config import config
import carb

class SVSDFDemo:
    """SVSDF演示类"""
    
    def __init__(self):
        self.world = None
        self.planner = None
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
    
    async def initialize_isaac_sim(self):
        """初始化Isaac Sim环境"""
        print("正在初始化Isaac Sim环境...")
        
        # 启用必要的扩展
        enable_extension("omni.isaac.core")
        enable_extension("omni.isaac.core_archive")
        enable_extension("omni.isaac.nucleus")
        
        # 创建世界
        self.world = World(stage_units_in_meters=1.0)
        await self.world.initialize_simulation_context_async()
        
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
    
    async def run_demo_scenario(self, scenario_index: int = 0):
        """运行指定的演示场景"""
        if scenario_index >= len(self.demo_scenarios):
            print(f"场景索引 {scenario_index} 超出范围")
            return
        
        scenario = self.demo_scenarios[scenario_index]
        print(f"\n{'='*50}")
        print(f"运行演示场景: {scenario['name']}")
        print(f"描述: {scenario['description']}")
        print(f"{'='*50}")
        
        # 重置环境
        if self.planner:
            self.planner.reset()
        
        # 创建规划器
        stage = omni.usd.get_context().get_stage()
        self.planner = SVSDFPlanner(stage)
        
        # 初始化机器人
        initial_pos = np.array([scenario['start_pos'][0], scenario['start_pos'][1], 0.1])
        self.planner.initialize_robot(initial_pos)
        
        # 设置障碍物
        self.planner.set_obstacles(scenario['obstacles'])
        
        # 等待物理稳定
        await self._wait_for_stability()
        
        # 执行轨迹规划
        print(f"\n开始轨迹规划...")
        planning_result = self.planner.plan_trajectory(
            scenario['start_pos'],
            scenario['goal_pos'],
            scenario['start_yaw'],
            scenario['goal_yaw']
        )
        
        if not planning_result.success:
            print("轨迹规划失败!")
            return
        
        # 显示性能指标
        self._display_performance_metrics(planning_result)
        
        # 询问用户是否执行轨迹
        print(f"\n规划完成! 是否执行轨迹? (y/n): ", end="")
        
        # 在实际应用中这里可以添加UI交互
        # 现在直接执行
        print("y (自动)")
        
        # 执行轨迹
        print(f"开始执行轨迹...")
        
        # 创建进度回调
        async def progress_callback(state, control, traj_time):
            # 每秒打印一次进度
            if int(traj_time * 10) % 10 == 0:  # 每0.1秒
                completion = min(100, (traj_time / planning_result.trajectory[-1][3]) * 100)
                print(f"执行进度: {completion:.1f}% - 位置: ({state.x:.2f}, {state.y:.2f})")
        
        success = await self.planner.execute_trajectory_async(progress_callback)
        
        if success:
            print(f"✓ 场景 '{scenario['name']}' 执行完成!")
            
            # 显示最终性能总结
            final_performance = self.planner.get_performance_summary()
            self._display_final_summary(final_performance)
            
            # 保存结果
            filename = f"svsdf_results_{scenario['name'].replace(' ', '_')}.npz"
            self.planner.save_results(filename)
            
        else:
            print(f"✗ 场景 '{scenario['name']}' 执行失败!")
    
    async def run_all_scenarios(self):
        """依次运行所有演示场景"""
        print(f"\n开始运行所有 {len(self.demo_scenarios)} 个演示场景")
        
        for i, scenario in enumerate(self.demo_scenarios):
            print(f"\n{'='*60}")
            print(f"场景 {i+1}/{len(self.demo_scenarios)}: {scenario['name']}")
            print(f"{'='*60}")
            
            await self.run_demo_scenario(i)
            
            # 场景间等待
            if i < len(self.demo_scenarios) - 1:
                print(f"\n等待 3 秒后开始下一个场景...")
                await asyncio.sleep(3)
        
        print(f"\n🎉 所有演示场景已完成!")
    
    async def interactive_demo(self):
        """交互式演示"""
        while True:
            print(f"\n{'='*50}")
            print("SVSDF轨迹规划演示系统")
            print(f"{'='*50}")
            
            print("可用场景:")
            for i, scenario in enumerate(self.demo_scenarios):
                print(f"  {i+1}. {scenario['name']} - {scenario['description']}")
            
            print(f"\n选项:")
            print(f"  {len(self.demo_scenarios)+1}. 运行所有场景")
            print(f"  {len(self.demo_scenarios)+2}. 自定义场景")
            print(f"  0. 退出")
            
            try:
                choice = input(f"\n请选择 (0-{len(self.demo_scenarios)+2}): ")
                choice = int(choice)
                
                if choice == 0:
                    print("退出演示")
                    break
                elif 1 <= choice <= len(self.demo_scenarios):
                    await self.run_demo_scenario(choice - 1)
                elif choice == len(self.demo_scenarios) + 1:
                    await self.run_all_scenarios()
                elif choice == len(self.demo_scenarios) + 2:
                    await self.custom_scenario()
                else:
                    print("无效选择，请重试")
                    
            except ValueError:
                print("请输入有效数字")
            except KeyboardInterrupt:
                print("\n\n用户中断，退出演示")
                break
    
    async def custom_scenario(self):
        """自定义场景"""
        print(f"\n--- 自定义场景设置 ---")
        
        try:
            # 获取用户输入
            start_x = float(input("起点X坐标 (默认0.0): ") or "0.0")
            start_y = float(input("起点Y坐标 (默认0.0): ") or "0.0")
            goal_x = float(input("终点X坐标 (默认5.0): ") or "5.0")
            goal_y = float(input("终点Y坐标 (默认3.0): ") or "3.0")
            
            start_yaw = float(input("起点偏航角/度 (默认0.0): ") or "0.0") * np.pi / 180
            goal_yaw = float(input("终点偏航角/度 (默认0.0): ") or "0.0") * np.pi / 180
            
            # 简化障碍物设置
            num_obstacles = int(input("障碍物数量 (默认1): ") or "1")
            
            obstacles = []
            for i in range(num_obstacles):
                print(f"\n障碍物 {i+1}:")
                obs_x = float(input(f"  X坐标 (默认{2.0+i}): ") or str(2.0+i))
                obs_y = float(input(f"  Y坐标 (默认{1.5+i*0.5}): ") or str(1.5+i*0.5))
                obs_r = float(input(f"  半径 (默认0.5): ") or "0.5")
                
                obstacles.append({
                    'type': 'circle',
                    'center': [obs_x, obs_y],
                    'radius': obs_r
                })
            
            # 创建自定义场景
            custom_scenario = {
                'name': '自定义场景',
                'description': '用户自定义的导航场景',
                'start_pos': np.array([start_x, start_y]),
                'goal_pos': np.array([goal_x, goal_y]),
                'start_yaw': start_yaw,
                'goal_yaw': goal_yaw,
                'obstacles': obstacles
            }
            
            # 临时添加到场景列表
            self.demo_scenarios.append(custom_scenario)
            
            # 运行自定义场景
            await self.run_demo_scenario(len(self.demo_scenarios) - 1)
            
            # 移除临时场景
            self.demo_scenarios.pop()
            
        except ValueError:
            print("输入格式错误，返回主菜单")
        except Exception as e:
            print(f"自定义场景设置失败: {e}")
    
    async def _wait_for_stability(self, duration: float = 2.0):
        """等待物理系统稳定"""
        print(f"等待物理系统稳定 ({duration}s)...")
        
        for _ in range(int(duration * 10)):
            await self.world.step_async()
            await asyncio.sleep(0.1)
    
    def _display_performance_metrics(self, result):
        """显示性能指标"""
        print(f"\n--- 性能指标 ---")
        print(f"总规划时间: {result.planning_time:.3f}s")
        
        if 'stage_times' in result.performance_metrics:
            stages = result.performance_metrics['stage_times']
            if 'astar' in stages:
                print(f"A*搜索时间: {stages['astar']:.3f}s")
            if 'minco_stage1' in stages:
                print(f"MINCO阶段1时间: {stages['minco_stage1']:.3f}s")
            if 'minco_stage2' in stages:
                print(f"MINCO阶段2时间: {stages['minco_stage2']:.3f}s")
        
        if 'trajectory_quality' in result.performance_metrics:
            quality = result.performance_metrics['trajectory_quality']
            print(f"轨迹总时间: {quality.get('total_time', 0):.3f}s")
            print(f"路径长度: {quality.get('path_length', 0):.3f}m")
            print(f"平均速度: {quality.get('average_speed', 0):.3f}m/s")
            print(f"扫掠面积: {quality.get('swept_volume', 0):.3f}m²")
    
    def _display_final_summary(self, performance):
        """显示最终性能总结"""
        print(f"\n--- 最终性能总结 ---")
        
        if 'mpc_avg_time' in performance:
            print(f"MPC平均计算时间: {performance['mpc_avg_time']:.3f}ms")
            print(f"MPC最大计算时间: {performance['mpc_max_time']:.3f}ms")
        
        if 'planning_performance' in performance:
            planning = performance['planning_performance']
            if 'mpc_computation_times' in planning:
                mpc_times = planning['mpc_computation_times']
                if mpc_times:
                    print(f"MPC调用次数: {len(mpc_times)}")
                    print(f"实时控制成功率: {len([t for t in mpc_times if t < 10])/len(mpc_times)*100:.1f}%")
    
    def cleanup(self):
        """清理资源"""
        if self.planner:
            self.planner.cleanup()
        
        if self.world:
            self.world.stop()
        
        print("演示系统已清理")

# 主函数
async def main():
    """主函数"""
    demo = SVSDFDemo()
    
    try:
        # 初始化Isaac Sim
        await demo.initialize_isaac_sim()
        
        # 运行交互式演示
        await demo.interactive_demo()
        
    except KeyboardInterrupt:
        print("\n\n用户中断演示")
    except Exception as e:
        print(f"演示运行异常: {e}")
    finally:
        demo.cleanup()

if __name__ == "__main__":
    # 设置事件循环策略
    import sys
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # 运行演示
    asyncio.run(main())