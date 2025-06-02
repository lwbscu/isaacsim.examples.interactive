# core/svsdf_planner.py
"""
SVSDF轨迹规划器主控制器
集成四个阶段的完整规划系统
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
import time
import asyncio
from dataclasses import dataclass

from core.astar_planner import AStarPlanner
from core.minco_trajectory import MINCOTrajectory
from core.sdf_calculator import SDFCalculator
from core.mpc_controller import MPCController, MPCState, MPCControl
from core.swept_volume_analyzer import SweptVolumeAnalyzer
from robot.differential_robot import DifferentialRobot
from visualization.isaac_sim_visualizer import IsaacSimVisualizer
from utils.config import config
from utils.math_utils import MathUtils

@dataclass
class PlanningResult:
    """规划结果数据结构"""
    success: bool = False
    trajectory: List[np.ndarray] = None
    planning_time: float = 0.0
    swept_volume_info: Dict = None
    performance_metrics: Dict = None
    
    def __post_init__(self):
        if self.trajectory is None:
            self.trajectory = []
        if self.swept_volume_info is None:
            self.swept_volume_info = {}
        if self.performance_metrics is None:
            self.performance_metrics = {}

class SVSDFPlanner:
    """
    扫掠体积感知轨迹规划器主控制器
    
    实现论文中的完整四阶段算法：
    1. A*初始路径搜索
    2. MINCO第一阶段优化（平滑化）
    3. MINCO第二阶段优化（扫掠体积最小化）
    4. MPC实时跟踪控制
    """
    
    def __init__(self, stage, robot_prim_path: str = "/World/Robot"):
        self.stage = stage
        self.robot_prim_path = robot_prim_path
        
        # 初始化各模块
        self.astar_planner = AStarPlanner(
            grid_resolution=config.planning.grid_resolution,
            heuristic_weight=config.planning.heuristic_weight
        )
        
        self.minco_trajectory = MINCOTrajectory(config.planning.num_segments)
        
        self.sdf_calculator = SDFCalculator(
            config.robot.length, 
            config.robot.width
        )
        
        self.mpc_controller = MPCController()
        
        self.swept_volume_analyzer = SweptVolumeAnalyzer(
            config.robot.length,
            config.robot.width
        )
        
        self.robot = DifferentialRobot(robot_prim_path)
        self.visualizer = IsaacSimVisualizer(stage)
        
        # 状态
        self.current_obstacles = []
        self.current_trajectory = []
        self.is_executing = False
        self.execution_start_time = 0.0
        
        # 性能监控
        self.performance_data = {
            'stage_times': {},
            'total_planning_time': 0.0,
            'mpc_computation_times': [],
            'trajectory_quality': {}
        }
        
        print("SVSDF轨迹规划器已初始化")
    
    def initialize_robot(self, initial_position: np.ndarray = np.array([0, 0, 0.1])):
        """初始化机器人"""
        self.robot.initialize(initial_position)
        print(f"机器人已初始化在位置: {initial_position}")
    
    def set_obstacles(self, obstacles: List[Dict]):
        """设置障碍物"""
        self.current_obstacles = obstacles
        self.astar_planner.set_obstacles(obstacles)
        
        # 创建障碍物可视化
        self.visualizer.create_obstacles_visualization(obstacles)
        print(f"已设置 {len(obstacles)} 个障碍物")
    
    def plan_trajectory(self, start_pos: np.ndarray, goal_pos: np.ndarray,
                       start_yaw: float = 0.0, goal_yaw: float = 0.0) -> PlanningResult:
        """
        执行完整的SVSDF轨迹规划
        
        Args:
            start_pos: 起点位置 [x, y]
            goal_pos: 终点位置 [x, y]
            start_yaw: 起点偏航角
            goal_yaw: 终点偏航角
            
        Returns:
            PlanningResult: 规划结果
        """
        print(f"\n=== 开始SVSDF轨迹规划 ===")
        print(f"起点: ({start_pos[0]:.2f}, {start_pos[1]:.2f}, {start_yaw:.2f})")
        print(f"终点: ({goal_pos[0]:.2f}, {goal_pos[1]:.2f}, {goal_yaw:.2f})")
        
        total_start_time = time.time()
        result = PlanningResult()
        
        try:
            # === 第一阶段：A*初始路径搜索 ===
            stage1_start = time.time()
            print("\n--- 第一阶段：A*路径搜索 ---")
            
            initial_path = self.astar_planner.search(start_pos, goal_pos)
            
            if not initial_path:
                print("A*路径搜索失败")
                result.success = False
                return result
            
            stage1_time = time.time() - stage1_start
            self.performance_data['stage_times']['astar'] = stage1_time
            print(f"A*搜索完成，耗时: {stage1_time:.3f}s，路径点数: {len(initial_path)}")
            
            # 创建A*路径可视化
            self.visualizer.create_trajectory_visualization(
                [np.array([p[0], p[1], 0]) for p in initial_path], "astar_path")
            
            # === 第二阶段：MINCO第一次优化（平滑化） ===
            stage2_start = time.time()
            print("\n--- 第二阶段：MINCO轨迹平滑化 ---")
            
            # 构建3D路径点
            waypoints = []
            for i, pos in enumerate(initial_path):
                if i == 0:
                    yaw = start_yaw
                elif i == len(initial_path) - 1:
                    yaw = goal_yaw
                else:
                    # 计算中间点的航向角
                    if i > 0:
                        direction = initial_path[i] - initial_path[i-1]
                        yaw = np.arctan2(direction[1], direction[0])
                    else:
                        yaw = start_yaw
                
                waypoints.append(np.array([pos[0], pos[1], yaw]))
            
            # 计算初始时间分配
            initial_times = []
            for i in range(len(waypoints) - 1):
                segment_length = np.linalg.norm(waypoints[i+1][:2] - waypoints[i][:2])
                segment_time = max(0.5, segment_length / (config.robot.max_linear_velocity * 0.7))
                initial_times.append(segment_time)
            
            # 初始化MINCO轨迹
            self.minco_trajectory.initialize_from_waypoints(waypoints, initial_times)
            
            # 第一阶段优化
            stage1_success = self.minco_trajectory.optimize_stage1(
                config.planning.stage1_weights['energy'],
                config.planning.stage1_weights['time'],
                config.planning.stage1_weights['path_deviation'],
                waypoints
            )
            
            if not stage1_success:
                print("MINCO第一阶段优化失败，使用初始轨迹")
            
            stage2_time = time.time() - stage2_start
            self.performance_data['stage_times']['minco_stage1'] = stage2_time
            print(f"MINCO第一阶段完成，耗时: {stage2_time:.3f}s")
            
            # === 第三阶段：MINCO第二次优化（扫掠体积最小化） ===
            stage3_start = time.time()
            print("\n--- 第三阶段：扫掠体积最小化优化 ---")
            
            # 定义障碍物代价函数
            def obstacle_cost_func(position, velocity):
                return self.sdf_calculator.compute_obstacle_cost(
                    [np.concatenate([position, [0]])], self.current_obstacles)
            
            # 定义扫掠体积代价函数
            def swept_volume_cost_func(segments):
                return self.swept_volume_analyzer.compute_swept_volume_for_minco(segments)
            
            # 第二阶段优化
            stage2_success = self.minco_trajectory.optimize_stage2(
                config.planning.stage2_weights['energy'],
                config.planning.stage2_weights['time'],
                config.planning.stage2_weights['obstacle'],
                config.planning.stage2_weights['swept_volume'],
                obstacle_cost_func,
                swept_volume_cost_func
            )
            
            if not stage2_success:
                print("MINCO第二阶段优化失败，使用第一阶段结果")
            
            stage3_time = time.time() - stage3_start
            self.performance_data['stage_times']['minco_stage2'] = stage3_time
            print(f"MINCO第二阶段完成，耗时: {stage3_time:.3f}s")
            
            # === 生成最终轨迹 ===
            print("\n--- 生成最终轨迹 ---")
            positions, velocities, accelerations, times = self.minco_trajectory.get_discretized_trajectory(
                config.mpc.sample_time)
            
            # 构建轨迹点列表
            self.current_trajectory = []
            for i in range(len(positions)):
                traj_point = np.array([
                    positions[i][0],  # x
                    positions[i][1],  # y  
                    positions[i][2],  # theta
                    times[i]          # time
                ])
                self.current_trajectory.append(traj_point)
            
            # === 扫掠体积分析 ===
            print("\n--- 扫掠体积分析 ---")
            swept_volume_info = self.swept_volume_analyzer.compute_detailed_swept_volume(
                self.current_trajectory)
            
            # === 创建可视化 ===
            print("\n--- 创建可视化 ---")
            
            # 轨迹可视化
            traj_viz_path = self.visualizer.create_trajectory_visualization(
                self.current_trajectory, "final_trajectory")
            
            # 扫掠体积可视化
            swept_viz_path = self.visualizer.create_swept_volume_visualization(
                swept_volume_info['boundary_points'],
                swept_volume_info.get('density_grid'),
                swept_volume_info.get('grid_bounds'),
                "swept_volume"
            )
            
            # === 结果统计 ===
            total_planning_time = time.time() - total_start_time
            self.performance_data['total_planning_time'] = total_planning_time
            
            # 轨迹质量指标
            total_time = self.current_trajectory[-1][3] - self.current_trajectory[0][3]
            path_length = 0.0
            for i in range(1, len(self.current_trajectory)):
                path_length += MathUtils.euclidean_distance(
                    self.current_trajectory[i][:2], self.current_trajectory[i-1][:2])
            
            self.performance_data['trajectory_quality'] = {
                'total_time': total_time,
                'path_length': path_length,
                'average_speed': path_length / total_time if total_time > 0 else 0,
                'swept_volume': swept_volume_info['area']
            }
            
            # 设置结果
            result.success = True
            result.trajectory = self.current_trajectory.copy()
            result.planning_time = total_planning_time
            result.swept_volume_info = swept_volume_info
            result.performance_metrics = self.performance_data.copy()
            
            print(f"\n=== 规划完成 ===")
            print(f"总耗时: {total_planning_time:.3f}s")
            print(f"轨迹时间: {total_time:.3f}s")
            print(f"路径长度: {path_length:.3f}m")
            print(f"扫掠面积: {swept_volume_info['area']:.3f}m²")
            
            return result
            
        except Exception as e:
            print(f"轨迹规划异常: {e}")
            result.success = False
            result.performance_metrics = self.performance_data.copy()
            return result
    
    async def execute_trajectory_async(self, update_callback=None):
        """异步执行轨迹跟踪"""
        if not self.current_trajectory:
            print("没有可执行的轨迹")
            return False
        
        print(f"\n=== 开始轨迹执行 ===")
        self.is_executing = True
        self.execution_start_time = time.time()
        
        # MPC控制循环
        control_dt = config.mpc.sample_time
        trajectory_start_time = self.current_trajectory[0][3]
        trajectory_end_time = self.current_trajectory[-1][3]
        
        try:
            while self.is_executing:
                loop_start = time.time()
                
                # 计算当前轨迹时间
                elapsed_time = time.time() - self.execution_start_time
                current_traj_time = trajectory_start_time + elapsed_time
                
                # 检查是否完成
                if current_traj_time >= trajectory_end_time:
                    print("轨迹执行完成")
                    break
                
                # 更新机器人状态
                self.robot.update_state()
                current_state = self.robot.get_state()
                
                # === 第四阶段：MPC控制 ===
                mpc_start = time.time()
                control = self.mpc_controller.compute_control(
                    current_state, self.current_trajectory, current_traj_time)
                mpc_time = time.time() - mpc_start
                
                self.performance_data['mpc_computation_times'].append(mpc_time)
                
                # 应用控制
                self.robot.apply_control(control)
                
                # 更新可视化
                self._update_real_time_visualization(current_state, control)
                
                # 调用更新回调
                if update_callback:
                    await update_callback(current_state, control, current_traj_time)
                
                # 控制频率
                loop_time = time.time() - loop_start
                sleep_time = max(0, control_dt - loop_time)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                
        except Exception as e:
            print(f"轨迹执行异常: {e}")
            return False
        
        finally:
            self.is_executing = False
            
            # 停止机器人
            stop_control = MPCControl()
            self.robot.apply_control(stop_control)
            
            print(f"轨迹执行结束，总耗时: {time.time() - self.execution_start_time:.3f}s")
        
        return True
    
    def _update_real_time_visualization(self, current_state: MPCState, 
                                      control: MPCControl):
        """更新实时可视化"""
        try:
            # 更新机器人位置
            robot_pose = np.array([current_state.x, current_state.y, current_state.theta])
            self.visualizer.update_robot_visualization(robot_pose, self.robot_prim_path)
            
            # 更新MPC预测轨迹
            if hasattr(self.mpc_controller, 'predicted_trajectory'):
                self.visualizer.create_mpc_prediction_visualization(
                    self.mpc_controller.predicted_trajectory, "mpc_prediction")
            
            # 更新机器人轨迹尾迹
            trail_positions = self.robot.get_trail_positions()
            if len(trail_positions) > 1:
                self.visualizer.create_robot_trail_visualization(
                    trail_positions, "robot_trail")
                
        except Exception as e:
            print(f"更新实时可视化失败: {e}")
    
    def stop_execution(self):
        """停止轨迹执行"""
        self.is_executing = False
        print("轨迹执行已停止")
    
    def get_performance_summary(self) -> Dict:
        """获取性能总结"""
        summary = {
            'planning_performance': self.performance_data.copy(),
            'mpc_performance': self.mpc_controller.get_performance_metrics(),
            'sdf_performance': self.sdf_calculator.get_performance_metrics() if hasattr(self.sdf_calculator, 'get_performance_metrics') else {},
            'swept_volume_performance': self.swept_volume_analyzer.get_performance_metrics()
        }
        
        # 计算平均MPC时间
        if self.performance_data['mpc_computation_times']:
            mpc_times = self.performance_data['mpc_computation_times']
            summary['mpc_avg_time'] = np.mean(mpc_times)
            summary['mpc_max_time'] = np.max(mpc_times)
        
        return summary
    
    def save_results(self, filename: str = "svsdf_results.npz"):
        """保存规划结果"""
        try:
            results_data = {
                'trajectory': np.array(self.current_trajectory),
                'obstacles': self.current_obstacles,
                'performance_data': self.performance_data,
                'robot_config': {
                    'length': config.robot.length,
                    'width': config.robot.width,
                    'max_vel': config.robot.max_linear_velocity
                }
            }
            
            np.savez(filename, **results_data)
            print(f"结果已保存到: {filename}")
            
        except Exception as e:
            print(f"保存结果失败: {e}")
    
    def reset(self):
        """重置规划器状态"""
        self.current_trajectory.clear()
        self.is_executing = False
        self.performance_data = {
            'stage_times': {},
            'total_planning_time': 0.0,
            'mpc_computation_times': [],
            'trajectory_quality': {}
        }
        
        # 重置机器人
        self.robot.reset()
        
        # 清理可视化
        self.visualizer.cleanup_visualizations()
        
        print("规划器已重置")
    
    def cleanup(self):
        """清理资源"""
        self.stop_execution()
        self.robot.cleanup()
        self.visualizer.cleanup_visualizations()
        print("SVSDF规划器资源已清理")