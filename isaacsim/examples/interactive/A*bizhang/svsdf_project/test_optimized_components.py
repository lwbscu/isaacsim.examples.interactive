"""
SVSDF优化组件综合测试套件
测试所有优化后的核心组件的性能和功能
"""
import numpy as np
import time
import sys
import os
from typing import List, Dict, Any
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import warnings

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入优化组件
from core.svsdf_planner_optimized import SVSDFPlannerOptimized
from core.sdf_calculator_optimized import SDFCalculatorOptimized  
from core.minco_trajectory_optimized import MINCOTrajectoryOptimized
from core.mpc_controller_optimized import MPCControllerOptimized
from core.swept_volume_analyzer_optimized import SweptVolumeAnalyzerOptimized

# 导入原始组件进行对比
from core.svsdf_planner import SVSDFPlanner
from core.sdf_calculator import SDFCalculator
from core.minco_trajectory import MINCOTrajectory
from core.mpc_controller import MPCController
from core.swept_volume_analyzer import SweptVolumeAnalyzer

from utils.config import config
from utils.math_utils import MathUtils

# 忽略性能警告
warnings.filterwarnings('ignore')

class OptimizedComponentsTester:
    """优化组件测试器"""
    
    def __init__(self):
        self.test_results = {}
        self.performance_metrics = {}
        
        # 测试场景参数
        self.robot_length = 1.0
        self.robot_width = 0.6
        self.map_size = (20.0, 20.0)
        self.resolution = 0.1
        
        # 测试障碍物
        self.obstacles = [
            {'center': [5.0, 5.0], 'radius': 1.5},
            {'center': [10.0, 8.0], 'radius': 1.0},
            {'center': [15.0, 12.0], 'radius': 2.0},
            {'center': [8.0, 15.0], 'radius': 1.2},
            {'center': [12.0, 3.0], 'radius': 0.8}
        ]
        
        print("优化组件测试器初始化完成")
        print(f"机器人尺寸: {self.robot_length}m x {self.robot_width}m")
        print(f"地图大小: {self.map_size[0]}m x {self.map_size[1]}m")
        print(f"障碍物数量: {len(self.obstacles)}")
    
    def test_sdf_calculator_optimized(self) -> Dict[str, Any]:
        """测试优化的SDF计算器"""
        print("\n=== 测试SDF计算器优化版本 ===")
        
        # 创建原始和优化版本
        sdf_original = SDFCalculator(self.robot_length, self.robot_width)
        sdf_optimized = SDFCalculatorOptimized(self.robot_length, self.robot_width)
        
        # 添加障碍物
        for obs in self.obstacles:
            sdf_original.add_obstacle(obs['center'], obs['radius'])
            sdf_optimized.add_obstacle(obs['center'], obs['radius'])
        
        # 生成测试点
        test_points = []
        for i in range(100):
            for j in range(100):
                x = i * self.map_size[0] / 99
                y = j * self.map_size[1] / 99
                test_points.append([x, y])
        
        print(f"测试点数量: {len(test_points)}")
        
        # 测试原始版本
        start_time = time.time()
        original_results = []
        for point in test_points:
            sdf_value = sdf_original.compute_sdf(point)
            original_results.append(sdf_value)
        original_time = time.time() - start_time
        
        # 测试优化版本
        start_time = time.time()
        optimized_results = sdf_optimized.compute_sdf_batch(test_points)
        optimized_time = time.time() - start_time
        
        # 计算精度差异
        diff = np.array(original_results) - np.array(optimized_results)
        max_diff = np.max(np.abs(diff))
        mean_diff = np.mean(np.abs(diff))
        
        # 性能提升
        speedup = original_time / optimized_time if optimized_time > 0 else float('inf')
        
        results = {
            'original_time': original_time,
            'optimized_time': optimized_time,
            'speedup': speedup,
            'max_diff': max_diff,
            'mean_diff': mean_diff,
            'test_points_count': len(test_points)
        }
        
        print(f"原始版本时间: {original_time:.3f}s")
        print(f"优化版本时间: {optimized_time:.3f}s")
        print(f"性能提升: {speedup:.1f}x")
        print(f"最大误差: {max_diff:.6f}")
        print(f"平均误差: {mean_diff:.6f}")
        
        self.test_results['sdf_calculator'] = results
        return results
    
    def test_minco_trajectory_optimized(self) -> Dict[str, Any]:
        """测试优化的MINCO轨迹生成器"""
        print("\n=== 测试MINCO轨迹生成器优化版本 ===")
        
        # 创建原始和优化版本
        minco_original = MINCOTrajectory()
        minco_optimized = MINCOTrajectoryOptimized()
        
        # 定义路径点
        waypoints = [
            [2.0, 2.0, 0.0],
            [6.0, 4.0, np.pi/4],
            [10.0, 8.0, np.pi/2],
            [14.0, 12.0, 3*np.pi/4],
            [18.0, 16.0, np.pi]
        ]
        
        time_segments = [2.0, 3.0, 2.5, 2.0]  # 每段时间
        
        print(f"路径点数量: {len(waypoints)}")
        print(f"时间段数量: {len(time_segments)}")
        
        # 测试原始版本
        start_time = time.time()
        try:
            original_trajectory = minco_original.generate_trajectory(waypoints, time_segments)
            original_time = time.time() - start_time
            original_success = True
        except Exception as e:
            print(f"原始版本失败: {e}")
            original_time = float('inf')
            original_success = False
            original_trajectory = None
        
        # 测试优化版本
        start_time = time.time()
        try:
            optimized_trajectory = minco_optimized.generate_trajectory_optimized(waypoints, time_segments)
            optimized_time = time.time() - start_time
            optimized_success = True
        except Exception as e:
            print(f"优化版本失败: {e}")
            optimized_time = float('inf')
            optimized_success = False
            optimized_trajectory = None
        
        # 计算质量指标
        quality_metrics = {}
        if optimized_success and optimized_trajectory:
            # 生成轨迹点进行分析
            trajectory_points = []
            current_time = 0.0
            
            for segment in optimized_trajectory.segments:
                duration = segment.duration
                num_samples = max(10, int(duration / 0.1))
                
                for i in range(num_samples + 1):
                    t = i * duration / num_samples
                    if t > duration:
                        t = duration
                    
                    try:
                        pos = segment.evaluate_position(t)
                        vel = segment.evaluate_velocity(t)
                        trajectory_points.append({
                            'time': current_time + t,
                            'position': pos,
                            'velocity': vel
                        })
                    except:
                        pass
                
                current_time += duration
            
            if trajectory_points:
                # 计算平滑度
                velocities = [np.linalg.norm(p['velocity'][:2]) for p in trajectory_points]
                accelerations = []
                for i in range(1, len(velocities)):
                    dt = trajectory_points[i]['time'] - trajectory_points[i-1]['time']
                    if dt > 0:
                        acc = (velocities[i] - velocities[i-1]) / dt
                        accelerations.append(acc)
                
                quality_metrics = {
                    'trajectory_length': len(trajectory_points),
                    'max_velocity': max(velocities) if velocities else 0.0,
                    'mean_velocity': np.mean(velocities) if velocities else 0.0,
                    'max_acceleration': max(np.abs(accelerations)) if accelerations else 0.0,
                    'mean_acceleration': np.mean(np.abs(accelerations)) if accelerations else 0.0
                }
        
        # 性能提升
        speedup = original_time / optimized_time if optimized_time > 0 else float('inf')
        
        results = {
            'original_time': original_time,
            'optimized_time': optimized_time,
            'original_success': original_success,
            'optimized_success': optimized_success,
            'speedup': speedup,
            'quality_metrics': quality_metrics,
            'waypoints_count': len(waypoints)
        }
        
        print(f"原始版本: {'成功' if original_success else '失败'} ({original_time:.3f}s)")
        print(f"优化版本: {'成功' if optimized_success else '失败'} ({optimized_time:.3f}s)")
        if optimized_success and original_success:
            print(f"性能提升: {speedup:.1f}x")
        
        if quality_metrics:
            print(f"轨迹质量指标:")
            for key, value in quality_metrics.items():
                print(f"  {key}: {value:.3f}")
        
        self.test_results['minco_trajectory'] = results
        return results
    
    def test_mpc_controller_optimized(self) -> Dict[str, Any]:
        """测试优化的MPC控制器"""
        print("\n=== 测试MPC控制器优化版本 ===")
        
        # 创建原始和优化版本
        mpc_original = MPCController(self.robot_length, self.robot_width)
        mpc_optimized = MPCControllerOptimized(self.robot_length, self.robot_width)
        
        # 生成参考轨迹
        reference_trajectory = []
        for i in range(50):
            t = i * 0.1
            x = 2.0 + 0.5 * t
            y = 2.0 + 0.3 * np.sin(0.5 * t)
            theta = 0.1 * t
            reference_trajectory.append(np.array([x, y, theta, t]))
        
        # 初始状态
        initial_state = np.array([2.0, 2.0, 0.0, 0.0, 0.0])  # [x, y, theta, v, omega]
        
        print(f"参考轨迹长度: {len(reference_trajectory)}")
        
        # 测试原始版本
        control_times_original = []
        for i in range(10):  # 测试10个控制周期
            start_time = time.time()
            try:
                control = mpc_original.compute_control(initial_state, reference_trajectory)
                control_time = time.time() - start_time
                control_times_original.append(control_time)
            except Exception as e:
                print(f"原始MPC控制失败: {e}")
                break
        
        # 测试优化版本
        control_times_optimized = []
        for i in range(10):  # 测试10个控制周期
            start_time = time.time()
            try:
                control = mpc_optimized.compute_control_optimized(initial_state, reference_trajectory)
                control_time = time.time() - start_time
                control_times_optimized.append(control_time)
            except Exception as e:
                print(f"优化MPC控制失败: {e}")
                break
        
        # 计算性能指标
        original_mean_time = np.mean(control_times_original) if control_times_original else float('inf')
        optimized_mean_time = np.mean(control_times_optimized) if control_times_optimized else float('inf')
        speedup = original_mean_time / optimized_mean_time if optimized_mean_time > 0 else float('inf')
        
        # 获取优化版本的性能指标
        performance_metrics = mpc_optimized.get_performance_metrics_detailed()
        
        results = {
            'original_mean_time': original_mean_time,
            'optimized_mean_time': optimized_mean_time,
            'speedup': speedup,
            'original_success_rate': len(control_times_original) / 10,
            'optimized_success_rate': len(control_times_optimized) / 10,
            'performance_metrics': performance_metrics.__dict__ if hasattr(performance_metrics, '__dict__') else {}
        }
        
        print(f"原始版本平均时间: {original_mean_time:.3f}s")
        print(f"优化版本平均时间: {optimized_mean_time:.3f}s")
        print(f"性能提升: {speedup:.1f}x")
        print(f"原始版本成功率: {len(control_times_original)/10:.1%}")
        print(f"优化版本成功率: {len(control_times_optimized)/10:.1%}")
        
        self.test_results['mpc_controller'] = results
        return results
    
    def test_swept_volume_analyzer_optimized(self) -> Dict[str, Any]:
        """测试优化的扫掠体积分析器"""
        print("\n=== 测试扫掠体积分析器优化版本 ===")
        
        # 创建原始和优化版本
        sva_original = SweptVolumeAnalyzer(self.robot_length, self.robot_width)
        sva_optimized = SweptVolumeAnalyzerOptimized(
            self.robot_length, self.robot_width, 
            enable_parallel=True, max_workers=4
        )
        
        # 生成测试轨迹
        test_trajectory = []
        for i in range(200):  # 更长的轨迹
            t = i * 0.05
            x = 2.0 + 0.8 * t
            y = 2.0 + 2.0 * np.sin(0.2 * t)
            theta = 0.3 * t
            test_trajectory.append(np.array([x, y, theta, t]))
        
        print(f"测试轨迹长度: {len(test_trajectory)}")
        
        # 测试原始版本
        start_time = time.time()
        try:
            original_result = sva_original.compute_detailed_swept_volume(test_trajectory)
            original_time = time.time() - start_time
            original_success = True
        except Exception as e:
            print(f"原始版本失败: {e}")
            original_time = float('inf')
            original_success = False
            original_result = None
        
        # 测试优化版本
        start_time = time.time()
        try:
            optimized_result = sva_optimized.compute_detailed_swept_volume_optimized(test_trajectory)
            optimized_time = time.time() - start_time
            optimized_success = True
        except Exception as e:
            print(f"优化版本失败: {e}")
            optimized_time = float('inf')
            optimized_success = False
            optimized_result = None
        
        # 比较结果
        area_diff = 0.0
        if original_success and optimized_success:
            if original_result and optimized_result:
                area_diff = abs(original_result.get('area', 0) - optimized_result.area)
        
        # 性能指标
        performance_metrics = sva_optimized.get_performance_metrics_detailed()
        speedup = original_time / optimized_time if optimized_time > 0 else float('inf')
        
        results = {
            'original_time': original_time,
            'optimized_time': optimized_time,
            'original_success': original_success,
            'optimized_success': optimized_success,
            'speedup': speedup,
            'area_difference': area_diff,
            'trajectory_length': len(test_trajectory),
            'performance_metrics': performance_metrics.__dict__ if hasattr(performance_metrics, '__dict__') else {}
        }
        
        print(f"原始版本: {'成功' if original_success else '失败'} ({original_time:.3f}s)")
        print(f"优化版本: {'成功' if optimized_success else '失败'} ({optimized_time:.3f}s)")
        if optimized_success and original_success:
            print(f"性能提升: {speedup:.1f}x")
            print(f"面积差异: {area_diff:.6f} m²")
        
        self.test_results['swept_volume_analyzer'] = results
        return results
    
    def test_svsdf_planner_optimized(self) -> Dict[str, Any]:
        """测试优化的SVSDF规划器"""
        print("\n=== 测试SVSDF规划器优化版本 ===")
        
        # 创建优化版本（原始版本可能太慢）
        svsdf_optimized = SVSDFPlannerOptimized(
            map_width=self.map_size[0],
            map_height=self.map_size[1],
            resolution=self.resolution,
            robot_length=self.robot_length,
            robot_width=self.robot_width
        )
        
        # 添加障碍物
        for obs in self.obstacles:
            svsdf_optimized.add_obstacle(obs['center'], obs['radius'])
        
        # 规划参数
        start = [2.0, 2.0, 0.0]
        goal = [18.0, 16.0, np.pi/2]
        
        print(f"起点: {start}")
        print(f"终点: {goal}")
        
        # 测试优化版本
        start_time = time.time()
        try:
            result = svsdf_optimized.plan_trajectory_optimized(start, goal)
            planning_time = time.time() - start_time
            planning_success = result is not None and result.get('success', False)
        except Exception as e:
            print(f"优化版本失败: {e}")
            planning_time = float('inf')
            planning_success = False
            result = None
        
        # 分析结果
        quality_metrics = {}
        if planning_success and result:
            trajectory = result.get('trajectory', [])
            if trajectory:
                # 计算轨迹质量
                path_length = 0.0
                for i in range(1, len(trajectory)):
                    path_length += np.linalg.norm(
                        np.array(trajectory[i][:2]) - np.array(trajectory[i-1][:2])
                    )
                
                direct_distance = np.linalg.norm(
                    np.array(goal[:2]) - np.array(start[:2])
                )
                
                quality_metrics = {
                    'path_length': path_length,
                    'direct_distance': direct_distance,
                    'efficiency': direct_distance / path_length if path_length > 0 else 0.0,
                    'trajectory_points': len(trajectory),
                    'swept_volume': result.get('swept_volume', 0.0)
                }
        
        results = {
            'planning_time': planning_time,
            'planning_success': planning_success,
            'quality_metrics': quality_metrics,
            'optimization_results': result.get('optimization_info', {}) if result else {}
        }
        
        print(f"规划结果: {'成功' if planning_success else '失败'} ({planning_time:.3f}s)")
        if quality_metrics:
            print(f"轨迹质量指标:")
            for key, value in quality_metrics.items():
                print(f"  {key}: {value:.3f}")
        
        self.test_results['svsdf_planner'] = results
        return results
    
    def test_parallel_performance(self) -> Dict[str, Any]:
        """测试并行性能"""
        print("\n=== 测试并行性能 ===")
        
        # 创建带并行支持的组件
        sva_parallel = SweptVolumeAnalyzerOptimized(
            self.robot_length, self.robot_width,
            enable_parallel=True, max_workers=4
        )
        
        sva_serial = SweptVolumeAnalyzerOptimized(
            self.robot_length, self.robot_width,
            enable_parallel=False
        )
        
        # 生成大型测试数据
        large_trajectory = []
        for i in range(1000):  # 大轨迹
            t = i * 0.02
            x = 2.0 + 0.1 * t
            y = 2.0 + 3.0 * np.sin(0.1 * t)
            theta = 0.2 * t
            large_trajectory.append(np.array([x, y, theta, t]))
        
        print(f"大型轨迹长度: {len(large_trajectory)}")
        
        # 测试串行版本
        start_time = time.time()
        try:
            serial_result = sva_serial.compute_detailed_swept_volume_optimized(large_trajectory)
            serial_time = time.time() - start_time
            serial_success = True
        except Exception as e:
            print(f"串行版本失败: {e}")
            serial_time = float('inf')
            serial_success = False
        
        # 测试并行版本
        start_time = time.time()
        try:
            parallel_result = sva_parallel.compute_detailed_swept_volume_optimized(large_trajectory)
            parallel_time = time.time() - start_time
            parallel_success = True
        except Exception as e:
            print(f"并行版本失败: {e}")
            parallel_time = float('inf')
            parallel_success = False
        
        # 计算并行加速比
        parallel_speedup = serial_time / parallel_time if parallel_time > 0 else float('inf')
        
        results = {
            'serial_time': serial_time,
            'parallel_time': parallel_time,
            'parallel_speedup': parallel_speedup,
            'serial_success': serial_success,
            'parallel_success': parallel_success,
            'trajectory_size': len(large_trajectory)
        }
        
        print(f"串行版本: {'成功' if serial_success else '失败'} ({serial_time:.3f}s)")
        print(f"并行版本: {'成功' if parallel_success else '失败'} ({parallel_time:.3f}s)")
        if parallel_success and serial_success:
            print(f"并行加速比: {parallel_speedup:.1f}x")
        
        self.test_results['parallel_performance'] = results
        return results
    
    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试"""
        print("开始运行SVSDF优化组件综合测试套件...")
        print("=" * 60)
        
        all_results = {}
        
        # 依次运行各项测试
        try:
            all_results['sdf_calculator'] = self.test_sdf_calculator_optimized()
        except Exception as e:
            print(f"SDF计算器测试失败: {e}")
            all_results['sdf_calculator'] = {'error': str(e)}
        
        try:
            all_results['minco_trajectory'] = self.test_minco_trajectory_optimized()
        except Exception as e:
            print(f"MINCO轨迹测试失败: {e}")
            all_results['minco_trajectory'] = {'error': str(e)}
        
        try:
            all_results['mpc_controller'] = self.test_mpc_controller_optimized()
        except Exception as e:
            print(f"MPC控制器测试失败: {e}")
            all_results['mpc_controller'] = {'error': str(e)}
        
        try:
            all_results['swept_volume_analyzer'] = self.test_swept_volume_analyzer_optimized()
        except Exception as e:
            print(f"扫掠体积分析器测试失败: {e}")
            all_results['swept_volume_analyzer'] = {'error': str(e)}
        
        try:
            all_results['svsdf_planner'] = self.test_svsdf_planner_optimized()
        except Exception as e:
            print(f"SVSDF规划器测试失败: {e}")
            all_results['svsdf_planner'] = {'error': str(e)}
        
        try:
            all_results['parallel_performance'] = self.test_parallel_performance()
        except Exception as e:
            print(f"并行性能测试失败: {e}")
            all_results['parallel_performance'] = {'error': str(e)}
        
        # 生成测试报告
        self.generate_test_report(all_results)
        
        return all_results
    
    def generate_test_report(self, results: Dict[str, Any]):
        """生成测试报告"""
        print("\n" + "=" * 60)
        print("SVSDF优化组件测试报告")
        print("=" * 60)
        
        # 汇总性能提升
        total_speedups = []
        
        for component, result in results.items():
            if 'error' in result:
                print(f"\n{component}: 测试失败 - {result['error']}")
                continue
                
            print(f"\n{component}:")
            
            if 'speedup' in result:
                speedup = result['speedup']
                if speedup != float('inf'):
                    print(f"  性能提升: {speedup:.1f}x")
                    total_speedups.append(speedup)
                else:
                    print(f"  性能提升: 显著")
            
            if 'original_time' in result and 'optimized_time' in result:
                print(f"  原始时间: {result['original_time']:.3f}s")
                print(f"  优化时间: {result['optimized_time']:.3f}s")
            
            if 'quality_metrics' in result and result['quality_metrics']:
                print(f"  质量指标: {len(result['quality_metrics'])}项")
        
        # 整体性能评估
        if total_speedups:
            avg_speedup = np.mean(total_speedups)
            max_speedup = max(total_speedups)
            print(f"\n整体性能评估:")
            print(f"  平均性能提升: {avg_speedup:.1f}x")
            print(f"  最大性能提升: {max_speedup:.1f}x")
            print(f"  测试成功的组件数: {len(total_speedups)}")
        
        print(f"\n测试完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
    
def main():
    """主测试函数"""
    print("SVSDF优化组件综合测试套件")
    print("版本: 1.0")
    print("作者: SVSDF开发团队")
    print()
    
    # 创建测试器
    tester = OptimizedComponentsTester()
    
    # 运行所有测试
    results = tester.run_all_tests()
    
    # 保存结果到文件
    import json
    
    # 转换numpy类型以便JSON序列化
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    serializable_results = convert_numpy(results)
    
    with open('test_results.json', 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n测试结果已保存到: test_results.json")
    
    return results

if __name__ == "__main__":
    main()
