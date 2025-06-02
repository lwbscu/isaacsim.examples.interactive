#!/usr/bin/env python3
"""
核心组件测试（避免可视化模块）
"""
import os
import sys

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_core_components():
    """测试核心组件"""
    print("🚀 SVSDF核心组件测试")
    print("=" * 50)
    
    success = True
    
    try:
        # 测试SDF计算器
        print("测试SDF计算器...")
        from core.sdf_calculator import SDFCalculator
        from core.sdf_calculator_optimized import SDFCalculatorOptimized
        
        sdf_calc = SDFCalculator(robot_length=0.6, robot_width=0.4)
        sdf_calc_opt = SDFCalculatorOptimized(robot_length=0.6, robot_width=0.4)
        print("✓ SDF计算器创建成功")
        
        # 测试轨迹优化器
        print("\n测试MINCO轨迹优化器...")
        from core.minco_trajectory import MINCOTrajectory
        from core.minco_trajectory_optimized import MINCOTrajectoryOptimized
        
        minco = MINCOTrajectory()
        minco_opt = MINCOTrajectoryOptimized()
        print("✓ MINCO轨迹优化器创建成功")
        
        # 测试MPC控制器
        print("\n测试MPC控制器...")
        from core.mpc_controller import MPCController
        from core.mpc_controller_optimized import MPCControllerOptimized
        
        mpc = MPCController()
        mpc_opt = MPCControllerOptimized()
        print("✓ MPC控制器创建成功")
        
        # 测试A*规划器
        print("\n测试A*规划器...")
        from core.astar_planner import AStarPlanner
        
        # 创建简单地图
        import numpy as np
        grid_map = np.zeros((50, 50))
        # 添加一些障碍物
        grid_map[20:30, 20:30] = 1
        
        planner = AStarPlanner(grid_map, resolution=0.1)
        print("✓ A*规划器创建成功")
        
        # 测试路径规划
        start = (5, 5)
        goal = (45, 45)
        path = planner.plan(start, goal)
        if path:
            print(f"✓ 路径规划成功，路径长度: {len(path)}")
        else:
            print("✗ 路径规划失败")
            success = False
        
    except Exception as e:
        print(f"✗ 核心组件测试失败: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    return success

def test_performance_comparison():
    """性能对比测试"""
    print("\n" + "=" * 50)
    print("性能对比测试")
    print("=" * 50)
    
    try:
        import numpy as np
        import time
        from core.swept_volume_analyzer import SweptVolumeAnalyzer
        from core.swept_volume_analyzer_optimized import SweptVolumeAnalyzerOptimized
        
        # 创建复杂轨迹
        trajectory = []
        for i in range(100):
            t = i * 0.05
            x = t
            y = 0.5 * np.sin(t * 2)
            theta = 0.1 * t
            trajectory.append(np.array([x, y, theta, t]))
        
        print(f"创建测试轨迹: {len(trajectory)} 个点")
        
        # 原始版本性能测试
        original = SweptVolumeAnalyzer(robot_length=0.6, robot_width=0.4)
        
        times_original = []
        for _ in range(5):
            start_time = time.time()
            area_original = original.compute_swept_volume_area(trajectory)
            times_original.append(time.time() - start_time)
        
        avg_time_original = np.mean(times_original)
        
        # 优化版本性能测试
        optimized = SweptVolumeAnalyzerOptimized(robot_length=0.6, robot_width=0.4)
        
        times_optimized = []
        for _ in range(5):
            start_time = time.time()
            result_optimized = optimized.compute_detailed_swept_volume_optimized(trajectory)
            times_optimized.append(time.time() - start_time)
        
        avg_time_optimized = np.mean(times_optimized)
        area_optimized = result_optimized.area
        
        # 结果分析
        speedup = avg_time_original / avg_time_optimized if avg_time_optimized > 0 else 1.0
        accuracy = 1.0 - abs(area_original - area_optimized) / area_original if area_original > 0 else 1.0
        
        print(f"\n性能对比结果:")
        print(f"  原始版本: {avg_time_original*1000:.1f} ms, 面积: {area_original:.3f} m²")
        print(f"  优化版本: {avg_time_optimized*1000:.1f} ms, 面积: {area_optimized:.3f} m²")
        print(f"  加速比: {speedup:.2f}x")
        print(f"  精度: {accuracy:.3f}")
        
        if speedup > 1.5 and accuracy > 0.95:
            print("✓ 性能优化成功！")
            return True
        else:
            print("✗ 性能优化不足")
            return False
            
    except Exception as e:
        print(f"✗ 性能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    success = True
    
    # 核心组件测试
    if not test_core_components():
        success = False
    
    # 性能对比测试
    if not test_performance_comparison():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 所有核心组件测试通过!")
        return 0
    else:
        print("❌ 部分测试失败")
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
