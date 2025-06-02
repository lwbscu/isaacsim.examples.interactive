#!/usr/bin/env python3
"""
简单的组件测试
"""
import os
import sys

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_imports():
    """测试基本导入"""
    print("开始测试基本导入...")
    
    try:
        import numpy as np
        print("✓ numpy")
    except Exception as e:
        print(f"✗ numpy: {e}")
        return False
    
    try:
        from utils.config import config
        print(f"✓ config (robot length: {config.robot.length})")
    except Exception as e:
        print(f"✗ config: {e}")
        return False
    
    try:
        from utils.math_utils import MathUtils
        dist = MathUtils.euclidean_distance([0, 0], [3, 4])
        print(f"✓ math_utils (distance test: {dist})")
    except Exception as e:
        print(f"✗ math_utils: {e}")
        return False
    
    try:
        from core.sdf_calculator import SDFCalculator
        print("✓ sdf_calculator")
    except Exception as e:
        print(f"✗ sdf_calculator: {e}")
        return False
    
    try:
        from core.swept_volume_analyzer import SweptVolumeAnalyzer
        print("✓ swept_volume_analyzer")
    except Exception as e:
        print(f"✗ swept_volume_analyzer: {e}")
        return False
    
    try:
        from core.swept_volume_analyzer_optimized import SweptVolumeAnalyzerOptimized
        print("✓ swept_volume_analyzer_optimized")
    except Exception as e:
        print(f"✗ swept_volume_analyzer_optimized: {e}")
        return False
    
    try:
        from core.astar_planner import AStarPlanner
        print("✓ astar_planner")
    except Exception as e:
        print(f"✗ astar_planner: {e}")
        return False
    
    print("所有基本导入测试完成!")
    return True

def test_basic_functionality():
    """测试基本功能"""
    print("\n开始测试基本功能...")
    
    try:
        import numpy as np
        from core.swept_volume_analyzer_optimized import SweptVolumeAnalyzerOptimized
        
        # 创建分析器
        analyzer = SweptVolumeAnalyzerOptimized(robot_length=0.6, robot_width=0.4)
        print("✓ 创建优化分析器")
        
        # 创建简单轨迹
        trajectory = []
        for i in range(10):
            x = i * 0.1
            y = 0.0
            theta = 0.0
            t = i * 0.1
            trajectory.append(np.array([x, y, theta, t]))
        
        print(f"✓ 创建轨迹 ({len(trajectory)} 个点)")
        
        # 测试边界计算
        boundary = analyzer.compute_swept_volume_boundary_optimized(trajectory)
        print(f"✓ 边界计算 ({len(boundary)} 个边界点)")
        
        # 测试详细计算
        result = analyzer.compute_detailed_swept_volume_optimized(trajectory, grid_resolution=0.1)
        print(f"✓ 详细计算 (面积: {result.area:.3f} m²)")
        
        # 测试性能报告
        report = analyzer.get_performance_metrics_detailed()
        cache_hit_rate = getattr(report, 'cache_hit_rate', 0.0)
        print(f"✓ 性能报告 (缓存命中率: {cache_hit_rate:.2f})")
        
        print("基本功能测试完成!")
        return True
        
    except Exception as e:
        print(f"✗ 功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance():
    """性能测试"""
    print("\n开始性能测试...")
    
    try:
        import numpy as np
        import time
        from core.swept_volume_analyzer import SweptVolumeAnalyzer
        from core.swept_volume_analyzer_optimized import SweptVolumeAnalyzerOptimized
        
        # 创建复杂轨迹
        trajectory = []
        for i in range(50):
            t = i * 0.1
            x = t
            y = 0.5 * np.sin(t)
            theta = 0.1 * t
            trajectory.append(np.array([x, y, theta, t]))
        
        print(f"✓ 创建复杂轨迹 ({len(trajectory)} 个点)")
        
        # 原始版本
        original = SweptVolumeAnalyzer(robot_length=0.6, robot_width=0.4)
        start_time = time.time()
        original_area = original.compute_swept_volume_area(trajectory)
        original_time = time.time() - start_time
        
        print(f"✓ 原始版本: {original_time*1000:.1f} ms, 面积: {original_area:.3f} m²")
        
        # 优化版本
        optimized = SweptVolumeAnalyzerOptimized(robot_length=0.6, robot_width=0.4)
        start_time = time.time()
        optimized_result = optimized.compute_detailed_swept_volume_optimized(trajectory)
        optimized_time = time.time() - start_time
        
        print(f"✓ 优化版本: {optimized_time*1000:.1f} ms, 面积: {optimized_result.area:.3f} m²")
        
        # 计算加速比
        speedup = original_time / optimized_time if optimized_time > 0 else 1.0
        print(f"✓ 加速比: {speedup:.2f}x")
        
        # 精度比较
        accuracy = 1.0 - abs(original_area - optimized_result.area) / original_area if original_area > 0 else 1.0
        print(f"✓ 精度: {accuracy:.3f}")
        
        print("性能测试完成!")
        return True
        
    except Exception as e:
        print(f"✗ 性能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 SVSDF优化组件简单测试")
    print("=" * 50)
    
    success = True
    
    # 基本导入测试
    if not test_imports():
        success = False
    
    # 基本功能测试
    if not test_basic_functionality():
        success = False
    
    # 性能测试
    if not test_performance():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 所有测试通过!")
        return 0
    else:
        print("❌ 部分测试失败")
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
