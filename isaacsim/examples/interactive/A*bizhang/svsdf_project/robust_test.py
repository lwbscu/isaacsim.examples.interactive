#!/usr/bin/env python3
"""
稳定的组件测试 - 带完整错误处理
"""
import os
import sys
import traceback
import time

# 强制刷新输出
def flush_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def safe_test_imports():
    """安全的导入测试"""
    flush_print("开始导入测试...")
    results = {}
    
    # 测试numpy
    try:
        import numpy as np
        results['numpy'] = True
        flush_print("✓ numpy 导入成功")
    except Exception as e:
        results['numpy'] = False
        flush_print(f"✗ numpy 导入失败: {e}")
    
    # 测试config
    try:
        from utils.config import config
        results['config'] = True
        flush_print(f"✓ config 导入成功 (robot length: {config.robot.length})")
    except Exception as e:
        results['config'] = False
        flush_print(f"✗ config 导入失败: {e}")
        traceback.print_exc()
    
    # 测试math_utils
    try:
        from utils.math_utils import MathUtils
        dist = MathUtils.euclidean_distance([0, 0], [3, 4])
        results['math_utils'] = True
        flush_print(f"✓ math_utils 导入成功 (distance test: {dist})")
    except Exception as e:
        results['math_utils'] = False
        flush_print(f"✗ math_utils 导入失败: {e}")
        traceback.print_exc()
    
    # 测试sdf_calculator
    try:
        from core.sdf_calculator import SDFCalculator
        results['sdf_calculator'] = True
        flush_print("✓ sdf_calculator 导入成功")
    except Exception as e:
        results['sdf_calculator'] = False
        flush_print(f"✗ sdf_calculator 导入失败: {e}")
        traceback.print_exc()
    
    # 测试swept_volume_analyzer
    try:
        from core.swept_volume_analyzer import SweptVolumeAnalyzer
        results['swept_volume_analyzer'] = True
        flush_print("✓ swept_volume_analyzer 导入成功")
    except Exception as e:
        results['swept_volume_analyzer'] = False
        flush_print(f"✗ swept_volume_analyzer 导入失败: {e}")
        traceback.print_exc()
    
    # 测试swept_volume_analyzer_optimized
    try:
        from core.swept_volume_analyzer_optimized import SweptVolumeAnalyzerOptimized
        results['swept_volume_analyzer_optimized'] = True
        flush_print("✓ swept_volume_analyzer_optimized 导入成功")
    except Exception as e:
        results['swept_volume_analyzer_optimized'] = False
        flush_print(f"✗ swept_volume_analyzer_optimized 导入失败: {e}")
        traceback.print_exc()
    
    # 测试astar_planner
    try:
        from core.astar_planner import AStarPlanner
        results['astar_planner'] = True
        flush_print("✓ astar_planner 导入成功")
    except Exception as e:
        results['astar_planner'] = False
        flush_print(f"✗ astar_planner 导入失败: {e}")
        traceback.print_exc()
    
    return results

def safe_test_basic_sdf():
    """安全的SDF测试"""
    flush_print("\n开始SDF基础功能测试...")
    
    try:
        from core.sdf_calculator import SDFCalculator
        import numpy as np
        
        # 创建SDF计算器
        sdf_calc = SDFCalculator(robot_length=0.6, robot_width=0.4)
        flush_print("✓ SDF计算器创建成功")
        
        # 设置障碍物
        obstacles = [{'type': 'circle', 'center': [2, 2], 'radius': 0.5}]
        sdf_calc.set_obstacles(obstacles)
        flush_print("✓ 障碍物设置成功")
        
        # 测试SDF计算
        sdf_value = sdf_calc.compute_sdf(np.array([0, 0]))
        flush_print(f"✓ SDF计算成功，远点SDF值: {sdf_value:.3f}")
        
        # 测试机器人SDF
        robot_pose = np.array([0, 0, 0])
        robot_sdf = sdf_calc.compute_robot_sdf(np.array([0, 0]), robot_pose)
        flush_print(f"✓ 机器人SDF计算成功，中心点SDF值: {robot_sdf:.3f}")
        
        return True
        
    except Exception as e:
        flush_print(f"✗ SDF测试失败: {e}")
        traceback.print_exc()
        return False

def safe_test_swept_volume():
    """安全的扫掠体积测试"""
    flush_print("\n开始扫掠体积测试...")
    
    try:
        from core.swept_volume_analyzer import SweptVolumeAnalyzer
        import numpy as np
        
        # 创建分析器
        analyzer = SweptVolumeAnalyzer(robot_length=0.6, robot_width=0.4)
        flush_print("✓ 扫掠体积分析器创建成功")
        
        # 创建简单轨迹
        trajectory = []
        for i in range(10):
            x = i * 0.1
            y = 0.0
            theta = 0.0
            t = i * 0.1
            trajectory.append(np.array([x, y, theta, t]))
        
        flush_print(f"✓ 创建轨迹成功 ({len(trajectory)} 个点)")
        
        # 测试边界计算
        boundary = analyzer.compute_swept_volume_boundary(trajectory)
        flush_print(f"✓ 边界计算成功 ({len(boundary)} 个边界点)")
        
        # 测试面积计算
        area = analyzer.compute_swept_volume_area(trajectory)
        flush_print(f"✓ 面积计算成功: {area:.3f} m²")
        
        return True
        
    except Exception as e:
        flush_print(f"✗ 扫掠体积测试失败: {e}")
        traceback.print_exc()
        return False

def safe_test_optimized_swept_volume():
    """安全的优化扫掠体积测试"""
    flush_print("\n开始优化扫掠体积测试...")
    
    try:
        from core.swept_volume_analyzer_optimized import SweptVolumeAnalyzerOptimized
        import numpy as np
        
        # 创建优化分析器
        analyzer = SweptVolumeAnalyzerOptimized(robot_length=0.6, robot_width=0.4)
        flush_print("✓ 优化扫掠体积分析器创建成功")
        
        # 创建轨迹
        trajectory = []
        for i in range(20):
            t = i * 0.1
            x = t
            y = 0.2 * np.sin(t)
            theta = 0.1 * t
            trajectory.append(np.array([x, y, theta, t]))
        
        flush_print(f"✓ 创建复杂轨迹成功 ({len(trajectory)} 个点)")
        
        # 测试详细计算
        start_time = time.time()
        result = analyzer.compute_detailed_swept_volume(trajectory, grid_resolution=0.1)
        computation_time = time.time() - start_time
        
        flush_print(f"✓ 详细计算成功:")
        flush_print(f"  - 面积: {result.area:.3f} m²")
        flush_print(f"  - 计算时间: {computation_time*1000:.1f} ms")
        flush_print(f"  - 网格分辨率: {result.grid_resolution}")
        
        if result.quality_metrics:
            flush_print(f"  - 质量分数: {result.quality_metrics.get('overall_quality', 0):.3f}")
        
        # 测试性能报告
        report = analyzer.get_performance_report()
        flush_print(f"✓ 性能报告:")
        flush_print(f"  - 总计算次数: {report['metrics']['total_computations']}")
        flush_print(f"  - 缓存命中率: {report['metrics']['cache_hit_rate']:.2f}")
        flush_print(f"  - 并行工作线程: {report['configuration']['max_workers']}")
        
        return True
        
    except Exception as e:
        flush_print(f"✗ 优化扫掠体积测试失败: {e}")
        traceback.print_exc()
        return False

def safe_test_performance_comparison():
    """安全的性能对比测试"""
    flush_print("\n开始性能对比测试...")
    
    try:
        from core.swept_volume_analyzer import SweptVolumeAnalyzer
        from core.swept_volume_analyzer_optimized import SweptVolumeAnalyzerOptimized
        import numpy as np
        import time
        
        # 创建测试轨迹
        trajectory = []
        for i in range(50):
            t = i * 0.05
            x = 2 * t
            y = np.sin(t) + 0.5 * np.sin(3 * t)
            theta = 0.2 * t
            trajectory.append(np.array([x, y, theta, t]))
        
        flush_print(f"✓ 创建性能测试轨迹 ({len(trajectory)} 个点)")
        
        # 原始版本测试
        original = SweptVolumeAnalyzer(robot_length=0.6, robot_width=0.4)
        start_time = time.time()
        original_area = original.compute_swept_volume_area(trajectory)
        original_time = time.time() - start_time
        
        flush_print(f"✓ 原始版本:")
        flush_print(f"  - 时间: {original_time*1000:.1f} ms")
        flush_print(f"  - 面积: {original_area:.3f} m²")
        
        # 优化版本测试
        optimized = SweptVolumeAnalyzerOptimized(robot_length=0.6, robot_width=0.4)
        start_time = time.time()
        optimized_result = optimized.compute_detailed_swept_volume(trajectory, compute_quality=False)
        optimized_time = time.time() - start_time
        
        flush_print(f"✓ 优化版本:")
        flush_print(f"  - 时间: {optimized_time*1000:.1f} ms")
        flush_print(f"  - 面积: {optimized_result.area:.3f} m²")
        
        # 性能分析
        if optimized_time > 0:
            speedup = original_time / optimized_time
            flush_print(f"✓ 性能分析:")
            flush_print(f"  - 加速比: {speedup:.2f}x")
        
        if original_area > 0:
            accuracy = 1.0 - abs(original_area - optimized_result.area) / original_area
            flush_print(f"  - 精度: {accuracy:.3f}")
        
        return True
        
    except Exception as e:
        flush_print(f"✗ 性能对比测试失败: {e}")
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    flush_print("🚀 SVSDF优化组件稳定测试")
    flush_print("=" * 60)
    
    test_results = {
        'imports': None,
        'sdf_basic': False,
        'swept_volume': False,
        'optimized_swept_volume': False,
        'performance_comparison': False
    }
    
    # 导入测试
    flush_print("\n📦 模块导入测试")
    flush_print("-" * 30)
    test_results['imports'] = safe_test_imports()
    
    # 只有在关键模块导入成功时才继续
    if (test_results['imports'].get('config', False) and 
        test_results['imports'].get('math_utils', False)):
        
        # SDF基础测试
        flush_print("\n🔍 SDF基础功能测试")
        flush_print("-" * 30)
        test_results['sdf_basic'] = safe_test_basic_sdf()
        
        # 扫掠体积测试
        flush_print("\n📐 扫掠体积测试")
        flush_print("-" * 30)
        test_results['swept_volume'] = safe_test_swept_volume()
        
        # 优化扫掠体积测试
        flush_print("\n⚡ 优化扫掠体积测试")
        flush_print("-" * 30)
        test_results['optimized_swept_volume'] = safe_test_optimized_swept_volume()
        
        # 性能对比测试
        flush_print("\n🏁 性能对比测试")
        flush_print("-" * 30)
        test_results['performance_comparison'] = safe_test_performance_comparison()
    
    else:
        flush_print("\n❌ 关键模块导入失败，跳过后续测试")
    
    # 测试总结
    flush_print("\n" + "=" * 60)
    flush_print("📊 测试总结")
    flush_print("-" * 30)
    
    total_tests = 0
    passed_tests = 0
    
    # 导入测试统计
    if test_results['imports']:
        import_passed = sum(1 for v in test_results['imports'].values() if v)
        import_total = len(test_results['imports'])
        flush_print(f"导入测试: {import_passed}/{import_total} 通过")
        total_tests += import_total
        passed_tests += import_passed
    
    # 功能测试统计
    functional_tests = ['sdf_basic', 'swept_volume', 'optimized_swept_volume', 'performance_comparison']
    for test_name in functional_tests:
        if test_results[test_name]:
            flush_print(f"{test_name}: ✓ 通过")
            passed_tests += 1
        else:
            flush_print(f"{test_name}: ✗ 失败")
        total_tests += 1
    
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    flush_print(f"\n总体成功率: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        flush_print("🎉 测试总体通过!")
        return 0
    else:
        flush_print("❌ 测试未达到预期")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        flush_print(f"\n程序即将退出，退出码: {exit_code}")
        time.sleep(1)  # 给用户时间看到结果
        sys.exit(exit_code)
    except Exception as e:
        flush_print(f"\n程序异常退出: {e}")
        traceback.print_exc()
        time.sleep(2)
        sys.exit(1)
