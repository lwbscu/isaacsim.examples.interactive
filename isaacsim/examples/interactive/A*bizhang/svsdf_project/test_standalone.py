#!/usr/bin/env python3
"""
独立的优化组件测试套件
不依赖 Isaac Sim，专注于测试算法性能
"""
import sys
import os
import time
import json
import numpy as np
from typing import Dict, List, Any
import traceback

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# 测试结果
test_results = {
    'summary': {
        'total_tests': 0,
        'passed': 0,
        'failed': 0,
        'skipped': 0
    },
    'tests': [],
    'performance': {},
    'errors': []
}

def log_test(test_name: str, status: str, details: Dict = None, error: str = None):
    """记录测试结果"""
    test_results['summary']['total_tests'] += 1
    test_results['summary'][status] += 1
    
    test_info = {
        'name': test_name,
        'status': status,
        'timestamp': time.time(),
        'details': details or {},
        'error': error
    }
    test_results['tests'].append(test_info)
    
    status_symbol = {'passed': '✓', 'failed': '✗', 'skipped': '-'}[status]
    print(f"{status_symbol} {test_name}")
    if error:
        print(f"  错误: {error}")

def test_math_utils():
    """测试数学工具类"""
    try:
        from utils.math_utils import MathUtils, GeometryUtils
        
        # 测试基本数学函数
        assert abs(MathUtils.normalize_angle(np.pi + 0.1) - (-np.pi + 0.1)) < 1e-6
        assert MathUtils.euclidean_distance([0, 0], [3, 4]) == 5.0
        
        # 测试几何函数
        points = [[0, 0], [1, 0], [1, 1], [0, 1]]
        area = GeometryUtils.polygon_area(points)
        assert abs(area - 1.0) < 1e-6
        
        log_test("MathUtils基础功能", "passed", {
            'euclidean_distance': True,
            'normalize_angle': True,
            'polygon_area': True
        })
        
    except Exception as e:
        log_test("MathUtils基础功能", "failed", error=str(e))

def test_config():
    """测试配置加载"""
    try:
        from utils.config import config
        
        # 验证配置参数
        assert hasattr(config.robot, 'length')
        assert hasattr(config.robot, 'width')
        assert hasattr(config.robot, 'mass')
        assert hasattr(config.planning, 'grid_resolution')
        
        log_test("配置加载", "passed", {
            'robot_config': True,
            'planning_config': True,
            'robot_length': config.robot.length,
            'robot_width': config.robot.width
        })
        
    except Exception as e:
        log_test("配置加载", "failed", error=str(e))

def test_sdf_calculator():
    """测试SDF计算器"""
    try:
        from core.sdf_calculator import SDFCalculator
        
        sdf_calc = SDFCalculator(robot_length=0.6, robot_width=0.4)
        
        # 测试基本SDF计算
        obstacles = [{'type': 'circle', 'center': [2, 2], 'radius': 0.5}]
        sdf_calc.set_obstacles(obstacles)
        
        # 测试远离障碍物的点
        sdf_far = sdf_calc.compute_sdf(np.array([0, 0]))
        assert sdf_far > 0, "远离障碍物的点SDF应为正值"
        
        # 测试机器人SDF
        robot_pose = np.array([0, 0, 0])
        sdf_robot = sdf_calc.compute_robot_sdf(np.array([0, 0]), robot_pose)
        assert sdf_robot <= 0, "机器人中心点SDF应为负值或零"
        
        log_test("SDF计算器", "passed", {
            'obstacle_sdf': True,
            'robot_sdf': True,
            'sdf_far_value': float(sdf_far)
        })
        
    except Exception as e:
        log_test("SDF计算器", "failed", error=str(e))

def test_swept_volume_analyzer():
    """测试扫掠体积分析器"""
    try:
        from core.swept_volume_analyzer import SweptVolumeAnalyzer
        
        analyzer = SweptVolumeAnalyzer(robot_length=0.6, robot_width=0.4)
        
        # 创建简单轨迹
        trajectory = []
        for i in range(10):
            x = i * 0.1
            y = 0.0
            theta = 0.0
            t = i * 0.1
            trajectory.append(np.array([x, y, theta, t]))
        
        # 测试边界计算
        start_time = time.time()
        boundary = analyzer.compute_swept_volume_boundary(trajectory)
        boundary_time = time.time() - start_time
        
        assert len(boundary) >= 4, "边界点数量应至少为4"
        
        # 测试面积计算
        start_time = time.time()
        area = analyzer.compute_swept_volume_area(trajectory)
        area_time = time.time() - start_time
        
        assert area > 0, "扫掠体积面积应为正值"
        
        log_test("扫掠体积分析器", "passed", {
            'boundary_points': len(boundary),
            'swept_area': float(area),
            'boundary_time_ms': boundary_time * 1000,
            'area_time_ms': area_time * 1000
        })
        
    except Exception as e:
        log_test("扫掠体积分析器", "failed", error=str(e))

def test_optimized_swept_volume_analyzer():
    """测试优化版扫掠体积分析器"""
    try:
        from core.swept_volume_analyzer_optimized import SweptVolumeAnalyzerOptimized
        
        analyzer = SweptVolumeAnalyzerOptimized(robot_length=0.6, robot_width=0.4)
        
        # 创建更复杂的轨迹
        trajectory = []
        for i in range(50):
            t = i * 0.1
            x = t
            y = 0.5 * np.sin(t)  # 正弦波轨迹
            theta = 0.1 * t
            trajectory.append(np.array([x, y, theta, t]))
        
        # 测试详细扫掠体积计算
        start_time = time.time()
        result = analyzer.compute_detailed_swept_volume(
            trajectory, grid_resolution=0.1, compute_quality=True)
        computation_time = time.time() - start_time
        
        assert result.area > 0, "优化版扫掠体积面积应为正值"
        assert result.density_grid is not None, "密度网格应存在"
        assert len(result.quality_metrics) > 0, "质量指标应存在"
        
        # 测试性能报告
        report = analyzer.get_performance_report()
        
        log_test("优化版扫掠体积分析器", "passed", {
            'swept_area': float(result.area),
            'computation_time_ms': computation_time * 1000,
            'grid_resolution': result.grid_resolution,
            'quality_metrics': result.quality_metrics,
            'cache_hit_rate': report['metrics']['cache_hit_rate'],
            'parallel_workers': report['configuration']['max_workers']
        })
        
        # 存储性能数据
        test_results['performance']['optimized_swept_volume'] = {
            'computation_time_ms': computation_time * 1000,
            'area': float(result.area),
            'quality_score': result.quality_metrics.get('overall_quality', 0.0)
        }
        
    except Exception as e:
        log_test("优化版扫掠体积分析器", "failed", error=str(e))

def test_performance_comparison():
    """性能对比测试"""
    try:
        from core.swept_volume_analyzer import SweptVolumeAnalyzer
        from core.swept_volume_analyzer_optimized import SweptVolumeAnalyzerOptimized
        
        # 创建测试轨迹
        trajectory = []
        for i in range(100):  # 更长的轨迹
            t = i * 0.05
            x = 2 * t
            y = np.sin(t) + 0.5 * np.sin(3 * t)  # 复杂轨迹
            theta = 0.2 * t
            trajectory.append(np.array([x, y, theta, t]))
        
        # 原始版本测试
        original_analyzer = SweptVolumeAnalyzer(robot_length=0.6, robot_width=0.4)
        start_time = time.time()
        original_area = original_analyzer.compute_swept_volume_area(trajectory)
        original_time = time.time() - start_time
        
        # 优化版本测试
        optimized_analyzer = SweptVolumeAnalyzerOptimized(robot_length=0.6, robot_width=0.4)
        start_time = time.time()
        optimized_result = optimized_analyzer.compute_detailed_swept_volume(
            trajectory, compute_quality=False)
        optimized_time = time.time() - start_time
        
        # 计算性能提升
        speedup = original_time / optimized_time if optimized_time > 0 else 1.0
        area_diff = abs(original_area - optimized_result.area) / original_area if original_area > 0 else 0.0
        
        log_test("性能对比测试", "passed", {
            'original_time_ms': original_time * 1000,
            'optimized_time_ms': optimized_time * 1000,
            'speedup': float(speedup),
            'original_area': float(original_area),
            'optimized_area': float(optimized_result.area),
            'area_difference_ratio': float(area_diff)
        })
        
        # 存储性能对比数据
        test_results['performance']['comparison'] = {
            'speedup': float(speedup),
            'original_time_ms': original_time * 1000,
            'optimized_time_ms': optimized_time * 1000,
            'accuracy': 1.0 - area_diff
        }
        
    except Exception as e:
        log_test("性能对比测试", "failed", error=str(e))

def test_astar_planner():
    """测试A*路径规划器"""
    try:
        from core.astar_planner import AStarPlanner
        
        # 创建简单地图
        grid_map = np.zeros((50, 50))
        # 添加一些障碍物
        grid_map[20:30, 20:30] = 1
        
        planner = AStarPlanner(grid_map, grid_resolution=0.1)
        
        start = (5, 5)
        goal = (45, 45)
        
        start_time = time.time()
        path = planner.plan(start, goal)
        planning_time = time.time() - start_time
        
        assert path is not None, "应该找到路径"
        assert len(path) > 0, "路径不应为空"
        assert path[0] == start, "路径应从起点开始"
        assert path[-1] == goal, "路径应到达终点"
        
        log_test("A*路径规划器", "passed", {
            'path_length': len(path),
            'planning_time_ms': planning_time * 1000,
            'start': start,
            'goal': goal
        })
        
    except Exception as e:
        log_test("A*路径规划器", "failed", error=str(e))

def run_all_tests():
    """运行所有测试"""
    print("🚀 开始运行SVSDF优化组件测试套件")
    print("=" * 60)
    
    # 运行测试
    test_functions = [
        test_math_utils,
        test_config,
        test_sdf_calculator,
        test_swept_volume_analyzer,
        test_optimized_swept_volume_analyzer,
        test_performance_comparison,
        test_astar_planner
    ]
    
    for test_func in test_functions:
        try:
            test_func()
        except Exception as e:
            test_name = test_func.__name__.replace('test_', '').replace('_', ' ').title()
            log_test(test_name, "failed", error=f"测试执行异常: {str(e)}")
            test_results['errors'].append({
                'test': test_name,
                'error': str(e),
                'traceback': traceback.format_exc()
            })
    
    # 输出测试总结
    print("\n" + "=" * 60)
    print("📊 测试总结")
    print(f"总测试数: {test_results['summary']['total_tests']}")
    print(f"通过: {test_results['summary']['passed']}")
    print(f"失败: {test_results['summary']['failed']}")
    print(f"跳过: {test_results['summary']['skipped']}")
    
    success_rate = (test_results['summary']['passed'] / 
                   test_results['summary']['total_tests'] * 100) if test_results['summary']['total_tests'] > 0 else 0
    print(f"成功率: {success_rate:.1f}%")
    
    # 性能总结
    if test_results['performance']:
        print("\n📈 性能总结")
        
        if 'comparison' in test_results['performance']:
            comp = test_results['performance']['comparison']
            print(f"优化版加速比: {comp['speedup']:.2f}x")
            print(f"准确性: {comp['accuracy']:.3f}")
        
        if 'optimized_swept_volume' in test_results['performance']:
            osv = test_results['performance']['optimized_swept_volume']
            print(f"优化版计算时间: {osv['computation_time_ms']:.1f} ms")
            print(f"质量分数: {osv['quality_score']:.3f}")
    
    # 保存详细结果
    result_file = os.path.join(project_root, 'test_results_standalone.json')
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n📁 详细测试结果已保存到: {result_file}")
    
    # 返回状态码
    return 0 if test_results['summary']['failed'] == 0 else 1

if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
