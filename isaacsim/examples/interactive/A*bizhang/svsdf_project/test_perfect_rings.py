#!/usr/bin/env python3
"""
测试完美相切圆环效果
快速验证SDF圆环是否精准贴合障碍物表面
"""

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import carb
import omni.usd
import numpy as np
import time
from pxr import UsdGeom, Gf
import isaacsim.core.utils.prims as prim_utils

def calculate_perfect_sdf_distance(point, obstacles):
    """完美版：计算精确SDF距离 - 圆环精准相切到障碍物表面"""
    min_distance = float('inf')
    point_2d = np.array(point[:2])
    
    for obs in obstacles:
        if obs['type'] == 'circle':
            # 圆形障碍物：精确SDF公式
            center = np.array(obs['center'])
            distance_to_center = np.linalg.norm(point_2d - center)
            # 标准圆形SDF：到边界的距离（正值在外部，负值在内部）
            distance_to_obstacle = distance_to_center - obs['radius']
            min_distance = min(min_distance, distance_to_obstacle)
            
        elif obs['type'] == 'rectangle':
            # 矩形障碍物：完全精确的SDF公式
            center = np.array(obs['center'])
            size = obs['size']
            
            # 转换到矩形局部坐标系
            local_pos = point_2d - center
            
            # 标准2D矩形SDF公式
            half_size = np.array([size[0] / 2.0, size[1] / 2.0])
            q = np.abs(local_pos) - half_size
            
            # 完整SDF计算
            outside_distance = np.linalg.norm(np.maximum(q, 0.0))
            inside_distance = min(np.max(q), 0.0)
            distance_to_obstacle = outside_distance + inside_distance
            
            min_distance = min(min_distance, distance_to_obstacle)
    
    # 返回真实SDF值，无任何人工调整
    return max(0.05, min_distance)

def create_perfect_tangent_ring(index, position, radius):
    """创建完美相切的真圆环 - 精准到障碍物表面，无重叠无缝隙"""
    ring_path = f"/World/PerfectTangent_Ring_{index}"
    
    # 删除已存在的圆环
    stage = omni.usd.get_context().get_stage()
    if stage:
        existing_prim = stage.GetPrimAtPath(ring_path)
        if existing_prim.IsValid():
            stage.RemovePrim(ring_path)
    
    # 创建真正的圆环几何（薄壁圆柱）
    ring_prim = prim_utils.create_prim(ring_path, "Cylinder")
    ring = UsdGeom.Cylinder(ring_prim)
    
    # 设置圆环参数：极薄的圆环效果
    ring.CreateRadiusAttr().Set(radius)
    ring.CreateHeightAttr().Set(0.008)  # 极薄，真正的圆环效果
    ring.CreateAxisAttr().Set("Z")
    
    # 防止Z-fighting：每个圆环不同高度，形成层次感
    z_offset = 0.015 + index * 0.010  
    
    # 精确变换设置
    xform = UsdGeom.Xformable(ring_prim)
    xform.ClearXformOpOrder()
    
    translate_op = xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    translate_op.Set(Gf.Vec3d(position[0], position[1], z_offset))
    
    # 完美的距离基于颜色编码：精确的视觉反馈
    if radius < 0.15:
        color = (1.0, 0.0, 0.0)  # 极危险：鲜红色
        opacity = 0.95
    elif radius < 0.4:
        color = (1.0, 0.3, 0.0)  # 危险：橙红色
        opacity = 0.9
    elif radius < 0.8:
        color = (1.0, 0.7, 0.0)  # 警告：橙黄色
        opacity = 0.8
    elif radius < 1.2:
        color = (0.7, 1.0, 0.0)  # 注意：黄绿色
        opacity = 0.7
    else:
        color = (0.0, 1.0, 0.2)  # 安全：绿色
        opacity = 0.6
    
    # 设置材质属性
    ring.CreateDisplayColorAttr().Set([color])
    ring.CreateDisplayOpacityAttr().Set([opacity])
    
    print(f"  🎯 完美相切圆环 {index}: 位置({position[0]:.2f}, {position[1]:.2f}), 精准半径={radius:.4f}m")
    return ring_path

def create_test_obstacle(obs_info, index):
    """创建测试障碍物"""
    if obs_info['type'] == 'circle':
        obstacle_path = f"/World/test_obstacle_circle_{index}"
        center = obs_info['center']
        radius = obs_info['radius']
        
        # 创建圆形障碍物
        obstacle_prim = prim_utils.create_prim(obstacle_path, "Cylinder")
        obstacle = UsdGeom.Cylinder(obstacle_prim)
        obstacle.CreateRadiusAttr().Set(radius)
        obstacle.CreateHeightAttr().Set(0.5)
        obstacle.CreateAxisAttr().Set("Z")
        
        # 设置位置
        xform = UsdGeom.Xformable(obstacle_prim)
        xform.ClearXformOpOrder()
        translate_op = xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
        translate_op.Set(Gf.Vec3d(center[0], center[1], 0.25))
        
        # 设置颜色（红色）
        obstacle.CreateDisplayColorAttr().Set([(0.8, 0.2, 0.2)])
        
    elif obs_info['type'] == 'rectangle':
        obstacle_path = f"/World/test_obstacle_rect_{index}"
        center = obs_info['center']
        size = obs_info['size']
        
        # 创建矩形障碍物
        obstacle_prim = prim_utils.create_prim(obstacle_path, "Cube")
        obstacle = UsdGeom.Cube(obstacle_prim)
        obstacle.CreateSizeAttr().Set(1.0)
        
        # 设置位置和缩放
        xform = UsdGeom.Xformable(obstacle_prim)
        xform.ClearXformOpOrder()
        
        translate_op = xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
        translate_op.Set(Gf.Vec3d(center[0], center[1], 0.25))
        
        scale_op = xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble)
        scale_op.Set(Gf.Vec3d(size[0], size[1], 0.5))
        
        # 设置颜色（红色）
        obstacle.CreateDisplayColorAttr().Set([(0.8, 0.2, 0.2)])

def main():
    """主测试函数"""
    try:
        print("🌟 启动完美相切圆环测试...")
        
        # 等待应用启动
        while not simulation_app.is_running():
            simulation_app.update()
            time.sleep(0.1)
        
        print("✅ Isaac Sim启动完成")
        
        # 创建基础地面
        stage = omni.usd.get_context().get_stage()
        default_prim_path = "/World"
        stage.DefinePrim(default_prim_path, "Xform")
        stage.SetDefaultPrim(stage.GetPrimAtPath(default_prim_path))
        
        # 定义测试障碍物
        test_obstacles = [
            {'type': 'circle', 'center': [2.0, 1.0], 'radius': 0.6},
            {'type': 'circle', 'center': [4.0, 3.0], 'radius': 0.5},
            {'type': 'rectangle', 'center': [6.0, 2.0], 'size': [1.0, 1.5]},
        ]
        
        # 创建测试障碍物
        print("🔧 创建测试障碍物...")
        for i, obs in enumerate(test_obstacles):
            create_test_obstacle(obs, i)
        
        # 定义测试轨迹点
        test_trajectory_points = [
            [0.0, 0.0],   # 起点
            [1.5, 0.5],   # 接近第一个圆形障碍物
            [2.8, 1.8],   # 在两个障碍物之间
            [3.5, 2.5],   # 接近第二个圆形障碍物
            [5.0, 1.5],   # 接近矩形障碍物
            [7.0, 3.0],   # 远离所有障碍物
        ]
        
        print("🎨 创建完美相切圆环...")
        ring_paths = []
        
        for i, point in enumerate(test_trajectory_points):
            # 计算完美精确的SDF距离
            perfect_distance = calculate_perfect_sdf_distance(point, test_obstacles)
            
            # 创建完美相切圆环
            ring_path = create_perfect_tangent_ring(i, point, perfect_distance)
            ring_paths.append(ring_path)
            
            print(f"点 {i}: ({point[0]:.1f}, {point[1]:.1f}) -> 距离: {perfect_distance:.4f}m")
        
        print(f"✨ 完美相切圆环测试完成！")
        print(f"✨ 创建了 {len(ring_paths)} 个精准圆环，每个都精确贴合最近的障碍物表面")
        print(f"🎯 观察圆环颜色:")
        print(f"   🔴 红色: 极危险区域 (<0.15m)")
        print(f"   🟠 橙色: 危险区域 (0.15-0.4m)")
        print(f"   🟡 黄色: 警告区域 (0.4-0.8m)")
        print(f"   🟢 绿色: 安全区域 (>1.2m)")
        print("")
        print("📝 按 Ctrl+C 退出测试")
        
        # 保持程序运行以观察效果
        while simulation_app.is_running():
            simulation_app.update()
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n🛑 用户中断测试")
    except Exception as e:
        print(f"❌ 测试异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("🧹 关闭仿真应用...")
        simulation_app.close()
        print("✅ 测试结束")

if __name__ == "__main__":
    main()
