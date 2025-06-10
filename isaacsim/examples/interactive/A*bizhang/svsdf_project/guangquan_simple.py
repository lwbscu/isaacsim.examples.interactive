#!/usr/bin/env python3
"""
Isaac Sim 虚光圈效果演示
创建发光圆环和扫积体积可视化效果
"""

import sys
import os

# 设置Isaac Sim路径
isaac_sim_path = "/home/lwb/isaacsim"
if isaac_sim_path not in sys.path:
    sys.path.append(isaac_sim_path)

# Isaac Sim仿真应用
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import numpy as np
import time
import carb
import omni
import omni.usd
from pxr import UsdGeom, Gf, Usd, UsdLux, UsdShade, Sdf
from isaacsim.core.api import World
import isaacsim.core.utils.prims as prim_utils


def create_simple_cylinder(prim_path: str, radius: float, height: float, 
                          color: tuple, position: tuple = (0, 0, 0)):
    """创建简单的彩色圆柱体"""
    # 创建圆柱体几何体
    cylinder_prim = prim_utils.create_prim(
        prim_path=prim_path,
        prim_type="Cylinder"
    )
    
    # 设置圆柱体属性
    cylinder = UsdGeom.Cylinder(cylinder_prim)
    cylinder.CreateRadiusAttr().Set(radius)
    cylinder.CreateHeightAttr().Set(height)
    cylinder.CreateAxisAttr().Set("Z")
    
    # 设置位置 - 使用正确的精度类型
    xform = UsdGeom.Xformable(cylinder_prim)
    xform.ClearXformOpOrder()
    
    translate_op = xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    translate_op.Set(Gf.Vec3d(position[0], position[1], position[2]))
    
    # 简单的颜色设置（通过displayColor）
    cylinder.CreateDisplayColorAttr().Set([color])
    
    return cylinder_prim


def create_swept_volume_rings():
    """创建扫积体积可视化圆环"""
    
    # 不同层级的圆环配置
    ring_configs = [
        {"name": "inner_core", "radius": 1.0, "height": 0.05, "color": (1.0, 0.1, 0.1), "z": 0.1},
        {"name": "warning_zone", "radius": 2.0, "height": 0.08, "color": (1.0, 0.5, 0.0), "z": 0.2},
        {"name": "safe_zone", "radius": 3.0, "height": 0.12, "color": (0.0, 1.0, 0.2), "z": 0.3},
        {"name": "boundary", "radius": 4.0, "height": 0.15, "color": (0.0, 1.0, 1.0), "z": 0.4},
    ]
    
    # 创建根容器
    prim_utils.create_prim("/World/SweptVolume", "Xform")
    
    rings = []
    for config in ring_configs:
        ring_path = f"/World/SweptVolume/{config['name']}"
        
        # 创建简单的彩色圆环
        ring_prim = create_simple_cylinder(
            ring_path,
            config["radius"],
            config["height"],
            config["color"],
            position=(0.0, 0.0, config["z"])
        )
        
        rings.append(ring_prim)
        print(f"✅ 创建发光圆环: {config['name']} (半径: {config['radius']}m)")
    
    return rings


def create_demo_scene():
    """创建演示场景"""
    
    # 创建地面
    ground_prim = prim_utils.create_prim("/World/GroundPlane", "Cube")
    ground = UsdGeom.Cube(ground_prim)
    ground.CreateSizeAttr().Set(1.0)
    ground.CreateDisplayColorAttr().Set([(0.3, 0.3, 0.3)])
    
    # 设置地面变换 - 使用一致的精度类型
    ground_xform = UsdGeom.Xformable(ground_prim)
    ground_xform.ClearXformOpOrder()
    
    # 使用特定精度的操作避免类型冲突
    translate_op = ground_xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    translate_op.Set(Gf.Vec3d(0.0, 0.0, -0.5))
    
    scale_op = ground_xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble)
    scale_op.Set(Gf.Vec3d(20.0, 20.0, 1.0))
    
    # 创建示例车辆
    vehicle_prim = prim_utils.create_prim("/World/Vehicle", "Cube")
    vehicle = UsdGeom.Cube(vehicle_prim)
    vehicle.CreateSizeAttr().Set(1.0)
    vehicle.CreateDisplayColorAttr().Set([(0.2, 0.2, 0.8)])
    
    # 设置车辆变换 - 使用一致的精度类型
    vehicle_xform = UsdGeom.Xformable(vehicle_prim)
    vehicle_xform.ClearXformOpOrder()
    
    vehicle_translate_op = vehicle_xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    vehicle_translate_op.Set(Gf.Vec3d(0.0, 0.0, 1.0))
    
    vehicle_scale_op = vehicle_xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble)
    vehicle_scale_op.Set(Gf.Vec3d(2.0, 1.0, 0.8))
    
    print("✅ 场景创建完成")


def animate_rings(rings, time_step: float):
    """动画化圆环效果"""
    
    for i, ring_prim in enumerate(rings):
        if ring_prim:
            try:
                xform = UsdGeom.Xformable(ring_prim)
                
                # 创建旋转动画
                rotation_speed = 10.0 + i * 5.0  # 不同圆环不同转速
                rotation_angle = time_step * rotation_speed
                
                # 重置变换操作并设置新的变换
                xform.ClearXformOpOrder()
                
                # 设置位置（从配置中获取）- 使用一致的精度
                z_pos = 0.1 + i * 0.1
                translate_op = xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
                translate_op.Set(Gf.Vec3d(0.0, 0.0, z_pos))
                
                # Z轴旋转 - 使用一致的精度
                rotate_op = xform.AddRotateZOp(UsdGeom.XformOp.PrecisionDouble)
                rotate_op.Set(rotation_angle)
                
                # 脉冲缩放效果 - 使用一致的精度
                pulse_scale = 1.0 + 0.1 * np.sin(time_step * 3.0 + i)
                scale_op = xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble)
                scale_op.Set(Gf.Vec3d(pulse_scale, pulse_scale, 1.0))
                
            except Exception as e:
                print(f"⚠️ 动画错误: {e}")
                continue


def run_simulation():
    """运行仿真演示"""
    
    print("🚀 启动虚光圈效果演示")
    
    # 等待仿真应用完全启动
    if hasattr(simulation_app, 'update'):
        for _ in range(10):  # 等待几帧让应用完全初始化
            simulation_app.update()
    
    try:
        # 创建世界
        world = World(stage_units_in_meters=1.0)
        
        print("🌍 创建仿真世界...")
        
        # 创建场景
        create_demo_scene()
        print("🎬 创建演示场景...")
        
        # 创建发光圆环
        rings = create_swept_volume_rings()
        print("✨ 创建虚光圈...")
        
        # 重置世界
        world.reset()
        print("🔄 重置仿真世界...")
        
        # 开始仿真
        world.play()  # 确保仿真开始运行
        print("▶️  开始仿真...")
        
        print("💡 虚光圈效果已启动，观察多层发光圆环效果")
        print("⏸️  按 Ctrl+C 退出演示")
        
        # 仿真循环
        simulation_time = 0.0
        frame_count = 0
        dt = 1.0/60.0  # 60 FPS
        
        while simulation_app.is_running():
            # 更新应用状态
            simulation_app.update()
            
            # 动画化圆环
            animate_rings(rings, simulation_time)
            
            # 仿真步进
            world.step(render=True)
            
            # 更新时间
            simulation_time += dt
            frame_count += 1
            
            # 定期输出状态
            if frame_count % 300 == 0:  # 每5秒
                print(f"⏱️  仿真时间: {simulation_time:.1f}s | 帧数: {frame_count}")
            
            # 简单延时控制帧率
            time.sleep(dt)
            
    except KeyboardInterrupt:
        print("\n🛑 用户中断仿真")
    
    except Exception as e:
        print(f"❌ 仿真错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if 'world' in locals():
            print("🧹 清理仿真资源...")
            world.clear()


if __name__ == "__main__":
    try:
        print("🌟 正在启动Isaac Sim虚光圈演示...")
        
        # 等待 simulation_app 完全初始化
        while not simulation_app.is_running():
            simulation_app.update()
            time.sleep(0.1)
        
        print("✅ Isaac Sim启动完成")
        
        run_simulation()
        
    except KeyboardInterrupt:
        print("\n🛑 用户中断演示")
    except Exception as e:
        print(f"❌ 演示异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("🧹 关闭仿真应用...")
        simulation_app.close()
        print("✅ 演示结束")
