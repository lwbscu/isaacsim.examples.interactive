#!/usr/bin/env python3
"""
Isaac Lab 虚光圈效果完整实现教程
演示如何创建各种发光圆环和扫积体积可视化效果
"""

import argparse
import torch
import numpy as np

from isaaclab.app import AppLauncher

# 创建参数解析器
parser = argparse.ArgumentParser(description="虚光圈效果演示")
parser.add_argument("--num_envs", type=int, default=1, help="环境数量")
parser.add_argument("--headless", action="store_true", help="无头模式运行")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 启动Omniverse应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""以下是主要实现代码"""

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import quat_from_angle_axis
import isaacsim.core.utils.prims as prim_utils


def create_glow_materials():
    """创建各种发光材质配置"""
    
    # 1. 基础青色虚光圈
    cyan_glow = sim_utils.PreviewSurfaceCfg(
        emissive_color=(0.0, 1.0, 1.0),    # 青色发光
        diffuse_color=(0.1, 0.1, 0.1),     # 深色基础
        opacity=0.7,                        # 半透明
        metallic=0.0,                       # 非金属
        roughness=0.1                       # 光滑表面
    )
    
    # 2. 橙色警告光圈
    orange_glow = sim_utils.PreviewSurfaceCfg(
        emissive_color=(1.0, 0.5, 0.0),    # 橙色发光
        diffuse_color=(0.1, 0.05, 0.0),    # 暖色基础
        opacity=0.8,
        metallic=0.1,
        roughness=0.2
    )
    
    # 3. 绿色安全光圈
    green_glow = sim_utils.PreviewSurfaceCfg(
        emissive_color=(0.0, 1.0, 0.2),    # 亮绿色发光
        diffuse_color=(0.0, 0.1, 0.02),    # 深绿基础
        opacity=0.6,
        metallic=0.0,
        roughness=0.15
    )
    
    # 4. 红色危险光圈
    red_glow = sim_utils.PreviewSurfaceCfg(
        emissive_color=(1.0, 0.1, 0.1),    # 红色发光
        diffuse_color=(0.1, 0.01, 0.01),   # 深红基础
        opacity=0.9,
        metallic=0.05,
        roughness=0.1
    )
    
    return {
        "cyan": cyan_glow,
        "orange": orange_glow, 
        "green": green_glow,
        "red": red_glow
    }


def create_swept_volume_visualization():
    """创建扫积体积可视化效果"""
    
    materials = create_glow_materials()
    
    # 1. 内层核心区域 - 红色危险区
    inner_core_cfg = sim_utils.CylinderCfg(
        radius=1.0,                         # 1米半径
        height=0.05,                        # 很薄的圆盘
        axis="Z",                           # Z轴向上
        visual_material=materials["red"],
        physics_material=None               # 不需要物理属性
    )
    
    # 2. 中层警告区域 - 橙色警告区  
    middle_warning_cfg = sim_utils.CylinderCfg(
        radius=2.0,                         # 2米半径
        height=0.08,
        axis="Z",
        visual_material=materials["orange"],
        physics_material=None
    )
    
    # 3. 外层安全区域 - 绿色安全区
    outer_safe_cfg = sim_utils.CylinderCfg(
        radius=3.0,                         # 3米半径  
        height=0.12,
        axis="Z",
        visual_material=materials["green"],
        physics_material=None
    )
    
    # 4. 最外层扫积边界 - 青色边界
    boundary_cfg = sim_utils.CylinderCfg(
        radius=4.0,                         # 4米半径
        height=0.15,
        axis="Z", 
        visual_material=materials["cyan"],
        physics_material=None
    )
    
    return {
        "inner": inner_core_cfg,
        "middle": middle_warning_cfg,
        "outer": outer_safe_cfg,
        "boundary": boundary_cfg
    }


def setup_scene():
    """设置完整的演示场景"""
    
    # 1. 创建地面
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/GroundPlane", ground_cfg)
    
    # 2. 创建光源
    light_cfg = sim_utils.DomeLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75)
    )
    light_cfg.func("/World/Light", light_cfg)
    
    # 3. 创建Transform容器
    prim_utils.create_prim("/World/SweptVolume", "Xform")
    
    # 4. 生成扫积体积可视化
    swept_configs = create_swept_volume_visualization()
    
    # 在不同高度生成多层光圈
    heights = [0.1, 0.2, 0.3, 0.4]  # 不同Z高度
    
    for i, (name, config) in enumerate(swept_configs.items()):
        config.func(
            f"/World/SweptVolume/{name}_ring",
            config,
            translation=(0.0, 0.0, heights[i])  # 设置不同高度
        )
    
    # 5. 创建示例车辆（简单立方体）
    vehicle_cfg = sim_utils.CuboidCfg(
        size=(4.0, 2.0, 1.5),  # 长4米，宽2米，高1.5米
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.2, 0.2, 0.8),    # 蓝色车身
            metallic=0.7,                      # 金属质感
            roughness=0.3
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=1500.0)
    )
    
    vehicle_cfg.func(
        "/World/Vehicle",
        vehicle_cfg,
        translation=(0.0, 0.0, 1.0)  # 车辆位置
    )


def create_dynamic_markers():
    """创建动态可视化标记器"""
    
    materials = create_glow_materials()
    
    # 配置可视化标记器
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/DynamicMarkers",
        markers={
            # 轨迹点标记
            "trajectory_point": sim_utils.SphereCfg(
                radius=0.1,
                visual_material=materials["cyan"]
            ),
            
            # 方向箭头
            "direction_arrow": sim_utils.ConeCfg(
                radius=0.2,
                height=0.6,
                visual_material=materials["orange"]
            ),
            
            # 扫积边界圆环
            "swept_boundary": sim_utils.CylinderCfg(
                radius=2.5,
                height=0.1,
                visual_material=materials["green"]
            )
        }
    )
    
    return VisualizationMarkers(marker_cfg)


def run_simulation():
    """运行仿真循环"""
    
    # 初始化仿真上下文
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, render_interval=4)
    sim = SimulationContext(sim_cfg)
    
    # 设置场景
    setup_scene()
    
    # 创建动态标记器
    markers = create_dynamic_markers()
    
    # 设置相机视角
    sim.set_camera_view([8.0, 8.0, 5.0], [0.0, 0.0, 0.0])
    
    # 仿真参数
    sim_time = 0.0
    count = 0
    
    print("🚀 虚光圈效果演示启动!")
    print("💡 观察多层发光圆环的扫积体积可视化效果")
    
    # 主仿真循环
    while simulation_app.is_running():
        
        # 动态更新标记器位置 (圆形轨迹)
        num_markers = 20
        angles = torch.linspace(0, 2*np.pi, num_markers)
        
        # 轨迹点位置
        trajectory_positions = torch.zeros(num_markers, 3)
        trajectory_positions[:, 0] = 3.0 * torch.cos(angles + sim_time)  # X坐标
        trajectory_positions[:, 1] = 3.0 * torch.sin(angles + sim_time)  # Y坐标  
        trajectory_positions[:, 2] = 0.5                                  # Z坐标
        
        # 方向箭头
        arrow_orientations = quat_from_angle_axis(
            angles + sim_time, 
            torch.tensor([0.0, 0.0, 1.0])
        )
        
        # 扫积边界位置 (随时间变化半径)
        boundary_positions = torch.zeros(8, 3)
        boundary_angles = torch.linspace(0, 2*np.pi, 8)
        radius_scale = 1.0 + 0.3 * torch.sin(sim_time * 2)  # 动态半径
        
        boundary_positions[:, 0] = radius_scale * torch.cos(boundary_angles)
        boundary_positions[:, 1] = radius_scale * torch.sin(boundary_angles)
        boundary_positions[:, 2] = 0.2
        
        # 更新标记器显示
        marker_indices = torch.zeros(num_markers + 8, dtype=torch.int32)
        marker_indices[:num_markers] = 0  # 轨迹点
        marker_indices[num_markers:] = 2  # 边界圆环
        
        all_positions = torch.cat([trajectory_positions, boundary_positions])
        all_orientations = torch.cat([
            torch.zeros(num_markers, 4),
            torch.zeros(8, 4)
        ])
        all_orientations[:, 0] = 1.0  # w=1 (单位四元数)
        
        # 可视化标记器
        markers.visualize(all_positions, all_orientations, marker_indices)
        
        # 执行仿真步骤
        sim.step()
        
        # 更新计时器
        sim_time += sim.get_physics_dt()
        count += 1
        
        # 定期输出状态信息
        if count % 100 == 0:
            print(f"⏱️  仿真时间: {sim_time:.2f}s | 帧数: {count}")


if __name__ == "__main__":
    try:
        run_simulation()
    except KeyboardInterrupt:
        print("\n🛑 用户中断仿真")
    finally:
        # 清理资源
        print("🧹 清理仿真资源...")
        simulation_app.close()