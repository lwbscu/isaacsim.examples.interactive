# isaac_sim_visual_effects.py
"""
Isaac Sim高级视觉效果库
专门针对Isaac Sim的高级可视化技术
"""
import numpy as np
import asyncio
from typing import List, Dict, Optional, Tuple
import omni
from omni.isaac.core.materials import PreviewSurface, OmniPBR, ParticleMaterial
from omni.isaac.core.objects import VisualCuboid, VisualSphere, VisualCylinder
from omni.isaac.core.utils.prims import create_prim, get_prim_at_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from pxr import Usd, UsdGeom, Gf, UsdShade, Sdf, UsdLux
import carb

class IsaacSimVisualEffects:
    """
    Isaac Sim专业视觉效果库
    提供丰富的可视化效果，替代传统的"小方块"
    """
    
    def __init__(self, stage):
        self.stage = stage
        self.effects_root = "/World/VisualEffects"
        self.materials_cache = {}
        self.active_effects = {}
        
        # 效果配置
        self.config = {
            'enable_lighting_effects': True,
            'enable_particle_systems': True,
            'enable_material_animations': True,
            'max_particles': 1000,
            'animation_quality': 'high'  # 'low', 'medium', 'high'
        }
        
        self._initialize_effects_system()
    
    def _initialize_effects_system(self):
        """初始化效果系统"""
        create_prim(self.effects_root, "Xform")
        
        # 创建效果子目录
        subdirs = ["Lighting", "Particles", "Materials", "Geometry", "Animations"]
        for subdir in subdirs:
            create_prim(f"{self.effects_root}/{subdir}", "Xform")
        
        # 初始化高级材质库
        self._create_professional_materials()
        
        # 设置动态光照
        self._setup_dynamic_lighting()
        
        print("Isaac Sim视觉效果系统已初始化")

    # ==================== 高级材质系统 ====================
    
    def _create_professional_materials(self):
        """创建专业级材质库"""
        materials_path = f"{self.effects_root}/Materials"
        
        # 1. 全息材质（用于轨迹）
        hologram_material = OmniPBR(
            prim_path=f"{materials_path}/hologram_material",
            name="hologram_material",
            color=np.array([0.0, 0.8, 1.0]),
            roughness=0.0,
            metallic=0.9,
            opacity=0.7
        )
        # 添加自发光
        hologram_material.set_emissive_color(np.array([0.0, 0.5, 1.0]))
        hologram_material.set_emissive_intensity(3.0)
        self.materials_cache['hologram'] = hologram_material
        
        # 2. 能量场材质
        energy_field_material = OmniPBR(
            prim_path=f"{materials_path}/energy_field_material",
            name="energy_field_material",
            color=np.array([1.0, 0.3, 0.8]),
            roughness=0.1,
            metallic=0.0,
            opacity=0.4
        )
        energy_field_material.set_emissive_color(np.array([1.0, 0.0, 0.8]))
        energy_field_material.set_emissive_intensity(5.0)
        self.materials_cache['energy_field'] = energy_field_material
        
        # 3. 光迹材质（用于路径可视化）
        light_trail_material = OmniPBR(
            prim_path=f"{materials_path}/light_trail_material",
            name="light_trail_material",
            color=np.array([0.0, 1.0, 0.3]),
            roughness=0.0,
            metallic=1.0,
            opacity=0.8
        )
        light_trail_material.set_emissive_color(np.array([0.0, 1.0, 0.0]))
        light_trail_material.set_emissive_intensity(4.0)
        self.materials_cache['light_trail'] = light_trail_material
        
        # 4. 热力图材质系列
        for i in range(10):
            heat_intensity = i / 9.0
            # 从蓝色到红色的热力图
            if heat_intensity < 0.5:
                color = np.array([0.0, heat_intensity * 2, 1.0 - heat_intensity * 2])
            else:
                color = np.array([(heat_intensity - 0.5) * 2, 1.0 - (heat_intensity - 0.5) * 2, 0.0])
            
            heat_material = OmniPBR(
                prim_path=f"{materials_path}/heat_material_{i}",
                name=f"heat_material_{i}",
                color=color,
                roughness=0.3,
                metallic=0.1,
                opacity=0.6 + 0.4 * heat_intensity
            )
            heat_material.set_emissive_color(color * 0.5)
            heat_material.set_emissive_intensity(2.0 * heat_intensity)
            self.materials_cache[f'heat_{i}'] = heat_material
    
    def _setup_dynamic_lighting(self):
        """设置动态光照系统"""
        if not self.config['enable_lighting_effects']:
            return
        
        lighting_path = f"{self.effects_root}/Lighting"
        
        # 1. 彩色聚光灯（用于强调）
        spot_light = create_prim(f"{lighting_path}/spot_light", "SphereLight")
        if spot_light:
            light = UsdLux.SphereLight(spot_light)
            light.CreateRadiusAttr(0.1)
            light.CreateIntensityAttr(1000)
            light.CreateColorAttr(Gf.Vec3f(0.0, 0.8, 1.0))
            
            # 设置位置
            xform = UsdGeom.Xformable(spot_light)
            xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, 5))
        
        # 2. 环境光增强
        env_light = create_prim(f"{lighting_path}/env_light", "DomeLight")
        if env_light:
            light = UsdLux.DomeLight(env_light)
            light.CreateIntensityAttr(300)
            light.CreateColorAttr(Gf.Vec3f(0.9, 0.95, 1.0))

    # ==================== 轨迹可视化效果 ====================
    
    async def create_holographic_trajectory(self, trajectory_points: List[np.ndarray], 
                                          name: str = "holo_trajectory"):
        """创建全息轨迹效果"""
        traj_path = f"{self.effects_root}/Geometry/{name}"
        create_prim(traj_path, "Xform")
        
        # 1. 主轨迹线
        await self._create_flowing_trajectory_line(trajectory_points, f"{traj_path}/main_line")
        
        # 2. 轨迹点光球
        await self._create_trajectory_light_orbs(trajectory_points, f"{traj_path}/light_orbs")
        
        # 3. 能量流效果
        await self._create_energy_flow_effect(trajectory_points, f"{traj_path}/energy_flow")
        
        # 4. 数据标签
        await self._create_trajectory_data_labels(trajectory_points, f"{traj_path}/data_labels")
        
        return traj_path
    
    async def _create_flowing_trajectory_line(self, points: List[np.ndarray], path: str):
        """创建流动的轨迹线"""
        create_prim(path, "Xform")
        
        # 创建多层线条以模拟发光效果
        line_widths = [0.05, 0.03, 0.01]
        opacities = [0.3, 0.6, 1.0]
        
        for layer_idx, (width, opacity) in enumerate(zip(line_widths, opacities)):
            layer_path = f"{path}/layer_{layer_idx}"
            create_prim(layer_path, "Xform")
            
            for i in range(len(points) - 1):
                segment_path = f"{layer_path}/segment_{i}"
                
                start_pos = points[i][:3]
                end_pos = points[i+1][:3]
                
                # 计算线段参数
                center = (start_pos + end_pos) / 2.0
                direction = end_pos - start_pos
                length = np.linalg.norm(direction)
                
                if length > 1e-6:
                    # 创建发光圆柱体
                    cylinder = VisualCylinder(
                        prim_path=segment_path,
                        position=center,
                        radius=width,
                        height=length,
                        color=np.array([0.0, 0.8, 1.0])
                    )
                    
                    # 应用全息材质
                    cylinder.apply_visual_material(self.materials_cache['hologram'])
                    
                    # 对齐方向
                    self._align_to_direction(cylinder, direction)
            
            # 启动流动动画
            asyncio.create_task(self._animate_flowing_effect(layer_path, len(points)))
    
    async def _create_trajectory_light_orbs(self, points: List[np.ndarray], path: str):
        """创建轨迹光球"""
        create_prim(path, "Xform")
        
        for i, point in enumerate(points[::2]):  # 每隔一个点创建光球
            orb_path = f"{path}/orb_{i}"
            
            # 创建多层光球（内核+光环）
            # 内核
            core = VisualSphere(
                prim_path=f"{orb_path}/core",
                position=point[:3] + np.array([0, 0, 0.05]),
                radius=0.02,
                color=np.array([1.0, 1.0, 1.0])
            )
            core.apply_visual_material(self.materials_cache['hologram'])
            
            # 光环
            for ring_idx in range(3):
                ring_radius = 0.04 + ring_idx * 0.02
                ring_path = f"{orb_path}/ring_{ring_idx}"
                
                # 创建环形（用薄圆柱体）
                ring = VisualCylinder(
                    prim_path=ring_path,
                    position=point[:3] + np.array([0, 0, 0.05]),
                    radius=ring_radius,
                    height=0.002,
                    color=np.array([0.0, 0.8, 1.0])
                )
                ring.apply_visual_material(self.materials_cache['energy_field'])
            
            # 启动脉冲动画
            asyncio.create_task(self._animate_orb_pulse(orb_path, delay=i * 0.1))
    
    async def _create_energy_flow_effect(self, points: List[np.ndarray], path: str):
        """创建能量流效果"""
        if not self.config['enable_particle_systems']:
            return
        
        create_prim(path, "Xform")
        
        # 沿轨迹创建能量粒子
        particle_count = min(self.config['max_particles'], len(points) * 5)
        
        for i in range(particle_count):
            # 随机选择轨迹点
            point_idx = np.random.randint(0, len(points)-1)
            t = np.random.rand()
            
            # 在轨迹段上插值
            if point_idx < len(points) - 1:
                pos = points[point_idx][:3] * (1-t) + points[point_idx+1][:3] * t
                # 添加随机扰动
                pos += np.random.normal(0, 0.02, 3)
                pos[2] = max(0.01, pos[2])  # 确保在地面以上
                
                particle_path = f"{path}/particle_{i}"
                
                # 创建小能量球
                particle = VisualSphere(
                    prim_path=particle_path,
                    position=pos,
                    radius=0.005,
                    color=np.array([0.0, 1.0, 0.8])
                )
                particle.apply_visual_material(self.materials_cache['energy_field'])
                
                # 启动粒子动画
                asyncio.create_task(self._animate_particle_flow(
                    particle_path, points, point_idx, delay=i * 0.02))

    # ==================== 扫掠体积可视化 ====================
    
    async def create_advanced_swept_volume(self, boundary_points: List[np.ndarray],
                                         density_data: Optional[np.ndarray] = None,
                                         name: str = "swept_volume"):
        """创建高级扫掠体积可视化"""
        volume_path = f"{self.effects_root}/Geometry/{name}"
        create_prim(volume_path, "Xform")
        
        # 1. 边界力场效果
        await self._create_boundary_force_field(boundary_points, f"{volume_path}/boundary")
        
        # 2. 3D热力图
        if density_data is not None:
            await self._create_3d_heatmap(density_data, f"{volume_path}/heatmap")
        
        # 3. 体积指示器
        await self._create_volume_indicators(boundary_points, f"{volume_path}/indicators")
        
        # 4. 扫掠动画
        await self._create_sweeping_animation(boundary_points, f"{volume_path}/sweep_anim")
        
        return volume_path
    
    async def _create_boundary_force_field(self, boundary_points: List[np.ndarray], path: str):
        """创建边界力场效果"""
        create_prim(path, "Xform")
        
        # 创建边界线的发光效果
        for i in range(len(boundary_points)):
            next_i = (i + 1) % len(boundary_points)
            
            start_pos = np.array([boundary_points[i][0], boundary_points[i][1], 0])
            end_pos = np.array([boundary_points[next_i][0], boundary_points[next_i][1], 0])
            
            # 主边界线
            edge_path = f"{path}/edge_{i}"
            await self._create_energy_beam(start_pos, end_pos, edge_path, intensity=1.0)
            
            # 向上的能量柱
            pillar_path = f"{path}/pillar_{i}"
            pillar_top = start_pos + np.array([0, 0, 0.5])
            await self._create_energy_beam(start_pos, pillar_top, pillar_path, intensity=0.6)
    
    async def _create_3d_heatmap(self, density_data: np.ndarray, path: str):
        """创建3D热力图"""
        create_prim(path, "Xform")
        
        height, width = density_data.shape
        max_density = np.max(density_data)
        
        if max_density <= 0:
            return
        
        for i in range(height):
            for j in range(width):
                density = density_data[i, j]
                if density > 0.01:  # 只显示有意义的密度
                    
                    # 计算位置
                    x = j - width / 2
                    y = i - height / 2
                    z = density / max_density * 0.3  # 高度基于密度
                    
                    # 选择热力图材质
                    material_idx = int((density / max_density) * 9)
                    material_key = f'heat_{material_idx}'
                    
                    # 创建密度立方体
                    cube_path = f"{path}/density_{i}_{j}"
                    
                    cube = VisualCuboid(
                        prim_path=cube_path,
                        position=np.array([x * 0.1, y * 0.1, z]),
                        size=np.array([0.08, 0.08, z * 2]),
                        color=np.array([density/max_density, 0.0, 1.0-density/max_density])
                    )
                    
                    if material_key in self.materials_cache:
                        cube.apply_visual_material(self.materials_cache[material_key])
                    
                    # 添加脉冲效果
                    asyncio.create_task(self._animate_density_pulse(
                        cube_path, density/max_density, delay=np.random.rand()))

    # ==================== MPC预测可视化 ====================
    
    async def create_mpc_prediction_field(self, predicted_states: List[np.ndarray],
                                        reference_trajectory: List[np.ndarray],
                                        name: str = "mpc_field"):
        """创建MPC预测场可视化"""
        field_path = f"{self.effects_root}/Geometry/{name}"
        create_prim(field_path, "Xform")
        
        # 1. 预测轨迹光束
        await self._create_prediction_beam(predicted_states, f"{field_path}/prediction")
        
        # 2. 参考轨迹
        await self._create_reference_trail(reference_trajectory, f"{field_path}/reference")
        
        # 3. 误差可视化
        await self._create_error_visualization(
            predicted_states, reference_trajectory, f"{field_path}/errors")
        
        # 4. 控制信心区域
        await self._create_confidence_region(predicted_states, f"{field_path}/confidence")
        
        return field_path
    
    async def _create_prediction_beam(self, predicted_states: List[np.ndarray], path: str):
        """创建预测轨迹光束"""
        create_prim(path, "Xform")
        
        for i in range(len(predicted_states) - 1):
            beam_path = f"{path}/beam_{i}"
            
            start_pos = predicted_states[i][:3]
            end_pos = predicted_states[i+1][:3]
            
            # 透明度递减
            alpha = 1.0 - (i / len(predicted_states)) * 0.8
            
            await self._create_energy_beam(start_pos, end_pos, beam_path, 
                                         intensity=alpha, color=np.array([1.0, 0.6, 0.0]))
    
    async def _create_error_visualization(self, predicted: List[np.ndarray], 
                                        reference: List[np.ndarray], path: str):
        """创建误差可视化"""
        create_prim(path, "Xform")
        
        min_len = min(len(predicted), len(reference))
        
        for i in range(min_len):
            pred_pos = predicted[i][:3]
            ref_pos = reference[i][:3]
            
            error = np.linalg.norm(pred_pos - ref_pos)
            
            if error > 0.01:  # 只显示明显误差
                error_path = f"{path}/error_{i}"
                
                # 创建误差指示器
                await self._create_error_indicator(pred_pos, ref_pos, error_path, error)

    # ==================== 辅助方法 ====================
    
    async def _create_energy_beam(self, start_pos: np.ndarray, end_pos: np.ndarray,
                                path: str, intensity: float = 1.0, 
                                color: np.ndarray = np.array([0.0, 0.8, 1.0])):
        """创建能量光束"""
        direction = end_pos - start_pos
        length = np.linalg.norm(direction)
        center = (start_pos + end_pos) / 2.0
        
        if length > 1e-6:
            # 主光束
            main_beam = VisualCylinder(
                prim_path=f"{path}/main",
                position=center,
                radius=0.01 * intensity,
                height=length,
                color=color * intensity
            )
            main_beam.apply_visual_material(self.materials_cache['light_trail'])
            self._align_to_direction(main_beam, direction)
            
            # 外光环
            outer_glow = VisualCylinder(
                prim_path=f"{path}/glow",
                position=center,
                radius=0.03 * intensity,
                height=length,
                color=color * intensity * 0.3
            )
            outer_glow.apply_visual_material(self.materials_cache['energy_field'])
            self._align_to_direction(outer_glow, direction)
            
            # 启动能量流动动画
            asyncio.create_task(self._animate_energy_flow(path, intensity))
    
    def _align_to_direction(self, object_prim, direction: np.ndarray):
        """将对象对齐到指定方向"""
        try:
            prim = get_prim_at_path(object_prim.prim_path)
            if prim and prim.IsValid():
                direction_normalized = direction / np.linalg.norm(direction)
                
                # 计算旋转
                default_dir = np.array([0, 0, 1])
                
                if not np.allclose(direction_normalized, default_dir):
                    rotation_axis = np.cross(default_dir, direction_normalized)
                    rotation_angle = np.arccos(np.clip(np.dot(default_dir, direction_normalized), -1, 1))
                    
                    if np.linalg.norm(rotation_axis) > 1e-6:
                        rotation_axis_normalized = rotation_axis / np.linalg.norm(rotation_axis)
                        
                        # 转换为欧拉角
                        from scipy.spatial.transform import Rotation
                        r = Rotation.from_rotvec(rotation_angle * rotation_axis_normalized)
                        euler_angles = r.as_euler('xyz', degrees=True)
                        
                        xform = UsdGeom.Xformable(prim)
                        xform.AddRotateXYZOp().Set(Gf.Vec3f(euler_angles[0], euler_angles[1], euler_angles[2]))
        except Exception as e:
            print(f"对齐方向失败: {e}")
    
    async def _animate_flowing_effect(self, path: str, num_segments: int):
        """动画流动效果"""
        try:
            while path in self.active_effects:
                for i in range(num_segments):
                    segment_path = f"{path}/segment_{i}"
                    
                    # 创建临时发光效果
                    self._enhance_segment_glow(segment_path)
                    await asyncio.sleep(0.05)
                    self._reduce_segment_glow(segment_path)
                
                await asyncio.sleep(0.5)  # 重复周期
        except Exception as e:
            print(f"流动动画失败: {e}")
    
    async def _animate_orb_pulse(self, orb_path: str, delay: float = 0.0):
        """光球脉冲动画"""
        await asyncio.sleep(delay)
        
        try:
            while orb_path in self.active_effects:
                # 放大
                for scale in np.linspace(1.0, 1.3, 10):
                    self._set_scale(orb_path, scale)
                    await asyncio.sleep(0.05)
                
                # 缩小
                for scale in np.linspace(1.3, 1.0, 10):
                    self._set_scale(orb_path, scale)
                    await asyncio.sleep(0.05)
                
                await asyncio.sleep(1.0)  # 脉冲间隔
        except Exception as e:
            print(f"脉冲动画失败: {e}")
    
    def _set_scale(self, prim_path: str, scale: float):
        """设置对象缩放"""
        try:
            prim = get_prim_at_path(prim_path)
            if prim and prim.IsValid():
                xform = UsdGeom.Xformable(prim)
                scale_op = xform.AddScaleOp()
                scale_op.Set(Gf.Vec3f(scale, scale, scale))
        except Exception as e:
            print(f"设置缩放失败: {e}")
    
    def start_effect(self, effect_name: str):
        """启动效果"""
        self.active_effects[effect_name] = True
    
    def stop_effect(self, effect_name: str):
        """停止效果"""
        if effect_name in self.active_effects:
            del self.active_effects[effect_name]
    
    def cleanup(self):
        """清理所有效果"""
        # 停止所有效果
        self.active_effects.clear()
        
        # 删除效果节点
        try:
            prim = get_prim_at_path(self.effects_root)
            if prim and prim.IsValid():
                self.stage.RemovePrim(prim.GetPath())
        except Exception as e:
            print(f"清理效果失败: {e}")
        
        # 清理材质缓存
        self.materials_cache.clear()
        
        print("视觉效果系统已清理")

# ==================== Isaac Sim UI增强 ====================

class IsaacSimUIEnhancer:
    """
    Isaac Sim UI增强器
    添加实时数据显示和交互控件
    """
    
    def __init__(self):
        self.ui_elements = {}
        self.data_displays = {}
        
    def create_performance_hud(self, position: Tuple[int, int] = (10, 10)):
        """创建性能抬头显示"""
        try:
            import omni.ui as ui
            
            # 创建性能显示窗口
            self.performance_window = ui.Window(
                "轨迹优化性能监控", 
                width=300, 
                height=200,
                flags=ui.WINDOW_FLAGS_NO_RESIZE
            )
            
            with self.performance_window.frame:
                with ui.VStack():
                    ui.Label("实时性能监控", style={"font_size": 16, "color": 0xFF00FF00})
                    ui.Separator()
                    
                    # 规划时间显示
                    with ui.HStack():
                        ui.Label("规划时间:", width=80)
                        self.planning_time_label = ui.Label("0.000s", style={"color": 0xFF00CCFF})
                    
                    # MPC频率显示
                    with ui.HStack():
                        ui.Label("MPC频率:", width=80)
                        self.mpc_freq_label = ui.Label("0.0Hz", style={"color": 0xFF00CCFF})
                    
                    # 扫掠体积显示
                    with ui.HStack():
                        ui.Label("扫掠体积:", width=80)
                        self.swept_vol_label = ui.Label("0.000m²", style={"color": 0xFF00CCFF})
                    
                    # 跟踪误差显示
                    with ui.HStack():
                        ui.Label("跟踪误差:", width=80)
                        self.tracking_error_label = ui.Label("0.000m", style={"color": 0xFF00CCFF})
                    
                    ui.Separator()
                    
                    # 控制按钮
                    with ui.HStack():
                        self.start_button = ui.Button("开始", width=60)
                        self.pause_button = ui.Button("暂停", width=60)
                        self.stop_button = ui.Button("停止", width=60)
            
        except ImportError:
            print("omni.ui不可用，跳过UI创建")
    
    def update_performance_data(self, data: Dict):
        """更新性能数据显示"""
        try:
            if hasattr(self, 'planning_time_label'):
                self.planning_time_label.text = f"{data.get('planning_time', 0):.3f}s"
            
            if hasattr(self, 'mpc_freq_label'):
                self.mpc_freq_label.text = f"{data.get('mpc_frequency', 0):.1f}Hz"
            
            if hasattr(self, 'swept_vol_label'):
                self.swept_vol_label.text = f"{data.get('swept_volume', 0):.3f}m²"
            
            if hasattr(self, 'tracking_error_label'):
                error = data.get('tracking_error', 0)
                color = 0xFF00FF00 if error < 0.1 else 0xFFFFFF00 if error < 0.3 else 0xFFFF0000
                self.tracking_error_label.text = f"{error:.3f}m"
                self.tracking_error_label.style = {"color": color}
                
        except Exception as e:
            print(f"更新UI数据失败: {e}")

# ==================== 使用示例 ====================

class ComprehensiveVisualizationDemo:
    """综合可视化演示"""
    
    def __init__(self, stage):
        self.stage = stage
        self.effects = IsaacSimVisualEffects(stage)
        self.ui_enhancer = IsaacSimUIEnhancer()
        
    async def run_full_demo(self):
        """运行完整演示"""
        print("开始综合可视化演示...")
        
        # 1. 创建UI
        self.ui_enhancer.create_performance_hud()
        
        # 2. 创建全息轨迹
        trajectory = [np.array([i*0.5, np.sin(i*0.3), 0.1]) for i in range(20)]
        await self.effects.create_holographic_trajectory(trajectory, "demo_trajectory")
        
        # 3. 创建扫掠体积效果
        boundary = [np.array([np.cos(i*0.314), np.sin(i*0.314)]) * 2 for i in range(20)]
        density = np.random.rand(10, 10)
        await self.effects.create_advanced_swept_volume(boundary, density, "demo_swept")
        
        # 4. 创建MPC预测场
        predicted = [np.array([i*0.3+1, np.cos(i*0.5), 0.1]) for i in range(10)]
        reference = [np.array([i*0.3, np.sin(i*0.4), 0.1]) for i in range(10)]
        await self.effects.create_mpc_prediction_field(predicted, reference, "demo_mpc")
        
        # 5. 启动所有效果
        self.effects.start_effect("demo_trajectory")
        self.effects.start_effect("demo_swept")
        self.effects.start_effect("demo_mpc")
        
        # 6. 模拟性能数据更新
        for i in range(100):
            performance_data = {
                'planning_time': 0.1 + 0.05 * np.sin(i * 0.1),
                'mpc_frequency': 30 + 5 * np.sin(i * 0.2),
                'swept_volume': 5.0 + 2.0 * np.sin(i * 0.15),
                'tracking_error': 0.1 * np.random.rand()
            }
            
            self.ui_enhancer.update_performance_data(performance_data)
            await asyncio.sleep(0.1)
        
        print("演示完成")
    
    def cleanup(self):
        """清理演示"""
        self.effects.cleanup()

# 在Isaac Sim中使用
async def run_advanced_visualization():
    """运行高级可视化演示"""
    stage = omni.usd.get_context().get_stage()
    demo = ComprehensiveVisualizationDemo(stage)
    
    try:
        await demo.run_full_demo()
        
        # 等待用户交互
        input("按Enter键结束演示...")
        
    finally:
        demo.cleanup()

if __name__ == "__main__":
    asyncio.run(run_advanced_visualization())