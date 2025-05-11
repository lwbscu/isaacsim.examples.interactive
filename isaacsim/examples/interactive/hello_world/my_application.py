from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import carb
import os
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
import numpy as np

# 设置正确的资源路径 - 指向 4.5 目录
asset_root = "/home/lwb/isaacsim_assets/Assets/Isaac/4.5"
carb.settings.get_settings().set("/persistent/isaac/asset_root/default", asset_root)

# 验证设置和目录结构
current_setting = carb.settings.get_settings().get("/persistent/isaac/asset_root/default")
print(f"当前设置的资源路径: {current_setting}")

isaac_path = os.path.join(asset_root, "Isaac")
nvidia_path = os.path.join(asset_root, "NVIDIA")
print(f"Isaac 目录存在: {os.path.exists(isaac_path)}")
print(f"NVIDIA 目录存在: {os.path.exists(nvidia_path)}")

# 创建世界
world = World()

# 现在应该可以使用默认地面平面了
try:
    world.scene.add_default_ground_plane()
    print("成功添加默认地面平面")
except Exception as e:
    print(f"无法添加默认地面平面: {e}")
    # 使用自定义地面平面作为备选
    world.scene.add_ground_plane(
        size=100.0,
        z_position=0,
        name='ground_plane',
        prim_path='/World/groundPlane',
        static_friction=0.5,
        dynamic_friction=0.5,
        restitution=0.8,
        color=np.array([0.5, 0.5, 0.5])
    )

# 添加立方体
fancy_cube = world.scene.add(
    DynamicCuboid(
        prim_path="/World/random_cube",
        name="fancy_cube",
        position=np.array([0, 0, 1.0]),
        scale=np.array([0.5015, 0.5015, 0.5015]),
        color=np.array([0, 0, 1.0]),
    ))

# 重置世界
world.reset()

# 模拟500步
for i in range(500):
    position, orientation = fancy_cube.get_world_pose()
    linear_velocity = fancy_cube.get_linear_velocity()
    
    # 每10步打印一次信息
    if i % 10 == 0:
        print(f"步骤 {i}:")
        print("立方体位置:", position)
        print("立方体方向:", orientation)
        print("立方体速度:", linear_velocity)
        print("-" * 30)
    
    # 执行物理和渲染步骤
    world.step(render=True)

# 关闭Isaac Sim
simulation_app.close()