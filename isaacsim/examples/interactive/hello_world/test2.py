# 检查已加载的场景
import omni.usd
stage = omni.usd.get_context().get_stage()

# 查找立方体并打印其属性
from isaacsim.core.api import World
world = World.instance()

try:
    cube = world.scene.get_object("fancy_cube")
    position, orientation = cube.get_world_pose()
    linear_velocity = cube.get_linear_velocity()
    
    print("立方体位置:", position)
    print("立方体方向:", orientation)
    print("立方体速度:", linear_velocity)
except Exception as e:
    print("未能找到立方体或获取其属性:", str(e))
    
    # 列出场景中的所有对象
    print("\n场景中的对象:")
    for obj in world.scene.get_objects():
        print(f"- {obj.name} (路径: {obj.prim_path})")