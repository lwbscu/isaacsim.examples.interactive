# 在任何其他导入之前启动Isaac Sim
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
import numpy as np
import omni.usd
from pxr import Gf, UsdLux, UsdGeom

# 创建世界
world = World()

# 获取舞台
stage = omni.usd.get_context().get_stage()

# 设置更明亮、更鲜艳的地面
ground = world.scene.add(
    FixedCuboid(
        prim_path="/World/ground",
        name="ground",
        position=np.array([0, 0, -0.05]),
        scale=np.array([20.0, 20.0, 0.1]),
        color=np.array([0.2, 0.7, 0.2]),  # 鲜绿色地面
    ))

# 创建格子状地面标记以增强深度感知 - 避免使用负数索引
for i in range(0, 10, 2):
    for j in range(0, 10, 2):
        if (i + j) % 2 == 0:  # 棋盘格模式
            # 在一个象限放置格子，避免命名问题
            world.scene.add(
                FixedCuboid(
                    prim_path=f"/World/grid_{i}_{j}",
                    name=f"grid_{i}_{j}",
                    position=np.array([i, j, -0.04]),
                    scale=np.array([1.8, 1.8, 0.01]),
                    color=np.array([0.1, 0.5, 0.1]),  # 深绿色
                ))

# 添加更鲜艳的立方体
fancy_cube = world.scene.add(
    DynamicCuboid(
        prim_path="/World/random_cube",
        name="fancy_cube",
        position=np.array([0, 0, 3.0]),  # 提高起始高度
        scale=np.array([1.0, 1.0, 1.0]),  # 增大尺寸
        color=np.array([1.0, 0.2, 0.2]),  # 鲜红色
    ))

# 添加第二个立方体作为参考
reference_cube = world.scene.add(
    DynamicCuboid(
        prim_path="/World/reference_cube",
        name="reference_cube",
        position=np.array([3.0, 0, 1.0]),
        scale=np.array([0.5, 0.5, 2.0]),  # 垂直长方体
        color=np.array([0.2, 0.2, 1.0]),  # 蓝色
    ))

# 添加多个光源使场景更明亮

# 1. 添加顶光(更亮)
dist_light = UsdLux.DistantLight.Define(stage, "/World/Lights/DistantLight")
dist_light.CreateIntensityAttr(3000.0)  # 非常高的强度
dist_light.CreateAngleAttr(0.53)
dist_light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))  # 白光

# 2. 添加球形光源在场景中心
sphere_light = UsdLux.SphereLight.Define(stage, "/World/Lights/SphereLight")
sphere_light.CreateIntensityAttr(20000.0)
sphere_light.CreateRadiusAttr(1.0)
sphere_light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 0.9))  # 微黄色光
sphere_light.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, 5.0))

# 3. 添加定向光源从侧面打光
key_light = UsdLux.DistantLight.Define(stage, "/World/Lights/KeyLight")
key_light.CreateIntensityAttr(2000.0)
key_light.CreateColorAttr(Gf.Vec3f(0.9, 0.9, 1.0))  # 微蓝色光
key_light.AddRotateXOp().Set(45.0)
key_light.AddRotateYOp().Set(45.0)

# 使用USD API直接设置相机位置，而不是使用MoveCameraCommand
# 获取默认相机
from pxr import Sdf, UsdGeom
camera_path = "/OmniverseKit_Persp"
if stage.GetPrimAtPath(camera_path):
    camera = UsdGeom.Camera(stage.GetPrimAtPath(camera_path))
    # 设置相机变换
    camera_xform = UsdGeom.Xformable(camera.GetPrim())
    camera_xform.ClearXformOpOrder()
    camera_xform.AddTranslateOp().Set(Gf.Vec3d(8, 8, 5))
    # 看向场景中心
    from pxr import UsdGeom
    aim = UsdGeom.Camera.CreateAimAttr(camera)
    aim.Set(Gf.Vec3f(0, 0, 0))

# 重置世界
world.reset()

print("开始模拟，立方体将从空中落下...")
print("请等待Isaac Sim窗口打开并完全加载（可能需要几秒钟）")

# 修复重力设置方法
physx = world.get_physics_context()
# 使用标量分别设置各个轴的重力
physx.set_gravity_vector(0.0, 0.0, -3.0)  # 使用指定各分量的方法

# 模拟300步
for i in range(5000):
    position, orientation = fancy_cube.get_world_pose()
    linear_velocity = fancy_cube.get_linear_velocity()
    
    # 每20步打印一次信息
    if i % 20 == 0:
        print(f"步骤 {i}:")
        print("立方体位置:", position)
        print("-" * 30)
    
    # 步进物理和渲染
    world.step(render=True)
    if i < 100:  # 只在开始阶段减慢
        import time
        time.sleep(0.02)  # 20毫秒延迟

print("\n模拟完成!")
print("程序将保持运行8秒钟，然后自动关闭")

# 保持窗口显示8秒
import time
time.sleep(8)