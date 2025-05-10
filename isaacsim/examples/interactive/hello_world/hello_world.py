# Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

# 从 isaacsim 的交互式示例基础模块导入 BaseSample 类
from isaacsim.examples.interactive.base_sample import BaseSample
# 导入 numpy 库，用于处理数值计算和数组操作
import numpy as np
# 尝试从 isaacsim 核心 API 对象模块导入 DynamicCuboid 类
from isaacsim.core.api.objects import DynamicCuboid

# 定义 HelloWorld 类，继承自 BaseSample 类
class HelloWorld(BaseSample):
    def __init__(self) -> None:
        """
        类的构造函数，初始化 HelloWorld 类的实例。
        调用父类的构造函数进行初始化。
        """
        # 调用父类 BaseSample 的构造函数
        super().__init__()
        return

    def setup_scene(self):
        """
        设置场景，添加默认地面平面和一个动态立方体。
        """
        # 获取当前世界实例
        world = self.get_world()
        # 向场景中添加默认的地面平面
        world.scene.add_default_ground_plane()
        # 向场景中添加一个动态立方体
        fancy_cube = world.scene.add(
            DynamicCuboid(
                # 立方体在场景中的 prim 路径
                prim_path="/World/random_cube",
                # 立方体的名称
                name="fancy_cube",
                # 立方体的初始位置
                position=np.array([0, 0, 1.0]),
                # 立方体的缩放比例
                scale=np.array([0.5015, 0.5015, 0.5015]),
                # 立方体的颜色
                color=np.array([0, 0, 1.0]),
            ))
        return

    async def setup_post_load(self):
        """
        场景加载完成后进行设置，获取立方体对象并添加物理回调。
        """
        # 获取当前世界实例
        self._world = self.get_world()
        # 从场景中获取名为 fancy_cube 的对象
        self._cube = self._world.scene.get_object("fancy_cube")
        # 向世界中添加一个物理回调，在每个物理步之前调用 print_cube_info 函数
        self._world.add_physics_callback("sim_step", callback_fn=self.print_cube_info) #callback names have to be unique
        return

    # 这里我们定义的物理回调会在每个物理步之前被调用，所有物理回调都必须接受 step_size 作为参数
    def print_cube_info(self, step_size):
        """
        物理回调函数，打印立方体的位置、方向和线速度信息。

        :param step_size: 物理模拟的步长
        """
        # 获取立方体在世界坐标系下的位置和方向
        position, orientation = self._cube.get_world_pose()
        # 获取立方体的线速度
        linear_velocity = self._cube.get_linear_velocity()
        # 在终端显示立方体的位置信息
        print("Cube position is : " + str(position))
        # 在终端显示立方体的方向信息
        print("Cube's orientation is : " + str(orientation))
        # 在终端显示立方体的线速度信息
        print("Cube's linear velocity is : " + str(linear_velocity))

    async def setup_pre_reset(self):
        """
        在场景重置前进行设置，当前为空实现。
        """
        return

    async def setup_post_reset(self):
        """
        在场景重置后进行设置，当前为空实现。
        """
        return

    def world_cleanup(self):
        """
        清理世界资源，当前为空实现。
        """
        return
