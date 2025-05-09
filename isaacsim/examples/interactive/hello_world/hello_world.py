# Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from isaacsim.examples.interactive.base_sample import BaseSample
import numpy as np
# Can be used to create a new cube or to point to an already existing cube in stage.
from isaacsim.core.api.objects import DynamicCuboid

class HelloWorld(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        return

    def setup_scene(self):
        # 初始化3D场景元素
        world = self.get_world()
        world.scene.add_default_ground_plane()  # 添加默认地面
        
        # 创建可交互的蓝色立方体
        fancy_cube = world.scene.add(
            DynamicCuboid(
                prim_path="/World/random_cube",  # USD舞台中的立方体路径
                name="fancy_cube",              # 对象唯一标识名称
                position=np.array([0, 0, 1.0]), # 初始位置（单位：米）
                scale=np.array([0.5015, 0.5015, 0.5015]),  # XYZ轴缩放比例
                color=np.array([0, 0, 1.0]),     # RGB颜色值（0-1范围）
            ))
        return

    async def setup_post_load(self):
        return

    async def setup_pre_reset(self):
        return

    async def setup_post_reset(self):
        return

    def world_cleanup(self):
        return
