# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import time

import omni.timeline
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.examples.interactive.base_sample import BaseSample


class GettingStartedRobot(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        self._timeline = omni.timeline.get_timeline_interface()
        self.print_state = False
        self.car_handle = None
        self.arm_handle = None
        return

    @property
    def name(self):
        return "Getting Started with a Robot"

    def setup_scene(self):
        world = self.get_world()
        world.scene.add_default_ground_plane()  # added ground, lighting

    async def setup_post_load(self):
        # move camera to a better vanatage point
        set_camera_view(eye=[5.0, 0.0, 1.5], target=[0.00, 0.00, 1.00], camera_prim_path="/OmniverseKit_Persp")

        # Add physics callback so that we can insert commands at each physics each step, such as print out the state at each step, or sending commands
        self.get_world().add_physics_callback("physics_step", callback_fn=self.on_physics_step)

        # do a quick start and stop to reset the physics timeline
        self._timeline.play()
        time.sleep(1)
        self._timeline.stop()

        return

    def on_physics_step(self, step_size) -> None:
        if self.print_state:
            if self.arm_handle:
                print("arm joint state: ", self.arm_handle.get_joint_positions())
            if self.car_handle:
                print("car joint state: ", self.car_handle.get_joint_positions())

    async def setup_pre_reset(self):
        return

    async def setup_post_reset(self):
        self._timeline.stop()
        return

    def world_cleanup(self):
        return
