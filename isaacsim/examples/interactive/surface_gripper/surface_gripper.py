# Copyright (c) 2018-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import asyncio
import weakref

import numpy as np
import omni
import omni.ext
import omni.kit.commands
import omni.kit.usd
import omni.physics.tensors as physics
import omni.physx as _physx
import omni.ui as ui
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.examples.browser import get_instance as get_browser_instance
from isaacsim.gui.components.menu import make_menu_item_description
from isaacsim.gui.components.ui_utils import (
    add_separator,
    btn_builder,
    combo_floatfield_slider_builder,
    get_style,
    setup_ui_headers,
    state_btn_builder,
)

# Import extension python module we are testing with absolute import path, as if we are external user (other extension)
from isaacsim.robot.surface_gripper._surface_gripper import Surface_Gripper, Surface_Gripper_Properties
from omni.kit.menu.utils import MenuItemDescription, add_menu_items, remove_menu_items
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics

EXTENSION_NAME = "Surface Gripper"


class Extension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        """Initialize extension and UI elements"""

        self._ext_id = ext_id

        # Loads interfaces
        self._timeline = omni.timeline.get_timeline_interface()
        self._usd_context = omni.usd.get_context()
        self._stage_event_sub = None
        self._window = None
        self._models = {}

        # register the example with examples browser
        get_browser_instance().register_example(
            name=EXTENSION_NAME, execute_entrypoint=self.build_window, ui_hook=self.build_ui, category="Manipulation"
        )

        self.surface_gripper = None
        self.cone = None
        self.box = None
        self._stage_id = -1

    def build_window(self):
        pass

    def build_ui(self):
        self._usd_context = omni.usd.get_context()
        if self._usd_context is not None:
            self._stage_event_sub = (
                omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(self._on_update_ui)
            )

        with ui.VStack(spacing=5, height=0):
            title = "Surface Gripper Example"
            doc_link = "https://docs.isaacsim.omniverse.nvidia.com/latest/robot_simulation/ext_isaacsim_robot_surface_gripper.html"

            overview = "This Example shows how to simulate a suction-cup gripper in Isaac Sim. "
            overview += "It simulates suction by creating a Joint between two bodies when the parent and child bodies are close at the gripper's point of contact."
            overview += "\n\nPress the 'Open in IDE' button to view the source code."

            setup_ui_headers(self._ext_id, __file__, title, doc_link, overview, info_collapsed=False)

            frame = ui.CollapsableFrame(
                title="Command Panel",
                height=0,
                collapsed=False,
                style=get_style(),
                style_type_name_override="CollapsableFrame",
                horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
            )
            with frame:
                with ui.VStack(style=get_style(), spacing=5):

                    args = {
                        "label": "Load Scene",
                        "type": "button",
                        "text": "Load",
                        "tooltip": "Load a gripper into the Scene",
                        "on_clicked_fn": self._on_create_scenario_button_clicked,
                    }
                    self._models["create_button"] = btn_builder(**args)

                    args = {
                        "label": "Gripper State",
                        "type": "button",
                        "a_text": "Close",
                        "b_text": "Open",
                        "tooltip": "Open and Close the Gripper",
                        "on_clicked_fn": self._on_toggle_gripper_button_clicked,
                    }
                    self._models["toggle_button"] = state_btn_builder(**args)

                    add_separator()

                    args = {
                        "label": "Gripper Force (UP)",
                        "default_val": 0,
                        "min": 0,
                        "max": 1.0e2,
                        "step": 1,
                        "tooltip": ["Force in ()", "Force in ()"],
                    }
                    self._models["force_slider"], slider = combo_floatfield_slider_builder(**args)

                    args = {
                        "label": "Set Force",
                        "type": "button",
                        "text": "APPLY",
                        "tooltip": "Apply the Gripper Force to the Z-Axis of the Cone",
                        "on_clicked_fn": self._on_force_button_clicked,
                    }
                    self._models["force_button"] = btn_builder(**args)

                    args = {
                        "label": "Gripper Speed (UP)",
                        "default_val": 0,
                        "min": 0,
                        "max": 5.0e1,
                        "step": 1,
                        "tooltip": ["Speed in ()", "Speed in ()"],
                    }

                    add_separator()

                    self._models["speed_slider"], slider = combo_floatfield_slider_builder(**args)

                    args = {
                        "label": "Set Speed",
                        "type": "button",
                        "text": "APPLY",
                        "tooltip": "Apply Cone Velocity in the Z-Axis",
                        "on_clicked_fn": self._on_speed_button_clicked,
                    }
                    self._models["speed_button"] = btn_builder(**args)

                    ui.Spacer()

    def on_shutdown(self):
        self._physx_subs = None
        self._stage_event_sub = None
        self._window = None
        get_browser_instance().deregister_example(name=EXTENSION_NAME, category="Manipulation")

    def _on_update_ui(self, widget):
        self._models["create_button"].enabled = self._timeline.is_playing()
        self._models["toggle_button"].enabled = self._timeline.is_playing()
        self._models["force_button"].enabled = self._timeline.is_playing()
        self._models["speed_button"].enabled = self._timeline.is_playing()
        # If the scene has been reloaded, reset UI to create Scenario
        if self._usd_context.get_stage_id() != self._stage_id:
            self._models["create_button"].enabled = True
            # self._models["create_button"].text = "Create Scenario"
            self._models["create_button"].set_tooltip("Creates a new scenario with the cone on top of the Cube")
            self._models["create_button"].set_clicked_fn(self._on_create_scenario_button_clicked)
            self.cone = None
            self.box = None
            self._stage_id = -1

    def _toggle_gripper_button_ui(self):
        # Checks if the surface gripper has been created
        if self.surface_gripper is not None:
            if self.surface_gripper.is_closed():
                self._models["toggle_button"].text = "OPEN"
            else:
                self._models["toggle_button"].text = "CLOSE"
        pass

    def _on_simulation_step(self, step):
        # Checks if the simulation is playing, and if the stage has been loaded
        if self._timeline.is_playing() and self._stage_id != -1:
            # Check if the handles for cone and box have been loaded
            if self.cone is None:
                self.cone = SingleRigidPrim("/GripperCone")
                self.box = SingleRigidPrim("/Box")

            # If the surface Gripper has been created, update wheter it has been broken or not
            if self.surface_gripper is not None:
                self.surface_gripper.update()
                if self.surface_gripper.is_closed():
                    self.coneGeom.GetDisplayColorAttr().Set([self.color_closed])
                else:
                    self.coneGeom.GetDisplayColorAttr().Set([self.color_open])
                self._toggle_gripper_button_ui()

    def _on_reset_scenario_button_clicked(self):
        if self._timeline.is_playing() and self._stage_id != -1:
            if self.surface_gripper is not None:
                self.surface_gripper.open()
            self.cone.set_linear_velocity([0, 0, 0])
            self.box.set_linear_velocity([0, 0, 0])
            self.cone.set_angular_velocity([0, 0, 0])
            self.box.set_angular_velocity([0, 0, 0])
            self.cone.set_world_pose(self.gripper_start_pose.p, self.gripper_start_pose.r)
            self.box.set_world_pose(self.box_start_pose.p, self.box_start_pose.r)

    async def _create_scenario(self, task):
        done, pending = await asyncio.wait({task})
        if task in done:
            # Repurpose button to reset Scene
            # self._models["create_button"].text = "Reset Scene"
            self._models["create_button"].set_tooltip("Resets scenario with the cone on top of the Cube")

            # Get Handle for stage and stage ID to check if stage was reloaded
            self._stage = self._usd_context.get_stage()
            self._stage_id = self._usd_context.get_stage_id()
            self._timeline.stop()
            self._models["create_button"].set_clicked_fn(self._on_reset_scenario_button_clicked)

            # Adds a light to the scene
            distantLight = UsdLux.DistantLight.Define(self._stage, Sdf.Path("/DistantLight"))
            distantLight.CreateIntensityAttr(500)
            distantLight.AddOrientOp().Set(Gf.Quatf(-0.3748, -0.42060, -0.0716, 0.823))

            # Set up stage with Z up, treat units as cm, set up gravity and ground plane
            UsdGeom.SetStageUpAxis(self._stage, UsdGeom.Tokens.z)
            UsdGeom.SetStageMetersPerUnit(self._stage, 1.0)
            self.scene = UsdPhysics.Scene.Define(self._stage, Sdf.Path("/physicsScene"))
            self.scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
            self.scene.CreateGravityMagnitudeAttr().Set(9.81)
            omni.kit.commands.execute(
                "AddGroundPlaneCommand",
                stage=self._stage,
                planePath="/groundPlane",
                axis="Z",
                size=10.000,
                position=Gf.Vec3f(0),
                color=Gf.Vec3f(0.5),
            )
            # Colors to represent when gripper is open or closed
            self.color_closed = Gf.Vec3f(1.0, 0.2, 0.2)
            self.color_open = Gf.Vec3f(0.2, 1.0, 0.2)

            # Cone that will represent the gripper
            self.gripper_start_pose = physics.Transform([0, 0, 0.301], [1, 0, 0, 0])
            self.coneGeom = self.createRigidBody(
                UsdGeom.Cone,
                "/GripperCone",
                0.100,
                [0.10, 0.10, 0.10],
                self.gripper_start_pose.p,
                self.gripper_start_pose.r,
                self.color_open,
            )

            # Box to be picked
            self.box_start_pose = physics.Transform([0, 0, 0.10], [1, 0, 0, 0])
            self.boxGeom = self.createRigidBody(
                UsdGeom.Cube, "/Box", 0.10, [0.1, 0.1, 0.1], self.box_start_pose.p, self.box_start_pose.r, [0.2, 0.2, 1]
            )

            # Reordering the quaternion to follow DC convention for later use.
            self.gripper_start_pose = physics.Transform([0, 0, 0.301], [0, 0, 0, 1])
            self.box_start_pose = physics.Transform([0, 0, 0.10], [0, 0, 0, 1])

            # Gripper properties
            self.sgp = Surface_Gripper_Properties()
            self.sgp.d6JointPath = "/GripperCone/SurfaceGripper"
            self.sgp.parentPath = "/GripperCone"
            self.sgp.offset = physics.Transform()
            self.sgp.offset.p.x = 0
            self.sgp.offset.p.z = -0.1001
            self.sgp.offset.r = [0, 0.7071, 0, 0.7071]  # Rotate to point gripper in Z direction
            self.sgp.gripThreshold = 0.02
            self.sgp.forceLimit = 1.0e2
            self.sgp.torqueLimit = 1.0e3
            self.sgp.bendAngle = np.pi / 4
            self.sgp.stiffness = 1.0e4
            self.sgp.damping = 1.0e3

            self.surface_gripper = Surface_Gripper()
            self.surface_gripper.initialize(self.sgp)
            # Set camera to a nearby pose and looking directly at the Gripper cone
            set_camera_view(
                eye=[4.00, 4.00, 4.00], target=self.gripper_start_pose.p, camera_prim_path="/OmniverseKit_Persp"
            )

            self._physx_subs = _physx.get_physx_interface().subscribe_physics_step_events(self._on_simulation_step)
            self._timeline.play()

    def _on_create_scenario_button_clicked(self):
        # wait for new stage before creating scenario
        task = asyncio.ensure_future(omni.usd.get_context().new_stage_async())
        asyncio.ensure_future(self._create_scenario(task))

    def _on_toggle_gripper_button_clicked(self, val=False):
        if self._timeline.is_playing():
            if self.surface_gripper.is_closed():
                self.surface_gripper.open()
            else:
                self.surface_gripper.close()
            if self.surface_gripper.is_closed():
                self._models["toggle_button"].text = "OPEN"
            else:
                self._models["toggle_button"].text = "CLOSE"

    def _on_speed_button_clicked(self):
        if self._timeline.is_playing():
            self.cone.set_linear_velocity([0, 0, self._models["speed_slider"].get_value_as_float()])

    def _on_force_button_clicked(self):
        if self._timeline.is_playing():
            self.cone._rigid_prim_view.apply_forces_and_torques_at_pos(
                forces=np.array([0, 0, self._models["force_slider"].get_value_as_float()]),
                positions=np.array([10, 10, 10]),
                is_global=True,
            )

    def createRigidBody(self, bodyType, boxActorPath, mass, scale, position, rotation, color):
        p = Gf.Vec3f(position[0], position[1], position[2])
        orientation = Gf.Quatf(rotation[0], rotation[1], rotation[2], rotation[3])
        scale = Gf.Vec3f(scale[0], scale[1], scale[2])

        bodyGeom = bodyType.Define(self._stage, boxActorPath)
        bodyPrim = self._stage.GetPrimAtPath(boxActorPath)
        bodyGeom.AddTranslateOp().Set(p)
        bodyGeom.AddOrientOp().Set(orientation)
        bodyGeom.AddScaleOp().Set(scale)
        bodyGeom.CreateDisplayColorAttr().Set([color])

        UsdPhysics.CollisionAPI.Apply(bodyPrim)
        if mass > 0:
            massAPI = UsdPhysics.MassAPI.Apply(bodyPrim)
            massAPI.CreateMassAttr(mass)
        UsdPhysics.RigidBodyAPI.Apply(bodyPrim)
        UsdPhysics.CollisionAPI(bodyPrim)
        print(bodyPrim.GetPath().pathString)
        return bodyGeom
