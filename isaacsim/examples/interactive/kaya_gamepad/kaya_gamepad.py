# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#


import carb
import omni.graph.core as og
import omni.usd
from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.examples.interactive.base_sample import BaseSample
from isaacsim.storage.native import get_assets_root_path
from pxr import Sdf, UsdLux


class KayaGamepad(BaseSample):
    def __init__(self) -> None:
        super().__init__()

    def setup_scene(self):
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            return

        # add kaya robot rigged with gamepad controller
        kaya_ogn_usd = assets_root_path + "/Isaac/Robots/Kaya/kaya_ogn_gamepad.usd"
        stage = omni.usd.get_context().get_stage()
        graph_prim = stage.DefinePrim("/kaya", "Xform")
        graph_prim.GetReferences().AddReference(kaya_ogn_usd)

        # add ground plane and light
        GroundPlane("/World/ground_plane", visible=True)
        dome_light = stage.DefinePrim("/World/DomeLight", "DomeLight")
        dome_light.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(450.0)

    def world_cleanup(self):
        pass
