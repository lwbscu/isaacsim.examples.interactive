# Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import os

import omni.ext
from isaacsim.examples.browser import get_instance as get_browser_instance
from isaacsim.examples.interactive.base_sample import BaseSampleUITemplate
from isaacsim.examples.interactive.kaya_gamepad import KayaGamepad


class KayaGamepadExtension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        self.example_name = "Kaya Gamepad"
        self.category = "Input Devices"

        overview = "This Example shows how to drive a NVIDIA Kaya robot using a Gamepad in Isaac Sim."
        overview += "\n\nConnect a gamepad to the robot, and the press PLAY to begin simulating."
        overview += "\n\nPress the 'Open in IDE' button to view the source code."

        ui_kwargs = {
            "ext_id": ext_id,
            "file_path": os.path.abspath(__file__),
            "title": "NVIDIA Kaya Gamepad Example",
            "doc_link": "https://docs.isaacsim.omniverse.nvidia.com/latest/introduction/examples.html",
            "overview": overview,
            "sample": KayaGamepad(),
        }

        ui_handle = BaseSampleUITemplate(**ui_kwargs)

        # register the example with examples browser
        get_browser_instance().register_example(
            name=self.example_name,
            execute_entrypoint=ui_handle.build_window,
            ui_hook=ui_handle.build_ui,
            category=self.category,
        )

        return

    def on_shutdown(self):
        get_browser_instance().deregister_example(name=self.example_name, category=self.category)

        return
