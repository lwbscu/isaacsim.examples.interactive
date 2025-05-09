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
from isaacsim.examples.interactive.omnigraph_keyboard import OmnigraphKeyboard


class OmnigraphKeyboardExtension(omni.ext.IExt):
    def on_startup(self, ext_id: str):

        self.example_name = "Omnigraph Keyboard"
        self.category = "Input Devices"

        overview = "This Example shows how to change the size of a cube using the keyboard through omnigraph progrmaming in Isaac Sim."
        overview += "\n\tKeybord Input:"
        overview += "\n\t\ta: Grow"
        overview += "\n\t\td: Shrink"
        overview += "\n\nPress the 'Open in IDE' button to view the source code."
        overview += "\nOpen Visual Scripting Window to see Omnigraph"

        ui_kwargs = {
            "ext_id": ext_id,
            "file_path": os.path.abspath(__file__),
            "title": "Omnigraph Keyboard Example",
            "doc_link": "https://docs.isaacsim.omniverse.nvidia.com/latest/introduction/examples.html",
            "overview": overview,
            "sample": OmnigraphKeyboard(),
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
