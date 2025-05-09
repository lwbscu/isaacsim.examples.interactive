# Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import asyncio
import os

import omni.ext
import omni.ui as ui
from isaacsim.examples.browser import get_instance as get_browser_instance
from isaacsim.examples.interactive.base_sample import BaseSampleUITemplate
from isaacsim.examples.interactive.simple_stack import SimpleStack
from isaacsim.gui.components.ui_utils import btn_builder


class SimpleStackExtension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        self.example_name = "Simple Stack"
        self.category = "Manipulation"

        ui_kwargs = {
            "ext_id": ext_id,
            "file_path": os.path.abspath(__file__),
            "title": "Simple Stack",
            "doc_link": "https://docs.isaacsim.omniverse.nvidia.com/latest/core_api_tutorials/tutorial_core_adding_manipulator.html",
            "overview": "This Example shows how to stack two cubes using Franka robot in Isaac Sim.\n\nPress the 'Open in IDE' button to view the source code.",
            "sample": SimpleStack(),
        }

        ui_handle = SimpleStackUI(**ui_kwargs)

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


class SimpleStackUI(BaseSampleUITemplate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_extra_frames(self):

        extra_stacks = self.get_extra_frames_handle()
        self.task_ui_elements = {}

        with extra_stacks:
            with ui.CollapsableFrame(
                title="Task Control",
                width=ui.Fraction(0.33),
                height=0,
                visible=True,
                collapsed=False,
                # style=get_style(),
                horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
            ):
                self.build_task_controls_ui()

    def _on_stacking_button_event(self):
        asyncio.ensure_future(self.sample._on_stacking_event_async())
        self.task_ui_elements["Start Stacking"].enabled = False
        return

    def post_reset_button_event(self):
        self.task_ui_elements["Start Stacking"].enabled = True
        return

    def post_load_button_event(self):
        self.task_ui_elements["Start Stacking"].enabled = True
        return

    def post_clear_button_event(self):
        self.task_ui_elements["Start Stacking"].enabled = False
        return

    def build_task_controls_ui(self):
        with ui.VStack(spacing=5):
            dict = {
                "label": "Start Stacking",
                "type": "button",
                "text": "Start Stacking",
                "tooltip": "Start Stacking",
                "on_clicked_fn": self._on_stacking_button_event,
            }

            self.task_ui_elements["Start Stacking"] = btn_builder(**dict)
            self.task_ui_elements["Start Stacking"].enabled = False
