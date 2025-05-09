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
from isaacsim.examples.interactive.robo_party import RoboParty
from isaacsim.gui.components.ui_utils import btn_builder


class RoboPartyExtension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        self.example_name = "RoboParty"
        self.category = "Multi-Robot"

        ui_kwargs = {
            "ext_id": ext_id,
            "file_path": os.path.abspath(__file__),
            "title": "RoboParty",
            "doc_link": "https://docs.isaacsim.omniverse.nvidia.com/latest/core_api_tutorials/tutorial_core_adding_multiple_robots.html",
            "overview": "This Example shows how to run multiple tasks in the same scene.\n\nPress 'LOAD' to load the scene, \npress 'START PARTY' to start moving the robots ",
            "sample": RoboParty(),
        }

        ui_handle = RoboPartyUI(**ui_kwargs)

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


class RoboPartyUI(BaseSampleUITemplate):
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

    def _on_start_party_button_event(self):
        asyncio.ensure_future(self.sample._on_start_party_event_async())
        self.task_ui_elements["Start Party"].enabled = False
        return

    def post_reset_button_event(self):
        self.task_ui_elements["Start Party"].enabled = True
        return

    def post_load_button_event(self):
        self.task_ui_elements["Start Party"].enabled = True
        return

    def post_clear_button_event(self):
        self.task_ui_elements["Start Party"].enabled = False
        return

    def build_task_controls_ui(self):
        with ui.VStack(spacing=5):

            dict = {
                "label": "Start Party",
                "type": "button",
                "text": "Start Party",
                "tooltip": "Start Party",
                "on_clicked_fn": self._on_start_party_button_event,
            }

            self.task_ui_elements["Start Party"] = btn_builder(**dict)
            self.task_ui_elements["Start Party"].enabled = False
