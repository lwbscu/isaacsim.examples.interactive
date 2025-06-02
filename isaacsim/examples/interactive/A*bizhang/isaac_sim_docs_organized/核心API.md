# Isaac Sim API - 核心API

生成时间: 2025-05-10 23:12:13

## 概述

核心API接口，包括World、PhysicsContext、SimulationContext等基础类

## 目录
- [主要类](#主要类)

## 主要类

### 类: isaacsim.core.api.PhysicsContext

```
Provides high level functions to deal with a physics scene and its settings. This will create a
   a PhysicsScene prim at the specified prim path in case there is no PhysicsScene present in the current
   stage.
   If there is a PhysicsScene present, it will discard the prim_path specified and sets the
   default settings on the current PhysicsScene found.

Args:
    physics_dt (float, optional): specifies the physics_dt of the simulation. Defaults to 1.0 / 60.0.
    prim_path (Optional[str], optional): specifies the prim path to create a PhysicsScene at,
                                         only in the case where no PhysicsScene already defined.
... [文档截断]
```

#### 主要方法:

- **_create_new_physics_scene**(self, prim_path: str)

- **_step**(self, current_time: float) -> None

- **enable_ccd**(self, flag: bool) -> None
  Enables a second broad phase after integration that makes it possible to prevent objects from tunneling

- **enable_fabric**(self, enable)

- **enable_gpu_dynamics**(self, flag: bool) -> None
  Enables gpu dynamics pipeline, required for deformables for instance.

- **enable_residual_reporting**(self, flag: bool)
  Set the physx scene flag to enable/disable solver residual reporting.

- **enable_stablization**(self, flag: bool) -> None
  Enables additional stabilization pass in the solver.

- **get_bounce_threshold**(self) -> float
  [summary]

- **get_broadphase_type**(self) -> str
  Gets current broadcast phase algorithm type.

- **get_current_physics_scene_prim**(self) -> Optional[pxr.Usd.Prim]
  Used to return the PhysicsScene prim in stage by traversing the stage.

### 类: isaacsim.core.api.SimulationContext

```
This class provide functions that take care of many time-related events such as
perform a physics or a render step for instance. Adding/ removing callback functions that
gets triggered with certain events such as a physics step, timeline event
(pause or play..etc), stage open/ close..etc.

It also includes an instance of PhysicsContext which takes care of many physics related
settings such as setting physics dt, solver type..etc.

Args:
    physics_dt (Optional[float], optional): dt between physics steps. Defaults to None.
... [文档截断]
```

#### 主要方法:

- **_init_stage**(self, physics_dt: 'Optional[float]' = None, rendering_dt: 'Optional[float]' = None, stage_units_in_meters: 'Optional[float]' = None, physics_prim_path: 'str' = '/physicsScene', sim_params: 'dict' = None, set_defaults: 'bool' = True, backend: 'str' = 'numpy', device: 'Optional[str]' = None) -> 'Usd.Stage'

- **_initialize_stage_async**(self, physics_dt: 'Optional[float]' = None, rendering_dt: 'Optional[float]' = None, stage_units_in_meters: 'Optional[float]' = None, physics_prim_path: 'str' = '/physicsScene', sim_params: 'dict' = None, set_defaults: 'bool' = True, device: 'Optional[str]' = None) -> 'Usd.Stage'

- **_on_post_physics_ready**(self, event)

- **_physics_timer_callback_fn**(self, step_size: 'int')

- **_setup_default_callback_fns**(self)

- **_stage_open_callback_fn**(self, event)

- **_timeline_timer_callback_fn**(self, event)

- **add_physics_callback**(self, callback_name: 'str', callback_fn: 'Callable[[float], None]') -> 'None'
  Add a callback which will be called before each physics step.

- **add_render_callback**(self, callback_name: 'str', callback_fn: 'Callable') -> 'None'
  Add a callback which will be called after each rendering event such as .render().

- **add_stage_callback**(self, callback_name: 'str', callback_fn: 'Callable') -> 'None'
  Add a callback which will be called after each stage event such as open/close among others

### 类: isaacsim.core.api.World

```
This class inherits from SimulationContext which provides the following.

SimulationContext provide functions that take care of many time-related events such as
perform a physics or a render step for instance. Adding/ removing callback functions that
gets triggered with certain events such as a physics step, timeline event
(pause or play..etc), stage open/ close..etc.

It also includes an instance of PhysicsContext which takes care of many physics related
settings such as setting physics dt, solver type..etc.

... [文档截断]
```

#### 主要方法:

- **_init_stage**(self, physics_dt: 'Optional[float]' = None, rendering_dt: 'Optional[float]' = None, stage_units_in_meters: 'Optional[float]' = None, physics_prim_path: 'str' = '/physicsScene', sim_params: 'dict' = None, set_defaults: 'bool' = True, backend: 'str' = 'numpy', device: 'Optional[str]' = None) -> 'Usd.Stage'

- **_initialize_stage_async**(self, physics_dt: 'Optional[float]' = None, rendering_dt: 'Optional[float]' = None, stage_units_in_meters: 'Optional[float]' = None, physics_prim_path: 'str' = '/physicsScene', sim_params: 'dict' = None, set_defaults: 'bool' = True, device: 'Optional[str]' = None) -> 'Usd.Stage'

- **_on_post_physics_ready**(self, event)

- **_physics_timer_callback_fn**(self, step_size: 'int')

- **_setup_default_callback_fns**(self)

- **_stage_open_callback_fn**(self, event)

- **_timeline_timer_callback_fn**(self, event)

- **add_physics_callback**(self, callback_name: 'str', callback_fn: 'Callable[[float], None]') -> 'None'
  Add a callback which will be called before each physics step.

- **add_render_callback**(self, callback_name: 'str', callback_fn: 'Callable') -> 'None'
  Add a callback which will be called after each rendering event such as .render().

- **add_stage_callback**(self, callback_name: 'str', callback_fn: 'Callable') -> 'None'
  Add a callback which will be called after each stage event such as open/close among others