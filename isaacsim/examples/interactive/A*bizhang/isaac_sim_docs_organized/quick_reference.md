# Isaac Sim API 快速参考

生成时间: 2025-05-10 23:12:13

## 核心类
- **World**: 主要仿真世界类
- **SimulationContext**: 仿真上下文管理
- **PhysicsContext**: 物理仿真设置

## 常用操作
```python
# 创建仿真世界
world = World()
world.scene.add_default_ground_plane()

# 创建机器人
robot = world.scene.add(Franka(prim_path="/World/Franka"))

# 运行仿真
world.reset()
for i in range(1000):
    world.step(render=True)
```