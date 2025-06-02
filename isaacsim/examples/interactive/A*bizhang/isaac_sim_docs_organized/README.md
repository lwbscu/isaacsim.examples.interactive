# Isaac Sim API 文档索引

生成时间: 2025-05-10 23:12:13

本文档集合按功能分类整理了Isaac Sim的主要API，适合学习和快速查阅。如果你需要学习isaac sim 最新api规则，请告诉我需要文档目录的哪些，我给你上传
# 设置正确的资源路径 - 指向 4.5 目录
asset_root = "/home/lwb/isaacsim_assets/Assets/Isaac/4.5"
carb.settings.get_settings().set("/persistent/isaac/asset_root/default", asset_root)

# 验证设置和目录结构
current_setting = carb.settings.get_settings().get("/persistent/isaac/asset_root/default")
print(f"当前设置的资源路径: {current_setting}")

isaac_path = os.path.join(asset_root, "Isaac")
nvidia_path = os.path.join(asset_root, "NVIDIA")
print(f"Isaac 目录存在: {os.path.exists(isaac_path)}")
print(f"NVIDIA 目录存在: {os.path.exists(nvidia_path)}")
## 文档目录
- [核心API](./核心API.md) - 核心API接口，包括World、PhysicsContext、SimulationContext等基础类
- [机器人](./机器人.md) - 机器人相关API，包括机械臂、轮式机器人、抓取器等
- [传感器](./传感器.md) - 传感器相关API，包括相机、激光雷达、接触传感器等
- [控制器](./控制器.md) - 各种控制器实现，包括关节控制、抓取控制、运动控制等
- [材质与物理](./材质与物理.md) - 材质系统和物理属性，包括视觉材质、物理材质等
- [对象与几何](./对象与几何.md) - 3D对象和几何体，包括基本形状、关节对象等
- [场景与任务](./场景与任务.md) - 场景管理和预定义任务
- [导入工具](./导入工具.md) - 资产导入工具，包括URDF、MJCF等格式导入
- [机器人示例](./机器人示例.md) - 机器人相关示例代码，包括Franka、UR10等常用机器人
- [仿真工具](./仿真工具.md) - 仿真相关工具，包括数据生成、UI组件等
- [运动规划](./运动规划.md) - 运动规划和行为控制，包括路径规划、行为树等
- [ROS2集成](./ROS2集成.md) - ROS2相关集成和桥接
