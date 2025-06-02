# SVSDF轨迹规划系统 - 工业级优化版本

## 项目概述
SVSDF (Swept Volume-aware Signed Distance Field) 是一个高性能的轨迹规划系统，集成了A*路径搜索、MINCO轨迹优化、扫掠体积感知规划和MPC实时控制。

## 核心特性
- 🚀 **工业级性能**: 采用Numba JIT、并行计算、GPU加速
- 🎯 **扫掠体积感知**: 基于SDF的精确碰撞检测和体积最小化
- 🔄 **实时控制**: 高频MPC控制器支持动态环境
- 🎨 **Isaac Sim集成**: 完整的仿真和可视化支持

## 项目结构

### 核心算法模块 (`core/`)
- `astar_planner.py` - A*路径搜索算法
- `sdf_calculator.py` / `sdf_calculator_optimized.py` - SDF计算引擎
- `swept_volume_analyzer.py` / `swept_volume_analyzer_optimized.py` - 扫掠体积分析器
- `minco_trajectory.py` / `minco_trajectory_optimized.py` - MINCO轨迹优化器
- `mpc_controller.py` / `mpc_controller_optimized.py` - MPC控制器
- `svsdf_planner.py` / `svsdf_planner_optimized.py` - 主规划器

### GPU加速模块
- `cuda_kernels/sdf_kernels.cu` - CUDA SDF计算内核
- `python_bindings/` - Python/C++接口
- `build_cuda.sh` - CUDA编译脚本

### 机器人模型 (`robot/`)
- `differential_robot.py` - 差分驱动机器人模型

### 工具和配置 (`utils/`)
- `config.py` - 系统配置参数
- `math_utils.py` - 数学工具函数

### 可视化模块 (`visualization/`)
- `isaac_sim_visualizer.py` - Isaac Sim集成可视化
- `professional_visualizer.py` - 专业级可视化效果
- `enhanced_materials.py` - 增强材质系统

### 测试和演示
- `simple_test.py` - 基础功能测试
- `test_core_components.py` - 核心组件测试
- `test_optimized_components.py` - 性能对比测试
- `svsdf_isaac_sim_demo.py` - Isaac Sim完整演示

## 性能指标
- **扫掠体积计算**: 7.22x 加速比
- **SDF计算**: GPU加速支持
- **MPC控制频率**: >100Hz
- **内存优化**: 智能缓存系统

## 快速开始

### 1. 基础测试
```bash
python simple_test.py
```

### 2. 性能测试
```bash
python test_core_components.py
```

### 3. Isaac Sim演示
```bash
# 需要Isaac Sim环境
python svsdf_isaac_sim_demo.py
```

### 4. CUDA编译（可选）
```bash
chmod +x build_cuda.sh
./build_cuda.sh
```

## 虚拟环境配置

如果您在Steam平台或特定虚拟环境中运行Isaac Sim，请按以下步骤配置：

### Isaac Sim虚拟环境
```bash
# 激活Isaac Sim虚拟环境
source ~/.local/share/ov/pkg/isaac_sim-*/python.sh

# 或者如果使用conda
conda activate isaac_sim_env
```

### 依赖安装
```bash
pip install numpy scipy numba matplotlib
```

## 配置选项

主要配置在 `utils/config.py` 中：
- 机器人参数（尺寸、质量、速度限制）
- SDF计算精度
- MPC控制参数
- 优化算法设置

## 开发说明

- 所有优化版本的文件以 `_optimized` 后缀命名
- 原始版本保留用于对比和备份
- 支持Isaac Sim环境和独立运行模式
- 完整的错误处理和性能监控

## 许可证
MIT License - 适用于学术研究和商业应用
