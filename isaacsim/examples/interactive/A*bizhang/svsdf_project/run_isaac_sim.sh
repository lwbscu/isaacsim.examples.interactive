#!/bin/bash
# run_isaac_sim.sh
# Isaac Sim环境下的SVSDF系统启动脚本

echo "🚀 SVSDF系统 - Isaac Sim环境启动"
echo "=================================="

# 检查Isaac Sim路径
if [ ! -d "$HOME/isaacsim" ]; then
    echo "❌ 错误: 未找到Isaac Sim安装目录 ~/isaacsim"
    echo "请确保Isaac Sim已正确安装"
    exit 1
fi

# 切换到Isaac Sim目录
cd ~/isaacsim

# 提示用户激活conda环境
echo "请确保已激活conda环境: conda activate isaaclab_4_5_0"
echo ""

# 提供运行选项
echo "选择要运行的组件:"
echo "1. 基础功能测试 (simple_test.py)"
echo "2. 核心组件测试 (test_core_components.py)"
echo "3. 完整演示 (svsdf_isaac_sim_demo.py)"
echo "4. 自定义脚本路径"

read -p "请选择 (1-4): " choice

PROJECT_PATH="/home/lwb/isaacsim/exts/isaacsim.examples.interactive/isaacsim/examples/interactive/A*bizhang/svsdf_project"

case $choice in
    1)
        echo "运行基础功能测试..."
        ./python.sh "$PROJECT_PATH/simple_test.py"
        ;;
    2)
        echo "运行核心组件测试..."
        ./python.sh "$PROJECT_PATH/test_core_components.py"
        ;;
    3)
        echo "运行完整演示..."
        ./python.sh "$PROJECT_PATH/svsdf_isaac_sim_demo.py"
        ;;
    4)
        read -p "请输入脚本路径: " custom_path
        ./python.sh "$custom_path"
        ;;
    *)
        echo "无效选择"
        exit 1
        ;;
esac
