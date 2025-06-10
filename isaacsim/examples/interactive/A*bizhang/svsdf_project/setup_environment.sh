#!/bin/bash
# setup_environment.sh
# SVSDF项目环境设置脚本

echo "🚀 SVSDF项目环境设置"
echo "=========================="

# 检测当前Python环境
echo "当前Python环境信息："
echo "Python路径: $(which python)"
echo "Python版本: $(python --version)"
echo "虚拟环境: ${VIRTUAL_ENV:-未激活}"

# 检测Isaac Sim环境
if [ -n "$ISAAC_SIM_PATH" ]; then
    echo "Isaac Sim路径: $ISAAC_SIM_PATH"
elif [ -d "$HOME/.local/share/ov/pkg" ]; then
    ISAAC_DIRS=$(find "$HOME/.local/share/ov/pkg" -name "isaac_sim-*" -type d 2>/dev/null)
    if [ -n "$ISAAC_DIRS" ]; then
        echo "检测到Isaac Sim安装："
        echo "$ISAAC_DIRS"
        
        # 询问是否激活Isaac Sim环境
        read -p "是否激活Isaac Sim Python环境？(y/n): " activate_isaac
        if [ "$activate_isaac" = "y" ] || [ "$activate_isaac" = "Y" ]; then
            ISAAC_PYTHON_SH=$(find "$HOME/.local/share/ov/pkg" -name "python.sh" | head -1)
            if [ -f "$ISAAC_PYTHON_SH" ]; then
                echo "激活Isaac Sim Python环境..."
                source "$ISAAC_PYTHON_SH"
                echo "Isaac Sim环境已激活"
            fi
        fi
    fi
else
    echo "未检测到Isaac Sim安装"
fi

# 检查必要的Python包
echo ""
echo "检查Python依赖..."
check_package() {
    python -c "import $1; print('✓ $1')" 2>/dev/null || echo "✗ $1 (需要安装)"
}

check_package "numpy"
check_package "scipy"
check_package "numba"
check_package "matplotlib"

# 询问是否安装缺失的包
echo ""
read -p "是否安装缺失的Python包？(y/n): " install_deps
if [ "$install_deps" = "y" ] || [ "$install_deps" = "Y" ]; then
    echo "安装Python依赖..."
    pip install numpy scipy numba matplotlib
fi

# 清理编译缓存
echo ""
echo "清理编译缓存..."
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

echo ""
echo "环境设置完成！"
echo "你现在可以运行以下命令测试系统："
echo "  python simple_test.py          # 基础功能测试"
echo "  python test_core_components.py # 核心组件测试"
echo "  python svsdf_isaac_sim_demo.py # Isaac Sim完整演示"
