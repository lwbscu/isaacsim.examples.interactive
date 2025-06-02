#!/bin/bash
# filepath: /home/lwb/isaacsim/exts/isaacsim.examples.interactive/isaacsim/examples/interactive/A*bizhang/svsdf_project/build_cuda.sh
# CUDA扩展构建脚本

set -e

echo "=== SVSDF CUDA Extensions Build Script ==="

# 检查CUDA是否可用
if ! command -v nvcc &> /dev/null; then
    echo "Warning: CUDA compiler (nvcc) not found. CUDA extensions will not be built."
    echo "Please install CUDA toolkit to enable GPU acceleration."
    exit 1
fi

# 检查Python开发环境
if ! python3-config --exists &> /dev/null; then
    echo "Error: Python development headers not found."
    echo "Please install python3-dev package."
    exit 1
fi

# 项目目录
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${PROJECT_DIR}/build"
CUDA_DIR="${PROJECT_DIR}/cuda_kernels"
BINDINGS_DIR="${PROJECT_DIR}/python_bindings"

echo "Project directory: $PROJECT_DIR"
echo "Build directory: $BUILD_DIR"

# 创建构建目录
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# 检测CUDA架构
echo "Detecting CUDA architecture..."
if command -v nvidia-smi &> /dev/null; then
    GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | head -1 | tr -d '.')
    if [ -z "$GPU_ARCH" ]; then
        GPU_ARCH="75"  # 默认为Turing架构
    fi
else
    GPU_ARCH="75"  # 默认为Turing架构
fi

echo "Using GPU architecture: sm_$GPU_ARCH"

# 编译CUDA内核
echo "Compiling CUDA kernels..."
nvcc -c "$CUDA_DIR/sdf_kernels.cu" \
    -o sdf_kernels.o \
    -arch=sm_$GPU_ARCH \
    -Xcompiler -fPIC \
    --expt-relaxed-constexpr \
    -O3

if [ $? -ne 0 ]; then
    echo "Error: CUDA kernel compilation failed"
    exit 1
fi

# 获取Python信息
PYTHON_INCLUDE=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
PYTHON_LIB=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

echo "Python include: $PYTHON_INCLUDE"
echo "Python lib: $PYTHON_LIB"
echo "Python version: $PYTHON_VERSION"

# 查找pybind11
PYBIND11_INCLUDE=""
if python3 -c "import pybind11" &> /dev/null; then
    PYBIND11_INCLUDE=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")
    if [ -d "$PYBIND11_INCLUDE/../include" ]; then
        PYBIND11_INCLUDE="$PYBIND11_INCLUDE/../include"
    else
        PYBIND11_INCLUDE=$(python3 -c "import pybind11; print(pybind11.get_include())")
    fi
fi

if [ -z "$PYBIND11_INCLUDE" ] || [ ! -d "$PYBIND11_INCLUDE" ]; then
    echo "Error: pybind11 not found. Please install: pip install pybind11"
    exit 1
fi

echo "pybind11 include: $PYBIND11_INCLUDE"

# 编译Python绑定
echo "Compiling Python bindings..."
g++ -shared -fPIC \
    -I"$PYTHON_INCLUDE" \
    -I"$PYBIND11_INCLUDE" \
    -I"$CUDA_DIR" \
    "$BINDINGS_DIR/sdf_binding.cpp" \
    "$BINDINGS_DIR/module.cpp" \
    sdf_kernels.o \
    -o svsdf_cuda$(python3-config --extension-suffix) \
    -L"$PYTHON_LIB" \
    -lpython$PYTHON_VERSION \
    -lcudart \
    -lcuda \
    -O3 \
    -DWITH_CUDA

if [ $? -ne 0 ]; then
    echo "Error: Python bindings compilation failed"
    exit 1
fi

# 创建安装脚本
cat > install.py << 'EOF'
#!/usr/bin/env python3
import os
import shutil
import sys

def install_module():
    build_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(build_dir)
    
    # 找到编译好的模块
    module_files = [f for f in os.listdir(build_dir) if f.startswith('svsdf_cuda') and f.endswith('.so')]
    
    if not module_files:
        print("Error: No compiled module found")
        return False
    
    module_file = module_files[0]
    src_path = os.path.join(build_dir, module_file)
    dst_path = os.path.join(project_dir, module_file)
    
    try:
        shutil.copy2(src_path, dst_path)
        print(f"Successfully installed {module_file} to {project_dir}")
        return True
    except Exception as e:
        print(f"Error installing module: {e}")
        return False

if __name__ == "__main__":
    if install_module():
        print("Installation completed successfully!")
        print("You can now use: import svsdf_cuda")
    else:
        print("Installation failed!")
        sys.exit(1)
EOF

chmod +x install.py

echo ""
echo "=== Build completed successfully! ==="
echo ""
echo "To install the module, run:"
echo "  cd $BUILD_DIR && python3 install.py"
echo ""
echo "Or manually copy the .so file to your project directory."
echo ""

# 运行测试
echo "Running basic tests..."
python3 -c "
import sys
sys.path.insert(0, '.')

try:
    import svsdf_cuda
    print('✓ Module import successful')
    
    if svsdf_cuda.check_cuda_available():
        print('✓ CUDA is available')
        device_count = svsdf_cuda.get_cuda_device_count()
        print(f'✓ Found {device_count} CUDA device(s)')
        
        device_info = svsdf_cuda.get_cuda_device_info(0)
        if device_info['available']:
            print(f'✓ GPU: {device_info[\"name\"]}')
            print(f'  Compute capability: {device_info[\"compute_capability\"]}')
            print(f'  Memory: {device_info[\"total_memory_mb\"]} MB')
    else:
        print('⚠ CUDA not available on this system')
    
    print('✓ All tests passed!')
    
except ImportError as e:
    print(f'✗ Module import failed: {e}')
except Exception as e:
    print(f'✗ Test failed: {e}')
"

echo ""
echo "Build and test completed!"
