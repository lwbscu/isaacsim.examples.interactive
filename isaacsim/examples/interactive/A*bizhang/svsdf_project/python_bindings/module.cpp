/**
 * 主模块配置文件
 * 为SVSDF项目提供C++/CUDA扩展
 */

#include <pybind11/pybind11.h>

namespace py = pybind11;

// 外部模块声明
void init_cuda_sdf(py::module &);

PYBIND11_MODULE(svsdf_cuda, m) {
    m.doc() = "SVSDF CUDA Extensions for High-Performance Computation";
    
    // 初始化CUDA SDF模块
    init_cuda_sdf(m);
    
    // 版本信息
    m.attr("__version__") = "1.0.0";
    m.attr("__author__") = "SVSDF Development Team";
    
    // 添加常量
    m.attr("PI") = 3.14159265358979323846;
    m.attr("DEFAULT_RESOLUTION") = 0.1;
    m.attr("DEFAULT_SAFETY_MARGIN") = 0.1;
    
    // 添加实用函数
    m.def("check_cuda_available", []() {
        int device_count;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        return (err == cudaSuccess && device_count > 0);
    }, "Check if CUDA devices are available");
    
    m.def("get_cuda_device_count", []() {
        int device_count;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        return (err == cudaSuccess) ? device_count : 0;
    }, "Get number of available CUDA devices");
    
    m.def("get_cuda_device_info", [](int device_id = 0) -> py::dict {
        py::dict info;
        
        int device_count;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_id >= device_count) {
            info["available"] = false;
            info["error"] = "Invalid device ID or no CUDA devices";
            return info;
        }
        
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, device_id);
        if (err != cudaSuccess) {
            info["available"] = false;
            info["error"] = "Failed to get device properties";
            return info;
        }
        
        info["available"] = true;
        info["name"] = std::string(prop.name);
        info["compute_capability"] = std::to_string(prop.major) + "." + std::to_string(prop.minor);
        info["total_memory_mb"] = prop.totalGlobalMem / (1024 * 1024);
        info["multiprocessor_count"] = prop.multiProcessorCount;
        info["max_threads_per_block"] = prop.maxThreadsPerBlock;
        info["max_grid_size"] = py::make_tuple(prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        info["warp_size"] = prop.warpSize;
        
        return info;
    }, "Get CUDA device information", py::arg("device_id") = 0);
}