/**
 * Python绑定文件用于CUDA SDF计算
 * 提供Python接口访问高性能CUDA内核
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <stdexcept>

namespace py = pybind11;

// 前向声明CUDA内核启动函数
extern "C" {
    struct Point2D {
        float x, y;
    };
    
    struct Obstacle {
        Point2D center;
        float radius;
        int type;
        float width, height, rotation;
    };
    
    struct RobotPose {
        float x, y, theta;
        float timestamp;
    };
    
    struct GridParams {
        float x_min, y_min, x_max, y_max;
        float resolution;
        int width, height;
    };
    
    void launch_sdf_grid_kernel(float* sdf_grid, 
                               const Obstacle* obstacles, 
                               int num_obstacles,
                               GridParams grid_params,
                               dim3 grid_size, dim3 block_size);
    
    void launch_swept_volume_density_kernel(float* density_grid,
                                           const RobotPose* trajectory,
                                           int trajectory_length,
                                           float robot_length,
                                           float robot_width,
                                           GridParams grid_params,
                                           dim3 grid_size, dim3 block_size);
    
    void launch_collision_check_kernel(bool* collision_results,
                                      const RobotPose* trajectory,
                                      int trajectory_length,
                                      const Obstacle* obstacles,
                                      int num_obstacles,
                                      float robot_length,
                                      float robot_width,
                                      float safety_margin,
                                      dim3 grid_size, dim3 block_size);
    
    void launch_sdf_gradient_kernel(float* gradient_x, float* gradient_y,
                                   const Obstacle* obstacles,
                                   int num_obstacles,
                                   GridParams grid_params,
                                   float h,
                                   dim3 grid_size, dim3 block_size);
    
    void launch_trajectory_smoothness_kernel(float* smoothness_values,
                                           const RobotPose* trajectory,
                                           int trajectory_length,
                                           float weight_position,
                                           float weight_orientation,
                                           dim3 grid_size, dim3 block_size);
}

/**
 * CUDA设备内存管理类
 */
template<typename T>
class CudaDeviceMemory {
private:
    T* device_ptr;
    size_t size;
    
public:
    CudaDeviceMemory(size_t count) : size(count * sizeof(T)) {
        cudaError_t err = cudaMalloc(&device_ptr, size);
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA memory allocation failed: " + 
                                   std::string(cudaGetErrorString(err)));
        }
    }
    
    ~CudaDeviceMemory() {
        if (device_ptr) {
            cudaFree(device_ptr);
        }
    }
    
    T* get() { return device_ptr; }
    
    void copy_from_host(const T* host_ptr, size_t count) {
        cudaError_t err = cudaMemcpy(device_ptr, host_ptr, 
                                   count * sizeof(T), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA host to device copy failed: " + 
                                   std::string(cudaGetErrorString(err)));
        }
    }
    
    void copy_to_host(T* host_ptr, size_t count) {
        cudaError_t err = cudaMemcpy(host_ptr, device_ptr, 
                                   count * sizeof(T), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA device to host copy failed: " + 
                                   std::string(cudaGetErrorString(err)));
        }
    }
};

/**
 * CUDA SDF计算器类
 */
class CudaSDFCalculator {
private:
    std::vector<Obstacle> obstacles;
    GridParams grid_params;
    bool initialized;
    
public:
    CudaSDFCalculator() : initialized(false) {
        // 检查CUDA设备
        int device_count;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count == 0) {
            throw std::runtime_error("No CUDA devices available");
        }
        initialized = true;
    }
    
    void add_circle_obstacle(float center_x, float center_y, float radius) {
        Obstacle obs;
        obs.center = {center_x, center_y};
        obs.radius = radius;
        obs.type = 0; // circle
        obs.width = obs.height = obs.rotation = 0.0f;
        obstacles.push_back(obs);
    }
    
    void add_box_obstacle(float center_x, float center_y, 
                         float width, float height, float rotation = 0.0f) {
        Obstacle obs;
        obs.center = {center_x, center_y};
        obs.width = width;
        obs.height = height;
        obs.rotation = rotation;
        obs.type = 1; // box
        obs.radius = 0.0f;
        obstacles.push_back(obs);
    }
    
    void clear_obstacles() {
        obstacles.clear();
    }
    
    py::array_t<float> compute_sdf_grid(float x_min, float y_min, 
                                       float x_max, float y_max, 
                                       float resolution) {
        if (!initialized) {
            throw std::runtime_error("CUDA SDF Calculator not initialized");
        }
        
        // 设置网格参数
        grid_params.x_min = x_min;
        grid_params.y_min = y_min;
        grid_params.x_max = x_max;
        grid_params.y_max = y_max;
        grid_params.resolution = resolution;
        grid_params.width = static_cast<int>((x_max - x_min) / resolution) + 1;
        grid_params.height = static_cast<int>((y_max - y_min) / resolution) + 1;
        
        size_t grid_size = grid_params.width * grid_params.height;
        
        // 分配设备内存
        CudaDeviceMemory<float> device_sdf_grid(grid_size);
        CudaDeviceMemory<Obstacle> device_obstacles(obstacles.size());
        
        // 复制障碍物数据到设备
        device_obstacles.copy_from_host(obstacles.data(), obstacles.size());
        
        // 设置CUDA内核启动参数
        dim3 block_size(16, 16);
        dim3 grid_size_cuda((grid_params.width + block_size.x - 1) / block_size.x,
                           (grid_params.height + block_size.y - 1) / block_size.y);
        
        // 启动CUDA内核
        launch_sdf_grid_kernel(device_sdf_grid.get(), 
                              device_obstacles.get(), 
                              obstacles.size(),
                              grid_params,
                              grid_size_cuda, block_size);
        
        // 同步GPU
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA kernel execution failed: " + 
                                   std::string(cudaGetErrorString(err)));
        }
        
        // 分配主机内存并复制结果
        auto result = py::array_t<float>({grid_params.height, grid_params.width});
        device_sdf_grid.copy_to_host(static_cast<float*>(result.mutable_unchecked().mutable_data(0, 0)), 
                                    grid_size);
        
        return result;
    }
    
    py::tuple compute_sdf_gradient(float x_min, float y_min, 
                                  float x_max, float y_max, 
                                  float resolution, float h = 0.01f) {
        if (!initialized) {
            throw std::runtime_error("CUDA SDF Calculator not initialized");
        }
        
        // 设置网格参数
        grid_params.x_min = x_min;
        grid_params.y_min = y_min;
        grid_params.x_max = x_max;
        grid_params.y_max = y_max;
        grid_params.resolution = resolution;
        grid_params.width = static_cast<int>((x_max - x_min) / resolution) + 1;
        grid_params.height = static_cast<int>((y_max - y_min) / resolution) + 1;
        
        size_t grid_size = grid_params.width * grid_params.height;
        
        // 分配设备内存
        CudaDeviceMemory<float> device_grad_x(grid_size);
        CudaDeviceMemory<float> device_grad_y(grid_size);
        CudaDeviceMemory<Obstacle> device_obstacles(obstacles.size());
        
        // 复制障碍物数据到设备
        device_obstacles.copy_from_host(obstacles.data(), obstacles.size());
        
        // 设置CUDA内核启动参数
        dim3 block_size(16, 16);
        dim3 grid_size_cuda((grid_params.width + block_size.x - 1) / block_size.x,
                           (grid_params.height + block_size.y - 1) / block_size.y);
        
        // 启动CUDA内核
        launch_sdf_gradient_kernel(device_grad_x.get(), device_grad_y.get(),
                                  device_obstacles.get(), 
                                  obstacles.size(),
                                  grid_params, h,
                                  grid_size_cuda, block_size);
        
        // 同步GPU
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA kernel execution failed: " + 
                                   std::string(cudaGetErrorString(err)));
        }
        
        // 分配主机内存并复制结果
        auto grad_x = py::array_t<float>({grid_params.height, grid_params.width});
        auto grad_y = py::array_t<float>({grid_params.height, grid_params.width});
        
        device_grad_x.copy_to_host(static_cast<float*>(grad_x.mutable_unchecked().mutable_data(0, 0)), 
                                  grid_size);
        device_grad_y.copy_to_host(static_cast<float*>(grad_y.mutable_unchecked().mutable_data(0, 0)), 
                                  grid_size);
        
        return py::make_tuple(grad_x, grad_y);
    }
};

/**
 * CUDA扫掠体积计算器类
 */
class CudaSweptVolumeCalculator {
private:
    bool initialized;
    
public:
    CudaSweptVolumeCalculator() : initialized(false) {
        // 检查CUDA设备
        int device_count;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count == 0) {
            throw std::runtime_error("No CUDA devices available");
        }
        initialized = true;
    }
    
    py::array_t<float> compute_density_grid(py::array_t<float> trajectory_array,
                                           float robot_length, float robot_width,
                                           float x_min, float y_min,
                                           float x_max, float y_max,
                                           float resolution) {
        if (!initialized) {
            throw std::runtime_error("CUDA Swept Volume Calculator not initialized");
        }
        
        // 解析轨迹数据
        auto trajectory_buf = trajectory_array.request();
        if (trajectory_buf.ndim != 2 || trajectory_buf.shape[1] < 4) {
            throw std::runtime_error("Trajectory array should be Nx4 (x, y, theta, time)");
        }
        
        int trajectory_length = trajectory_buf.shape[0];
        float* trajectory_data = static_cast<float*>(trajectory_buf.ptr);
        
        // 转换为RobotPose结构体
        std::vector<RobotPose> trajectory(trajectory_length);
        for (int i = 0; i < trajectory_length; i++) {
            trajectory[i].x = trajectory_data[i * 4 + 0];
            trajectory[i].y = trajectory_data[i * 4 + 1];
            trajectory[i].theta = trajectory_data[i * 4 + 2];
            trajectory[i].timestamp = trajectory_data[i * 4 + 3];
        }
        
        // 设置网格参数
        GridParams grid_params;
        grid_params.x_min = x_min;
        grid_params.y_min = y_min;
        grid_params.x_max = x_max;
        grid_params.y_max = y_max;
        grid_params.resolution = resolution;
        grid_params.width = static_cast<int>((x_max - x_min) / resolution) + 1;
        grid_params.height = static_cast<int>((y_max - y_min) / resolution) + 1;
        
        size_t grid_size = grid_params.width * grid_params.height;
        
        // 分配设备内存
        CudaDeviceMemory<float> device_density_grid(grid_size);
        CudaDeviceMemory<RobotPose> device_trajectory(trajectory_length);
        
        // 复制轨迹数据到设备
        device_trajectory.copy_from_host(trajectory.data(), trajectory_length);
        
        // 设置CUDA内核启动参数
        dim3 block_size(16, 16);
        dim3 grid_size_cuda((grid_params.width + block_size.x - 1) / block_size.x,
                           (grid_params.height + block_size.y - 1) / block_size.y);
        
        // 启动CUDA内核
        launch_swept_volume_density_kernel(device_density_grid.get(),
                                         device_trajectory.get(),
                                         trajectory_length,
                                         robot_length, robot_width,
                                         grid_params,
                                         grid_size_cuda, block_size);
        
        // 同步GPU
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA kernel execution failed: " + 
                                   std::string(cudaGetErrorString(err)));
        }
        
        // 分配主机内存并复制结果
        auto result = py::array_t<float>({grid_params.height, grid_params.width});
        device_density_grid.copy_to_host(static_cast<float*>(result.mutable_unchecked().mutable_data(0, 0)), 
                                        grid_size);
        
        return result;
    }
    
    py::array_t<bool> check_trajectory_collision(py::array_t<float> trajectory_array,
                                                py::array_t<float> obstacles_array,
                                                float robot_length, float robot_width,
                                                float safety_margin = 0.1f) {
        if (!initialized) {
            throw std::runtime_error("CUDA Swept Volume Calculator not initialized");
        }
        
        // 解析轨迹数据
        auto trajectory_buf = trajectory_array.request();
        if (trajectory_buf.ndim != 2 || trajectory_buf.shape[1] < 4) {
            throw std::runtime_error("Trajectory array should be Nx4 (x, y, theta, time)");
        }
        
        int trajectory_length = trajectory_buf.shape[0];
        float* trajectory_data = static_cast<float*>(trajectory_buf.ptr);
        
        // 转换为RobotPose结构体
        std::vector<RobotPose> trajectory(trajectory_length);
        for (int i = 0; i < trajectory_length; i++) {
            trajectory[i].x = trajectory_data[i * 4 + 0];
            trajectory[i].y = trajectory_data[i * 4 + 1];
            trajectory[i].theta = trajectory_data[i * 4 + 2];
            trajectory[i].timestamp = trajectory_data[i * 4 + 3];
        }
        
        // 解析障碍物数据 (假设格式为 Nx6: center_x, center_y, radius/width, height, rotation, type)
        auto obstacles_buf = obstacles_array.request();
        if (obstacles_buf.ndim != 2 || obstacles_buf.shape[1] < 6) {
            throw std::runtime_error("Obstacles array should be Nx6");
        }
        
        int num_obstacles = obstacles_buf.shape[0];
        float* obstacles_data = static_cast<float*>(obstacles_buf.ptr);
        
        std::vector<Obstacle> obstacles(num_obstacles);
        for (int i = 0; i < num_obstacles; i++) {
            obstacles[i].center.x = obstacles_data[i * 6 + 0];
            obstacles[i].center.y = obstacles_data[i * 6 + 1];
            obstacles[i].radius = obstacles_data[i * 6 + 2];
            obstacles[i].width = obstacles_data[i * 6 + 3];
            obstacles[i].height = obstacles_data[i * 6 + 4];
            obstacles[i].rotation = obstacles_data[i * 6 + 5];
            obstacles[i].type = static_cast<int>(obstacles_data[i * 6 + 2] > 0 ? 0 : 1); // 简化类型判断
        }
        
        // 分配设备内存
        CudaDeviceMemory<bool> device_collision_results(trajectory_length);
        CudaDeviceMemory<RobotPose> device_trajectory(trajectory_length);
        CudaDeviceMemory<Obstacle> device_obstacles(num_obstacles);
        
        // 复制数据到设备
        device_trajectory.copy_from_host(trajectory.data(), trajectory_length);
        device_obstacles.copy_from_host(obstacles.data(), num_obstacles);
        
        // 设置CUDA内核启动参数
        dim3 block_size(256);
        dim3 grid_size_cuda((trajectory_length + block_size.x - 1) / block_size.x);
        
        // 启动CUDA内核
        launch_collision_check_kernel(device_collision_results.get(),
                                     device_trajectory.get(),
                                     trajectory_length,
                                     device_obstacles.get(),
                                     num_obstacles,
                                     robot_length, robot_width,
                                     safety_margin,
                                     grid_size_cuda, block_size);
        
        // 同步GPU
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA kernel execution failed: " + 
                                   std::string(cudaGetErrorString(err)));
        }
        
        // 分配主机内存并复制结果
        auto result = py::array_t<bool>(trajectory_length);
        device_collision_results.copy_to_host(static_cast<bool*>(result.mutable_unchecked().mutable_data(0)), 
                                             trajectory_length);
        
        return result;
    }
    
    py::array_t<float> evaluate_trajectory_smoothness(py::array_t<float> trajectory_array,
                                                     float weight_position = 1.0f,
                                                     float weight_orientation = 1.0f) {
        if (!initialized) {
            throw std::runtime_error("CUDA Swept Volume Calculator not initialized");
        }
        
        // 解析轨迹数据
        auto trajectory_buf = trajectory_array.request();
        if (trajectory_buf.ndim != 2 || trajectory_buf.shape[1] < 4) {
            throw std::runtime_error("Trajectory array should be Nx4 (x, y, theta, time)");
        }
        
        int trajectory_length = trajectory_buf.shape[0];
        if (trajectory_length < 3) {
            throw std::runtime_error("Trajectory must have at least 3 points for smoothness evaluation");
        }
        
        float* trajectory_data = static_cast<float*>(trajectory_buf.ptr);
        
        // 转换为RobotPose结构体
        std::vector<RobotPose> trajectory(trajectory_length);
        for (int i = 0; i < trajectory_length; i++) {
            trajectory[i].x = trajectory_data[i * 4 + 0];
            trajectory[i].y = trajectory_data[i * 4 + 1];
            trajectory[i].theta = trajectory_data[i * 4 + 2];
            trajectory[i].timestamp = trajectory_data[i * 4 + 3];
        }
        
        int smoothness_length = trajectory_length - 2; // 需要三个点计算曲率
        
        // 分配设备内存
        CudaDeviceMemory<float> device_smoothness_values(smoothness_length);
        CudaDeviceMemory<RobotPose> device_trajectory(trajectory_length);
        
        // 复制轨迹数据到设备
        device_trajectory.copy_from_host(trajectory.data(), trajectory_length);
        
        // 设置CUDA内核启动参数
        dim3 block_size(256);
        dim3 grid_size_cuda((smoothness_length + block_size.x - 1) / block_size.x);
        
        // 启动CUDA内核
        launch_trajectory_smoothness_kernel(device_smoothness_values.get(),
                                          device_trajectory.get(),
                                          trajectory_length,
                                          weight_position,
                                          weight_orientation,
                                          grid_size_cuda, block_size);
        
        // 同步GPU
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA kernel execution failed: " + 
                                   std::string(cudaGetErrorString(err)));
        }
        
        // 分配主机内存并复制结果
        auto result = py::array_t<float>(smoothness_length);
        device_smoothness_values.copy_to_host(static_cast<float*>(result.mutable_unchecked().mutable_data(0)), 
                                             smoothness_length);
        
        return result;
    }
};

// Python模块绑定
PYBIND11_MODULE(cuda_sdf, m) {
    m.doc() = "CUDA-accelerated SDF and Swept Volume calculations";
    
    // CUDA SDF计算器
    py::class_<CudaSDFCalculator>(m, "CudaSDFCalculator")
        .def(py::init<>())
        .def("add_circle_obstacle", &CudaSDFCalculator::add_circle_obstacle,
             "Add a circular obstacle",
             py::arg("center_x"), py::arg("center_y"), py::arg("radius"))
        .def("add_box_obstacle", &CudaSDFCalculator::add_box_obstacle,
             "Add a rectangular obstacle",
             py::arg("center_x"), py::arg("center_y"), 
             py::arg("width"), py::arg("height"), py::arg("rotation") = 0.0f)
        .def("clear_obstacles", &CudaSDFCalculator::clear_obstacles,
             "Clear all obstacles")
        .def("compute_sdf_grid", &CudaSDFCalculator::compute_sdf_grid,
             "Compute SDF grid using CUDA",
             py::arg("x_min"), py::arg("y_min"), py::arg("x_max"), py::arg("y_max"), 
             py::arg("resolution"))
        .def("compute_sdf_gradient", &CudaSDFCalculator::compute_sdf_gradient,
             "Compute SDF gradient using CUDA",
             py::arg("x_min"), py::arg("y_min"), py::arg("x_max"), py::arg("y_max"), 
             py::arg("resolution"), py::arg("h") = 0.01f);
    
    // CUDA扫掠体积计算器
    py::class_<CudaSweptVolumeCalculator>(m, "CudaSweptVolumeCalculator")
        .def(py::init<>())
        .def("compute_density_grid", &CudaSweptVolumeCalculator::compute_density_grid,
             "Compute swept volume density grid using CUDA",
             py::arg("trajectory_array"), py::arg("robot_length"), py::arg("robot_width"),
             py::arg("x_min"), py::arg("y_min"), py::arg("x_max"), py::arg("y_max"),
             py::arg("resolution"))
        .def("check_trajectory_collision", &CudaSweptVolumeCalculator::check_trajectory_collision,
             "Check trajectory collision using CUDA",
             py::arg("trajectory_array"), py::arg("obstacles_array"),
             py::arg("robot_length"), py::arg("robot_width"), 
             py::arg("safety_margin") = 0.1f)
        .def("evaluate_trajectory_smoothness", &CudaSweptVolumeCalculator::evaluate_trajectory_smoothness,
             "Evaluate trajectory smoothness using CUDA",
             py::arg("trajectory_array"), 
             py::arg("weight_position") = 1.0f, py::arg("weight_orientation") = 1.0f);
}