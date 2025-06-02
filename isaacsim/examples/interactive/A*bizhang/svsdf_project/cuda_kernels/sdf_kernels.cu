/**
 * CUDA内核函数用于高性能SDF和扫掠体积计算
 * 支持大规模并行计算和实时性能要求
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <float.h>

// 数学常量
#define PI 3.14159265358979323846f
#define EPS 1e-6f
#define MAX_OBSTACLES 1000
#define MAX_TRAJECTORY_POINTS 10000

// 结构体定义
struct Point2D {
    float x, y;
};

struct Obstacle {
    Point2D center;
    float radius;
    int type; // 0: circle, 1: rectangle
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

// 设备函数 - 数学工具
__device__ __forceinline__ float device_min(float a, float b) {
    return fminf(a, b);
}

__device__ __forceinline__ float device_max(float a, float b) {
    return fmaxf(a, b);
}

__device__ __forceinline__ float device_clamp(float x, float min_val, float max_val) {
    return fmaxf(min_val, fminf(x, max_val));
}

__device__ __forceinline__ float device_length(Point2D p) {
    return sqrtf(p.x * p.x + p.y * p.y);
}

__device__ __forceinline__ Point2D device_rotate(Point2D p, float angle) {
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);
    Point2D result;
    result.x = p.x * cos_a - p.y * sin_a;
    result.y = p.x * sin_a + p.y * cos_a;
    return result;
}

// 设备函数 - SDF计算
__device__ float sdf_circle(Point2D p, Point2D center, float radius) {
    Point2D diff = {p.x - center.x, p.y - center.y};
    return device_length(diff) - radius;
}

__device__ float sdf_box(Point2D p, Point2D center, float width, float height, float rotation) {
    // 将点转换到盒子局部坐标系
    Point2D local_p = {p.x - center.x, p.y - center.y};
    local_p = device_rotate(local_p, -rotation);
    
    // 计算到盒子边界的距离
    Point2D d = {fabsf(local_p.x) - width * 0.5f, fabsf(local_p.y) - height * 0.5f};
    Point2D max_d = {device_max(d.x, 0.0f), device_max(d.y, 0.0f)};
    return device_length(max_d) + device_min(device_max(d.x, d.y), 0.0f);
}

__device__ float sdf_robot(Point2D query_point, RobotPose robot_pose, 
                          float robot_length, float robot_width) {
    Point2D robot_center = {robot_pose.x, robot_pose.y};
    return sdf_box(query_point, robot_center, robot_length, robot_width, robot_pose.theta);
}

// CUDA内核 - 批量SDF计算
__global__ void compute_sdf_grid_kernel(float* sdf_grid, 
                                       const Obstacle* obstacles, 
                                       int num_obstacles,
                                       GridParams grid_params) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx >= grid_params.width || idy >= grid_params.height) return;
    
    // 计算世界坐标
    float world_x = grid_params.x_min + idx * grid_params.resolution;
    float world_y = grid_params.y_min + idy * grid_params.resolution;
    Point2D query_point = {world_x, world_y};
    
    // 计算到所有障碍物的最小距离
    float min_sdf = FLT_MAX;
    
    for (int i = 0; i < num_obstacles; i++) {
        float sdf_value;
        
        if (obstacles[i].type == 0) {
            // 圆形障碍物
            sdf_value = sdf_circle(query_point, obstacles[i].center, obstacles[i].radius);
        } else {
            // 矩形障碍物
            sdf_value = sdf_box(query_point, obstacles[i].center, 
                               obstacles[i].width, obstacles[i].height, 
                               obstacles[i].rotation);
        }
        
        min_sdf = device_min(min_sdf, sdf_value);
    }
    
    // 存储结果
    int grid_idx = idy * grid_params.width + idx;
    sdf_grid[grid_idx] = min_sdf;
}

// CUDA内核 - 扫掠体积密度计算
__global__ void compute_swept_volume_density_kernel(float* density_grid,
                                                   const RobotPose* trajectory,
                                                   int trajectory_length,
                                                   float robot_length,
                                                   float robot_width,
                                                   GridParams grid_params) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx >= grid_params.width || idy >= grid_params.height) return;
    
    // 计算世界坐标
    float world_x = grid_params.x_min + idx * grid_params.resolution;
    float world_y = grid_params.y_min + idy * grid_params.resolution;
    Point2D query_point = {world_x, world_y};
    
    float total_coverage_time = 0.0f;
    
    // 检查每个轨迹点的机器人覆盖
    for (int i = 0; i < trajectory_length; i++) {
        float sdf = sdf_robot(query_point, trajectory[i], robot_length, robot_width);
        
        if (sdf <= 0.0f) { // 点在机器人内部
            float dt;
            if (i < trajectory_length - 1) {
                dt = trajectory[i + 1].timestamp - trajectory[i].timestamp;
            } else {
                dt = 0.1f; // 默认时间步长
            }
            total_coverage_time += dt;
        }
    }
    
    // 存储密度结果
    int grid_idx = idy * grid_params.width + idx;
    density_grid[grid_idx] = total_coverage_time;
}

// CUDA内核 - 机器人覆盖检测（用于碰撞检测）
__global__ void check_robot_collision_kernel(bool* collision_results,
                                            const RobotPose* trajectory,
                                            int trajectory_length,
                                            const Obstacle* obstacles,
                                            int num_obstacles,
                                            float robot_length,
                                            float robot_width,
                                            float safety_margin) {
    int traj_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (traj_idx >= trajectory_length) return;
    
    RobotPose robot_pose = trajectory[traj_idx];
    bool has_collision = false;
    
    // 机器人的四个角点（局部坐标）
    Point2D local_corners[4] = {
        {-robot_length * 0.5f, -robot_width * 0.5f},
        { robot_length * 0.5f, -robot_width * 0.5f},
        { robot_length * 0.5f,  robot_width * 0.5f},
        {-robot_length * 0.5f,  robot_width * 0.5f}
    };
    
    // 检查与每个障碍物的碰撞
    for (int obs_idx = 0; obs_idx < num_obstacles && !has_collision; obs_idx++) {
        // 检查机器人中心到障碍物的距离
        Point2D robot_center = {robot_pose.x, robot_pose.y};
        float center_sdf;
        
        if (obstacles[obs_idx].type == 0) {
            center_sdf = sdf_circle(robot_center, obstacles[obs_idx].center, 
                                   obstacles[obs_idx].radius);
        } else {
            center_sdf = sdf_box(robot_center, obstacles[obs_idx].center,
                                obstacles[obs_idx].width, obstacles[obs_idx].height,
                                obstacles[obs_idx].rotation);
        }
        
        // 如果中心距离小于机器人半径+安全裕度，需要详细检查
        float robot_radius = sqrtf(robot_length * robot_length + robot_width * robot_width) * 0.5f;
        if (center_sdf < robot_radius + safety_margin) {
            // 检查机器人的四个角点
            for (int corner_idx = 0; corner_idx < 4; corner_idx++) {
                Point2D local_corner = local_corners[corner_idx];
                Point2D world_corner = device_rotate(local_corner, robot_pose.theta);
                world_corner.x += robot_pose.x;
                world_corner.y += robot_pose.y;
                
                float corner_sdf;
                if (obstacles[obs_idx].type == 0) {
                    corner_sdf = sdf_circle(world_corner, obstacles[obs_idx].center,
                                           obstacles[obs_idx].radius);
                } else {
                    corner_sdf = sdf_box(world_corner, obstacles[obs_idx].center,
                                        obstacles[obs_idx].width, obstacles[obs_idx].height,
                                        obstacles[obs_idx].rotation);
                }
                
                if (corner_sdf < safety_margin) {
                    has_collision = true;
                    break;
                }
            }
        }
    }
    
    collision_results[traj_idx] = has_collision;
}

// CUDA内核 - 梯度计算（用于优化）
__global__ void compute_sdf_gradient_kernel(float* gradient_x, float* gradient_y,
                                           const Obstacle* obstacles,
                                           int num_obstacles,
                                           GridParams grid_params,
                                           float h) { // 有限差分步长
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx >= grid_params.width || idy >= grid_params.height) return;
    
    // 计算世界坐标
    float world_x = grid_params.x_min + idx * grid_params.resolution;
    float world_y = grid_params.y_min + idy * grid_params.resolution;
    
    // 计算x方向梯度
    Point2D p_plus_x = {world_x + h, world_y};
    Point2D p_minus_x = {world_x - h, world_y};
    
    float sdf_plus_x = FLT_MAX, sdf_minus_x = FLT_MAX;
    
    for (int i = 0; i < num_obstacles; i++) {
        float sdf_val_plus, sdf_val_minus;
        
        if (obstacles[i].type == 0) {
            sdf_val_plus = sdf_circle(p_plus_x, obstacles[i].center, obstacles[i].radius);
            sdf_val_minus = sdf_circle(p_minus_x, obstacles[i].center, obstacles[i].radius);
        } else {
            sdf_val_plus = sdf_box(p_plus_x, obstacles[i].center, 
                                  obstacles[i].width, obstacles[i].height, 
                                  obstacles[i].rotation);
            sdf_val_minus = sdf_box(p_minus_x, obstacles[i].center,
                                   obstacles[i].width, obstacles[i].height,
                                   obstacles[i].rotation);
        }
        
        sdf_plus_x = device_min(sdf_plus_x, sdf_val_plus);
        sdf_minus_x = device_min(sdf_minus_x, sdf_val_minus);
    }
    
    // 计算y方向梯度
    Point2D p_plus_y = {world_x, world_y + h};
    Point2D p_minus_y = {world_x, world_y - h};
    
    float sdf_plus_y = FLT_MAX, sdf_minus_y = FLT_MAX;
    
    for (int i = 0; i < num_obstacles; i++) {
        float sdf_val_plus, sdf_val_minus;
        
        if (obstacles[i].type == 0) {
            sdf_val_plus = sdf_circle(p_plus_y, obstacles[i].center, obstacles[i].radius);
            sdf_val_minus = sdf_circle(p_minus_y, obstacles[i].center, obstacles[i].radius);
        } else {
            sdf_val_plus = sdf_box(p_plus_y, obstacles[i].center,
                                  obstacles[i].width, obstacles[i].height,
                                  obstacles[i].rotation);
            sdf_val_minus = sdf_box(p_minus_y, obstacles[i].center,
                                   obstacles[i].width, obstacles[i].height,
                                   obstacles[i].rotation);
        }
        
        sdf_plus_y = device_min(sdf_plus_y, sdf_val_plus);
        sdf_minus_y = device_min(sdf_minus_y, sdf_val_minus);
    }
    
    // 存储梯度结果
    int grid_idx = idy * grid_params.width + idx;
    gradient_x[grid_idx] = (sdf_plus_x - sdf_minus_x) / (2.0f * h);
    gradient_y[grid_idx] = (sdf_plus_y - sdf_minus_y) / (2.0f * h);
}

// CUDA内核 - 轨迹平滑度评估
__global__ void evaluate_trajectory_smoothness_kernel(float* smoothness_values,
                                                     const RobotPose* trajectory,
                                                     int trajectory_length,
                                                     float weight_position,
                                                     float weight_orientation) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= trajectory_length - 2) return; // 需要三个点来计算曲率
    
    // 获取三个连续的轨迹点
    RobotPose p1 = trajectory[idx];
    RobotPose p2 = trajectory[idx + 1];
    RobotPose p3 = trajectory[idx + 2];
    
    // 计算位置曲率
    Point2D v1 = {p2.x - p1.x, p2.y - p1.y};
    Point2D v2 = {p3.x - p2.x, p3.y - p2.y};
    
    float v1_len = device_length(v1);
    float v2_len = device_length(v2);
    
    float position_curvature = 0.0f;
    if (v1_len > EPS && v2_len > EPS) {
        float cross_product = v1.x * v2.y - v1.y * v2.x;
        position_curvature = fabsf(cross_product) / (v1_len * v2_len);
    }
    
    // 计算角度变化率
    float angle_change1 = p2.theta - p1.theta;
    float angle_change2 = p3.theta - p2.theta;
    
    // 标准化角度到[-pi, pi]
    while (angle_change1 > PI) angle_change1 -= 2.0f * PI;
    while (angle_change1 < -PI) angle_change1 += 2.0f * PI;
    while (angle_change2 > PI) angle_change2 -= 2.0f * PI;
    while (angle_change2 < -PI) angle_change2 += 2.0f * PI;
    
    float angular_acceleration = fabsf(angle_change2 - angle_change1);
    
    // 综合平滑度评分（越小越平滑）
    float smoothness = weight_position * position_curvature + weight_orientation * angular_acceleration;
    
    smoothness_values[idx] = smoothness;
}

// Host函数声明
extern "C" {
    // SDF计算相关
    void launch_sdf_grid_kernel(float* sdf_grid, 
                               const Obstacle* obstacles, 
                               int num_obstacles,
                               GridParams grid_params,
                               dim3 grid_size, dim3 block_size);
    
    void launch_sdf_gradient_kernel(float* gradient_x, float* gradient_y,
                                   const Obstacle* obstacles,
                                   int num_obstacles,
                                   GridParams grid_params,
                                   float h,
                                   dim3 grid_size, dim3 block_size);
    
    // 扫掠体积计算相关
    void launch_swept_volume_density_kernel(float* density_grid,
                                           const RobotPose* trajectory,
                                           int trajectory_length,
                                           float robot_length,
                                           float robot_width,
                                           GridParams grid_params,
                                           dim3 grid_size, dim3 block_size);
    
    // 碰撞检测相关
    void launch_collision_check_kernel(bool* collision_results,
                                      const RobotPose* trajectory,
                                      int trajectory_length,
                                      const Obstacle* obstacles,
                                      int num_obstacles,
                                      float robot_length,
                                      float robot_width,
                                      float safety_margin,
                                      dim3 grid_size, dim3 block_size);
    
    // 轨迹评估相关
    void launch_trajectory_smoothness_kernel(float* smoothness_values,
                                           const RobotPose* trajectory,
                                           int trajectory_length,
                                           float weight_position,
                                           float weight_orientation,
                                           dim3 grid_size, dim3 block_size);
}

// Host函数实现
void launch_sdf_grid_kernel(float* sdf_grid, 
                           const Obstacle* obstacles, 
                           int num_obstacles,
                           GridParams grid_params,
                           dim3 grid_size, dim3 block_size) {
    compute_sdf_grid_kernel<<<grid_size, block_size>>>(
        sdf_grid, obstacles, num_obstacles, grid_params);
}

void launch_sdf_gradient_kernel(float* gradient_x, float* gradient_y,
                               const Obstacle* obstacles,
                               int num_obstacles,
                               GridParams grid_params,
                               float h,
                               dim3 grid_size, dim3 block_size) {
    compute_sdf_gradient_kernel<<<grid_size, block_size>>>(
        gradient_x, gradient_y, obstacles, num_obstacles, grid_params, h);
}

void launch_swept_volume_density_kernel(float* density_grid,
                                       const RobotPose* trajectory,
                                       int trajectory_length,
                                       float robot_length,
                                       float robot_width,
                                       GridParams grid_params,
                                       dim3 grid_size, dim3 block_size) {
    compute_swept_volume_density_kernel<<<grid_size, block_size>>>(
        density_grid, trajectory, trajectory_length, 
        robot_length, robot_width, grid_params);
}

void launch_collision_check_kernel(bool* collision_results,
                                  const RobotPose* trajectory,
                                  int trajectory_length,
                                  const Obstacle* obstacles,
                                  int num_obstacles,
                                  float robot_length,
                                  float robot_width,
                                  float safety_margin,
                                  dim3 grid_size, dim3 block_size) {
    check_robot_collision_kernel<<<grid_size, block_size>>>(
        collision_results, trajectory, trajectory_length,
        obstacles, num_obstacles, robot_length, robot_width, safety_margin);
}

void launch_trajectory_smoothness_kernel(float* smoothness_values,
                                       const RobotPose* trajectory,
                                       int trajectory_length,
                                       float weight_position,
                                       float weight_orientation,
                                       dim3 grid_size, dim3 block_size) {
    evaluate_trajectory_smoothness_kernel<<<grid_size, block_size>>>(
        smoothness_values, trajectory, trajectory_length,
        weight_position, weight_orientation);
}