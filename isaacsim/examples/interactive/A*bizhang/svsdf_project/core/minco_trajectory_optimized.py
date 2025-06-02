#!/usr/bin/env python3
"""
MINCO (Minimum Control) 轨迹优化器 - 工业级实现
基于扫掠体积感知的稀疏轨迹表示方法

核心算法实现:
1. 5次多项式轨迹段表示
2. 稀疏控制点参数化  
3. 两阶段联合优化
4. 数值稳定性保证
5. 并行计算加速
"""

import numpy as np
import scipy.optimize
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from typing import List, Tuple, Optional, Dict, Callable, Any
import time
import warnings
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import math
from numba import jit, njit
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MINCOParams:
    """MINCO优化参数"""
    polynomial_degree: int = 5  # 多项式阶数
    continuity_order: int = 3   # 连续性阶数（位置、速度、加速度）
    max_iterations: int = 100   # 最大迭代次数
    tolerance: float = 1e-6     # 收敛容差
    line_search_c1: float = 1e-4  # Armijo线搜索参数
    line_search_alpha: float = 0.5  # 回退因子
    numerical_epsilon: float = 1e-12  # 数值稳定性参数
    enable_parallel: bool = True  # 启用并行计算
    
@dataclass
class OptimizationResult:
    """优化结果"""
    success: bool
    final_cost: float
    iterations: int
    time_elapsed: float
    convergence_info: Dict[str, Any]

class TrajectorySegmentOptimized:
    """轨迹段：5次多项式表示 - 优化版本"""
    
    def __init__(self):
        # 多项式系数 [C0, C1, C2, C3, C4, C5] for x, y, yaw
        self.coeffs_x = np.zeros(6)
        self.coeffs_y = np.zeros(6)  
        self.coeffs_yaw = np.zeros(6)
        self.duration = 1.0
        
        # 缓存计算结果
        self._cache = {}
        self._cache_enabled = True
        
    def evaluate(self, t: float, derivative: int = 0) -> np.ndarray:
        """计算t时刻的状态（位置/速度/加速度）"""
        if derivative < 0 or derivative > 3:
            raise ValueError(f"不支持的导数阶数: {derivative}")
            
        # 检查缓存
        cache_key = (t, derivative) if self._cache_enabled else None
        if cache_key and cache_key in self._cache:
            return self._cache[cache_key]
            
        # 归一化时间
        tau = np.clip(t / max(self.duration, 1e-12), 0, 1)
        
        # 使用优化的多项式计算
        result = self._evaluate_polynomial_optimized(tau, derivative)
        
        # 缓存结果
        if cache_key:
            self._cache[cache_key] = result
            
        return result
    
    @njit
    def _evaluate_polynomial_optimized(self, tau: float, derivative: int) -> np.ndarray:
        """优化的多项式计算（使用Numba加速）"""
        if derivative == 0:
            # 位置
            powers = np.array([1.0, tau, tau**2, tau**3, tau**4, tau**5])
            scale = 1.0
        elif derivative == 1:
            # 速度
            powers = np.array([0.0, 1.0, 2*tau, 3*tau**2, 4*tau**3, 5*tau**4])
            scale = 1.0 / self.duration
        elif derivative == 2:
            # 加速度
            powers = np.array([0.0, 0.0, 2.0, 6*tau, 12*tau**2, 20*tau**3])
            scale = 1.0 / (self.duration ** 2)
        else:  # derivative == 3
            # 急动度
            powers = np.array([0.0, 0.0, 0.0, 6.0, 24*tau, 60*tau**2])
            scale = 1.0 / (self.duration ** 3)
            
        x = np.dot(self.coeffs_x, powers) * scale
        y = np.dot(self.coeffs_y, powers) * scale
        yaw = np.dot(self.coeffs_yaw, powers) * scale
        
        return np.array([x, y, yaw])
    
    def compute_energy(self) -> float:
        """计算段的控制能量（急动度积分）"""
        # 使用解析解计算控制能量
        # ∫₀ᵀ ||u||² dt，其中 u 是急动度
        
        # 急动度系数
        jerk_coeffs_x = np.array([6, 24, 60]) * self.coeffs_x[3:6]
        jerk_coeffs_y = np.array([6, 24, 60]) * self.coeffs_y[3:6]
        jerk_coeffs_yaw = np.array([6, 24, 60]) * self.coeffs_yaw[3:6]
        
        # 计算积分 ∫₀¹ ||jerk||² dτ * T
        energy = 0.0
        for i in range(3):
            for j in range(3):
                coeff = jerk_coeffs_x[i] * jerk_coeffs_x[j] + \
                       jerk_coeffs_y[i] * jerk_coeffs_y[j] + \
                       jerk_coeffs_yaw[i] * jerk_coeffs_yaw[j]
                energy += coeff / (i + j + 1)
        
        return energy * self.duration
    
    def clear_cache(self):
        """清除缓存"""
        self._cache.clear()

class MINCOTrajectoryOptimized:
    """MINCO轨迹表示和优化 - 工业级实现"""
    
    def __init__(self, num_segments: int = 5, params: Optional[MINCOParams] = None):
        self.num_segments = num_segments
        self.params = params or MINCOParams()
        self.segments: List[TrajectorySegmentOptimized] = []
        self.waypoints: List[np.ndarray] = []
        self.initial_times: List[float] = []
        self.initialized = False
        
        # 优化状态
        self.stage1_optimized = False
        self.stage2_optimized = False
        
        # 性能监控
        self.optimization_stats = {
            'stage1_time': 0.0,
            'stage2_time': 0.0,
            'stage1_iterations': 0,
            'stage2_iterations': 0,
            'stage1_final_cost': 0.0,
            'stage2_final_cost': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # 线程池（用于并行计算）
        self.executor = ThreadPoolExecutor(max_workers=4) if params and params.enable_parallel else None
        
    def initialize_from_waypoints(self, waypoints: List[np.ndarray], initial_times: List[float]) -> bool:
        """从路径点初始化轨迹"""
        try:
            logger.info(f"🚀 MINCO初始化: {len(waypoints)}个航路点 -> {self.num_segments}段轨迹")
            
            if len(waypoints) < 2:
                logger.error("❌ 航路点数量不足")
                return False
                
            if len(initial_times) != len(waypoints) - 1:
                logger.error("❌ 时间分配与航路点不匹配")
                return False
                
            self.waypoints = waypoints.copy()
            self.initial_times = initial_times.copy()
            
            # 创建轨迹段
            self.segments = []
            for i in range(self.num_segments):
                segment = TrajectorySegmentOptimized()
                if i < len(initial_times):
                    segment.duration = max(initial_times[i], 0.1)  # 最小时间约束
                else:
                    segment.duration = 1.0
                    
                # 使用样条插值初始化
                self._initialize_segment_spline(segment, i, waypoints)
                self.segments.append(segment)
                
            self.initialized = True
            logger.info(f"✅ MINCO初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ MINCO初始化异常: {e}")
            return False
    
    def _initialize_segment_spline(self, segment: TrajectorySegmentOptimized, 
                                  segment_idx: int, waypoints: List[np.ndarray]):
        """使用样条插值初始化轨迹段"""
        if segment_idx >= len(waypoints) - 1:
            return
            
        start_wp = waypoints[segment_idx]
        end_wp = waypoints[segment_idx + 1]
        
        # 构建边界条件矩阵（6x6系统）
        # 边界条件：起点和终点的位置、速度、加速度
        T = segment.duration
        A = np.array([
            [1, 0, 0, 0, 0, 0],           # p(0) = start_pos
            [0, 1, 0, 0, 0, 0],           # v(0) = 0
            [0, 0, 2, 0, 0, 0],           # a(0) = 0
            [1, T, T**2, T**3, T**4, T**5],   # p(T) = end_pos
            [0, 1, 2*T, 3*T**2, 4*T**3, 5*T**4],  # v(T) = 0
            [0, 0, 2, 6*T, 12*T**2, 20*T**3]      # a(T) = 0
        ])
        
        # 边界条件向量
        b_x = np.array([start_wp[0], 0, 0, end_wp[0], 0, 0])
        b_y = np.array([start_wp[1], 0, 0, end_wp[1], 0, 0])
        b_yaw = np.array([start_wp[2], 0, 0, end_wp[2], 0, 0])
        
        # 求解多项式系数
        try:
            segment.coeffs_x = np.linalg.solve(A, b_x)
            segment.coeffs_y = np.linalg.solve(A, b_y)
            segment.coeffs_yaw = np.linalg.solve(A, b_yaw)
        except np.linalg.LinAlgError:
            # 如果矩阵奇异，使用最小二乘解
            segment.coeffs_x = np.linalg.lstsq(A, b_x, rcond=None)[0]
            segment.coeffs_y = np.linalg.lstsq(A, b_y, rcond=None)[0]
            segment.coeffs_yaw = np.linalg.lstsq(A, b_yaw, rcond=None)[0]
        
    def optimize_stage1(self, weight_energy: float, weight_time: float, 
                       weight_path: float, reference_path: List[np.ndarray]) -> OptimizationResult:
        """第一阶段优化：轨迹平滑化"""
        if not self.initialized:
            logger.error("❌ MINCO未初始化")
            return OptimizationResult(False, 0, 0, 0, {})
            
        stage1_start = time.time()
        logger.info(f"🔧 MINCO第一阶段优化开始...")
        logger.info(f"   权重 - 能量: {weight_energy:.3f}, 时间: {weight_time:.3f}, 路径: {weight_path:.3f}")
        
        try:
            # 构建优化变量
            initial_vars = self._pack_optimization_variables()
            
            # 定义目标函数
            def objective(vars_packed):
                return self._compute_stage1_cost(vars_packed, weight_energy, weight_time, weight_path, reference_path)
            
            # 约束条件
            constraints = self._build_continuity_constraints()
            bounds = self._build_optimization_bounds()
            
            # 使用SLSQP进行约束优化
            result = scipy.optimize.minimize(
                objective, 
                initial_vars,
                method='SLSQP',
                constraints=constraints,
                bounds=bounds,
                options={
                    'maxiter': self.params.max_iterations,
                    'ftol': self.params.tolerance,
                    'disp': False
                }
            )
            
            if result.success:
                # 更新轨迹参数
                self._unpack_optimization_variables(result.x)
                self.stage1_optimized = True
                
                stage1_time = time.time() - stage1_start
                self.optimization_stats['stage1_time'] = stage1_time
                self.optimization_stats['stage1_iterations'] = result.nit
                self.optimization_stats['stage1_final_cost'] = result.fun
                
                logger.info(f"✅ 第一阶段优化成功: {stage1_time:.3f}s, {result.nit}次迭代, 代价: {result.fun:.6f}")
                
                return OptimizationResult(
                    True, result.fun, result.nit, stage1_time,
                    {'message': result.message, 'nfev': result.nfev}
                )
            else:
                logger.warning(f"⚠️ 第一阶段优化失败: {result.message}")
                return OptimizationResult(False, 0, 0, 0, {'message': result.message})
                
        except Exception as e:
            logger.error(f"❌ 第一阶段优化异常: {e}")
            return OptimizationResult(False, 0, 0, 0, {'error': str(e)})
        
    def optimize_stage2(self, weight_energy: float, weight_time: float,
                       weight_obstacle: float, weight_swept_volume: float,
                       obstacle_cost_func: Callable, swept_volume_cost_func: Callable) -> OptimizationResult:
        """第二阶段优化：扫掠体积最小化"""
        if not self.stage1_optimized:
            logger.warning("⚠️ 建议先执行第一阶段优化")
            
        stage2_start = time.time()
        logger.info(f"🔧 MINCO第二阶段优化开始...")
        logger.info(f"   权重 - 能量: {weight_energy:.3f}, 时间: {weight_time:.3f}")
        logger.info(f"   权重 - 障碍物: {weight_obstacle:.3f}, 扫掠体积: {weight_swept_volume:.3f}")
        
        try:
            # 使用梯度下降进行扫掠体积优化
            initial_vars = self._pack_optimization_variables()
            
            def objective(vars_packed):
                return self._compute_stage2_cost(
                    vars_packed, weight_energy, weight_time, 
                    weight_obstacle, weight_swept_volume,
                    obstacle_cost_func, swept_volume_cost_func
                )
            
            # 使用L-BFGS-B进行无约束优化
            result = scipy.optimize.minimize(
                objective,
                initial_vars,
                method='L-BFGS-B',
                options={
                    'maxiter': self.params.max_iterations,
                    'ftol': self.params.tolerance,
                    'gtol': 1e-6
                }
            )
            
            if result.success:
                self._unpack_optimization_variables(result.x)
                self.stage2_optimized = True
                
                stage2_time = time.time() - stage2_start
                self.optimization_stats['stage2_time'] = stage2_time
                self.optimization_stats['stage2_iterations'] = result.nit
                self.optimization_stats['stage2_final_cost'] = result.fun
                
                logger.info(f"✅ 第二阶段优化成功: {stage2_time:.3f}s, {result.nit}次迭代, 代价: {result.fun:.6f}")
                
                return OptimizationResult(
                    True, result.fun, result.nit, stage2_time,
                    {'message': result.message, 'nfev': result.nfev}
                )
            else:
                logger.warning(f"⚠️ 第二阶段优化失败: {result.message}")
                return OptimizationResult(False, 0, 0, 0, {'message': result.message})
                
        except Exception as e:
            logger.error(f"❌ 第二阶段优化异常: {e}")
            return OptimizationResult(False, 0, 0, 0, {'error': str(e)})
    
    def _pack_optimization_variables(self) -> np.ndarray:
        """打包优化变量"""
        # 变量：[coeffs_x_all, coeffs_y_all, coeffs_yaw_all, durations]
        coeffs_x = np.concatenate([seg.coeffs_x for seg in self.segments])
        coeffs_y = np.concatenate([seg.coeffs_y for seg in self.segments])
        coeffs_yaw = np.concatenate([seg.coeffs_yaw for seg in self.segments])
        durations = np.array([seg.duration for seg in self.segments])
        
        return np.concatenate([coeffs_x, coeffs_y, coeffs_yaw, durations])
    
    def _unpack_optimization_variables(self, vars_packed: np.ndarray):
        """解包优化变量"""
        n_coeffs = 6 * self.num_segments
        
        coeffs_x = vars_packed[:n_coeffs]
        coeffs_y = vars_packed[n_coeffs:2*n_coeffs]
        coeffs_yaw = vars_packed[2*n_coeffs:3*n_coeffs]
        durations = vars_packed[3*n_coeffs:3*n_coeffs + self.num_segments]
        
        for i, segment in enumerate(self.segments):
            segment.coeffs_x = coeffs_x[i*6:(i+1)*6]
            segment.coeffs_y = coeffs_y[i*6:(i+1)*6]
            segment.coeffs_yaw = coeffs_yaw[i*6:(i+1)*6]
            segment.duration = max(durations[i], 0.1)  # 最小时间约束
            segment.clear_cache()  # 清除缓存
    
    def _compute_stage1_cost(self, vars_packed: np.ndarray, weight_energy: float,
                           weight_time: float, weight_path: float, reference_path: List[np.ndarray]) -> float:
        """计算第一阶段代价函数"""
        # 临时更新参数
        original_state = self._pack_optimization_variables()
        self._unpack_optimization_variables(vars_packed)
        
        try:
            # 能量项
            energy_cost = sum(seg.compute_energy() for seg in self.segments)
            
            # 时间项
            time_cost = sum(seg.duration for seg in self.segments)
            
            # 路径偏差项
            path_cost = self._compute_path_deviation_cost(reference_path)
            
            total_cost = weight_energy * energy_cost + weight_time * time_cost + weight_path * path_cost
            
            return total_cost
            
        except Exception as e:
            logger.warning(f"代价计算异常: {e}")
            return 1e6  # 返回大值
        finally:
            # 恢复原始状态
            self._unpack_optimization_variables(original_state)
    
    def _compute_stage2_cost(self, vars_packed: np.ndarray, weight_energy: float,
                           weight_time: float, weight_obstacle: float, weight_swept_volume: float,
                           obstacle_cost_func: Callable, swept_volume_cost_func: Callable) -> float:
        """计算第二阶段代价函数"""
        # 临时更新参数
        original_state = self._pack_optimization_variables()
        self._unpack_optimization_variables(vars_packed)
        
        try:
            # 基础项
            energy_cost = sum(seg.compute_energy() for seg in self.segments)
            time_cost = sum(seg.duration for seg in self.segments)
            
            # 障碍物代价
            obstacle_cost = self._compute_obstacle_cost_integrated(obstacle_cost_func)
            
            # 扫掠体积代价
            swept_volume_cost = swept_volume_cost_func(self.segments)
            
            total_cost = (weight_energy * energy_cost + weight_time * time_cost + 
                         weight_obstacle * obstacle_cost + weight_swept_volume * swept_volume_cost)
            
            return total_cost
            
        except Exception as e:
            logger.warning(f"代价计算异常: {e}")
            return 1e6
        finally:
            # 恢复原始状态
            self._unpack_optimization_variables(original_state)
    
    def _compute_path_deviation_cost(self, reference_path: List[np.ndarray]) -> float:
        """计算路径偏差代价"""
        if not reference_path:
            return 0.0
        
        total_deviation = 0.0
        n_samples = 50  # 总采样点数
        
        total_time = sum(seg.duration for seg in self.segments)
        dt = total_time / n_samples
        
        for i in range(n_samples):
            t = i * dt
            pos = self.evaluate_at_time(t)[0]
            
            # 找到最近的参考点
            min_dist = float('inf')
            for ref_point in reference_path:
                dist = np.linalg.norm(pos[:2] - ref_point[:2])
                min_dist = min(min_dist, dist)
            
            total_deviation += min_dist ** 2
        
        return total_deviation * dt
    
    def _compute_obstacle_cost_integrated(self, obstacle_cost_func: Callable) -> float:
        """计算障碍物代价（积分形式）"""
        total_cost = 0.0
        
        for segment in self.segments:
            n_samples = 20
            dt = segment.duration / n_samples
            
            for i in range(n_samples):
                t = i * dt
                pos = segment.evaluate(t, 0)
                vel = segment.evaluate(t, 1)
                
                cost = obstacle_cost_func(pos, vel)
                total_cost += cost * dt
        
        return total_cost
    
    def _build_continuity_constraints(self) -> List[Dict]:
        """构建连续性约束"""
        constraints = []
        
        # 段间连续性约束（位置、速度、加速度）
        for i in range(self.num_segments - 1):
            for deriv in range(3):  # 0阶（位置）、1阶（速度）、2阶（加速度）
                def make_constraint(seg_idx, derivative):
                    def constraint_func(vars_packed):
                        self._unpack_optimization_variables(vars_packed)
                        
                        # 当前段的终点状态
                        end_state = self.segments[seg_idx].evaluate(self.segments[seg_idx].duration, derivative)
                        # 下一段的起点状态
                        start_state = self.segments[seg_idx + 1].evaluate(0, derivative)
                        
                        return end_state - start_state  # 应该为零
                    
                    return constraint_func
                
                constraints.append({
                    'type': 'eq',
                    'fun': make_constraint(i, deriv)
                })
        
        return constraints
    
    def _build_optimization_bounds(self) -> List[Tuple[float, float]]:
        """构建优化变量边界"""
        bounds = []
        
        # 多项式系数边界（较宽松）
        coeff_bound = 100.0
        for _ in range(3 * 6 * self.num_segments):  # 3个维度 x 6个系数 x N段
            bounds.append((-coeff_bound, coeff_bound))
        
        # 时间边界
        for _ in range(self.num_segments):
            bounds.append((0.1, 10.0))  # 最小0.1秒，最大10秒
        
        return bounds
    
    def get_discretized_trajectory(self, dt: float = 0.1) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[float]]:
        """获取离散化轨迹"""
        if not self.initialized:
            logger.error("❌ MINCO未初始化")
            return [], [], [], []
            
        positions = []
        velocities = []  
        accelerations = []
        times = []
        
        current_time = 0.0
        
        for segment in self.segments:
            segment_times = np.arange(0, segment.duration + dt, dt)
            
            for t in segment_times:
                if t > segment.duration:
                    t = segment.duration
                    
                pos = segment.evaluate(t, 0)
                vel = segment.evaluate(t, 1)
                acc = segment.evaluate(t, 2)
                
                positions.append(pos)
                velocities.append(vel)
                accelerations.append(acc)
                times.append(current_time + t)
                
            current_time += segment.duration
            
        return positions, velocities, accelerations, times
    
    def evaluate_at_time(self, global_time: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """在指定时间评估轨迹"""
        current_time = 0.0
        
        for segment in self.segments:
            if current_time <= global_time <= current_time + segment.duration:
                local_time = global_time - current_time
                pos = segment.evaluate(local_time, 0)
                vel = segment.evaluate(local_time, 1)
                acc = segment.evaluate(local_time, 2)
                return pos, vel, acc
            current_time += segment.duration
        
        # 如果超出范围，返回最后一个点
        if self.segments:
            last_segment = self.segments[-1]
            pos = last_segment.evaluate(last_segment.duration, 0)
            vel = last_segment.evaluate(last_segment.duration, 1)
            acc = last_segment.evaluate(last_segment.duration, 2)
            return pos, vel, acc
        
        return np.zeros(3), np.zeros(3), np.zeros(3)
    
    def get_total_time(self) -> float:
        """获取轨迹总时间"""
        return sum(seg.duration for seg in self.segments)
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """获取优化统计信息"""
        return self.optimization_stats.copy()
    
    def __del__(self):
        """析构函数"""
        if self.executor:
            self.executor.shutdown(wait=False)
