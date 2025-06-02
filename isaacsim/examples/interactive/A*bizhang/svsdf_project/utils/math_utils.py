# utils/math_utils.py
"""
数学工具函数模块
"""
import numpy as np
import numba
from typing import Tuple, List, Optional
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

class MathUtils:
    """数学工具类"""
    
    @staticmethod
    @numba.jit(nopython=True)
    def normalize_angle(angle: float) -> float:
        """角度归一化到[-π, π]"""
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi
        return angle
    
    @staticmethod
    @numba.jit(nopython=True)
    def euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> float:
        """计算欧几里得距离"""
        return np.sqrt(np.sum((p1 - p2) ** 2))
    
    @staticmethod
    @numba.jit(nopython=True)
    def rotation_matrix_2d(theta: float) -> np.ndarray:
        """2D旋转矩阵"""
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        return np.array([[cos_theta, -sin_theta],
                        [sin_theta, cos_theta]])
    
    @staticmethod
    @numba.jit(nopython=True)
    def transform_point_2d(point: np.ndarray, translation: np.ndarray, 
                          rotation: float) -> np.ndarray:
        """2D点变换"""
        R = MathUtils.rotation_matrix_2d(rotation)
        return R @ point + translation
    
    @staticmethod
    @numba.jit(nopython=True)
    def rectangle_sdf(point_local: np.ndarray, length: float, width: float) -> float:
        """
        矩形SDF计算（局部坐标系）
        基于论文Equation (7)的快速实现
        """
        dx = np.abs(point_local[0]) - length / 2.0
        dy = np.abs(point_local[1]) - width / 2.0
        
        if dx > 0.0 and dy > 0.0:
            # 外部角点：欧几里得距离
            return np.sqrt(dx * dx + dy * dy)
        else:
            # 边界或内部：切比雪夫距离
            return max(dx, dy)
    
    @staticmethod
    def world_to_robot_frame(point_world: np.ndarray, robot_pose: np.ndarray) -> np.ndarray:
        """世界坐标系转机器人局部坐标系"""
        x, y, theta = robot_pose[0], robot_pose[1], robot_pose[2]
        
        # 平移
        translated = point_world - np.array([x, y])
        
        # 旋转（逆变换）
        R_inv = MathUtils.rotation_matrix_2d(-theta)
        return R_inv @ translated
    
    @staticmethod
    def interpolate_trajectory(trajectory: List[np.ndarray], time: float) -> np.ndarray:
        """轨迹插值"""
        if len(trajectory) == 0:
            return np.zeros(3)
        
        if len(trajectory) == 1:
            return trajectory[0][:3]
        
        # 边界情况
        if time <= trajectory[0][3]:
            return trajectory[0][:3]
        if time >= trajectory[-1][3]:
            return trajectory[-1][:3]
        
        # 线性插值
        for i in range(len(trajectory) - 1):
            t0 = trajectory[i][3]
            t1 = trajectory[i + 1][3]
            
            if t0 <= time <= t1:
                alpha = (time - t0) / (t1 - t0)
                
                pos0 = trajectory[i][:3]
                pos1 = trajectory[i + 1][:3]
                
                # 位置线性插值
                result = np.zeros(3)
                result[:2] = (1.0 - alpha) * pos0[:2] + alpha * pos1[:2]
                
                # 角度插值（考虑周期性）
                theta0 = pos0[2]
                theta1 = pos1[2]
                dtheta = MathUtils.normalize_angle(theta1 - theta0)
                result[2] = MathUtils.normalize_angle(theta0 + alpha * dtheta)
                
                return result
        
        return trajectory[-1][:3]
    
    @staticmethod
    def compute_trajectory_curvature(positions: List[np.ndarray]) -> List[float]:
        """计算轨迹曲率"""
        if len(positions) < 3:
            return [0.0] * len(positions)
        
        curvatures = [0.0]  # 第一个点曲率为0
        
        for i in range(1, len(positions) - 1):
            p1 = positions[i - 1][:2]
            p2 = positions[i][:2]
            p3 = positions[i + 1][:2]
            
            # 计算曲率
            v1 = p2 - p1
            v2 = p3 - p2
            
            # 避免除零
            if np.linalg.norm(v1) < 1e-8 or np.linalg.norm(v2) < 1e-8:
                curvatures.append(0.0)
                continue
            
            # 曲率计算
            cross_product = v1[0] * v2[1] - v1[1] * v2[0]
            norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
            
            if norm_product > 1e-8:
                curvature = abs(cross_product) / norm_product
            else:
                curvature = 0.0
            
            curvatures.append(curvature)
        
        curvatures.append(0.0)  # 最后一个点曲率为0
        return curvatures

class OptimizationUtils:
    """优化工具类"""
    
    @staticmethod
    def armijo_line_search(f, x0: np.ndarray, d: np.ndarray, 
                          c1: float = 1e-4, alpha_max: float = 1.0,
                          max_iter: int = 50) -> float:
        """
        Armijo线搜索算法
        找到满足Armijo条件的步长
        """
        alpha = alpha_max
        f0 = f(x0)
        grad_f0_d = np.dot(f(x0 + 1e-8 * d) - f0, d) / 1e-8  # 数值梯度
        
        for _ in range(max_iter):
            if f(x0 + alpha * d) <= f0 + c1 * alpha * grad_f0_d:
                return alpha
            alpha *= 0.5
        
        return alpha
    
    @staticmethod
    def lbfgs_optimize(objective_func, gradient_func, x0: np.ndarray,
                      max_iter: int = 100, tolerance: float = 1e-6) -> np.ndarray:
        """L-BFGS优化器"""
        def combined_func(x):
            f_val = objective_func(x)
            g_val = gradient_func(x)
            return f_val, g_val
        
        result = minimize(combined_func, x0, method='L-BFGS-B', 
                         jac=True, options={'maxiter': max_iter, 'ftol': tolerance})
        
        return result.x if result.success else x0
    
    @staticmethod
    def quadratic_programming_solve(H: np.ndarray, g: np.ndarray,
                                  A: Optional[np.ndarray] = None,
                                  b: Optional[np.ndarray] = None) -> np.ndarray:
        """
        求解二次规划问题: min 0.5 * x^T * H * x + g^T * x
        s.t. A * x <= b
        """
        try:
            # 简化实现：如果H正定，直接求解
            if A is None:
                # 无约束QP
                return -np.linalg.solve(H, g)
            else:
                # 约束QP - 使用投影梯度法
                x = -np.linalg.solve(H, g)  # 无约束解
                
                # 投影到可行域
                if A is not None and b is not None:
                    violations = A @ x - b
                    for i, violation in enumerate(violations):
                        if violation > 0:
                            # 简单投影
                            normal = A[i, :]
                            x -= (violation / np.dot(normal, normal)) * normal
                
                return x
        except np.linalg.LinAlgError:
            # 如果求解失败，返回零向量
            return np.zeros(len(g))

class GeometryUtils:
    """几何工具类"""
    
    @staticmethod
    def point_in_polygon(point: np.ndarray, polygon: List[np.ndarray]) -> bool:
        """判断点是否在多边形内（射线法）"""
        x, y = point[0], point[1]
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0][0], polygon[0][1]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n][0], polygon[i % n][1]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    @staticmethod
    def polygon_area(vertices: List[np.ndarray]) -> float:
        """计算多边形面积（鞋带公式）"""
        if len(vertices) < 3:
            return 0.0
        
        area = 0.0
        n = len(vertices)
        
        for i in range(n):
            j = (i + 1) % n
            area += vertices[i][0] * vertices[j][1]
            area -= vertices[j][0] * vertices[i][1]
        
        return abs(area) / 2.0
    
    @staticmethod
    def convex_hull_2d(points: List[np.ndarray]) -> List[np.ndarray]:
        """2D凸包算法（Graham扫描）"""
        if len(points) < 3:
            return points
        
        def cross_product(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        
        # 按x坐标排序
        points = sorted(points, key=lambda p: (p[0], p[1]))
        
        # 构建下凸包
        lower = []
        for p in points:
            while len(lower) >= 2 and cross_product(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        
        # 构建上凸包
        upper = []
        for p in reversed(points):
            while len(upper) >= 2 and cross_product(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        
        # 移除重复点
        return lower[:-1] + upper[:-1]