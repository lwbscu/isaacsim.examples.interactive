# core/astar_planner.py
"""
A*路径搜索算法实现
SVSDF轨迹规划的第一阶段
"""
import numpy as np
import heapq
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass
from utils.math_utils import MathUtils
from utils.config import config

@dataclass
class Node:
    """A*搜索节点"""
    x: int
    y: int
    g_cost: float = 0.0      # 从起点到当前节点的实际代价
    h_cost: float = 0.0      # 从当前节点到终点的启发式代价
    f_cost: float = 0.0      # 总代价 f = g + h
    parent: Optional['Node'] = None
    
    def __post_init__(self):
        self.f_cost = self.g_cost + self.h_cost
    
    def __lt__(self, other):
        return self.f_cost < other.f_cost
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __hash__(self):
        return hash((self.x, self.y))

class AStarPlanner:
    """
    A*路径搜索器
    实现论文中第一阶段的初始路径生成
    """
    
    def __init__(self, grid_resolution: float = 0.1, heuristic_weight: float = 1.0):
        self.grid_resolution = grid_resolution
        self.heuristic_weight = heuristic_weight
        self.obstacles = []
        self.robot_radius = max(config.robot.length, config.robot.width) / 2.0 + config.robot.safety_margin
        
        # 8邻域搜索方向
        self.directions = [
            (-1, -1, np.sqrt(2)), (-1, 0, 1), (-1, 1, np.sqrt(2)),
            (0, -1, 1),                       (0, 1, 1),
            (1, -1, np.sqrt(2)),  (1, 0, 1),  (1, 1, np.sqrt(2))
        ]
    
    def set_obstacles(self, obstacles: List[dict]):
        """设置障碍物"""
        self.obstacles = obstacles
    
    def world_to_grid(self, world_pos: np.ndarray, origin: np.ndarray) -> Tuple[int, int]:
        """世界坐标转网格坐标"""
        grid_x = int((world_pos[0] - origin[0]) / self.grid_resolution)
        grid_y = int((world_pos[1] - origin[1]) / self.grid_resolution)
        return grid_x, grid_y
    
    def grid_to_world(self, grid_pos: Tuple[int, int], origin: np.ndarray) -> np.ndarray:
        """网格坐标转世界坐标"""
        world_x = origin[0] + grid_pos[0] * self.grid_resolution
        world_y = origin[1] + grid_pos[1] * self.grid_resolution
        return np.array([world_x, world_y])
    
    def is_collision_free(self, grid_pos: Tuple[int, int], origin: np.ndarray) -> bool:
        """检查网格位置是否无碰撞"""
        world_pos = self.grid_to_world(grid_pos, origin)
        
        # 检查与障碍物的碰撞
        for obstacle in self.obstacles:
            if obstacle['type'] == 'circle':
                center = np.array(obstacle['center'])
                radius = obstacle['radius']
                distance = MathUtils.euclidean_distance(world_pos, center)
                if distance < radius + self.robot_radius:
                    return False
            elif obstacle['type'] == 'rectangle':
                # 简化的矩形碰撞检测
                center = np.array(obstacle['center'])
                size = obstacle['size']
                half_size = np.array(size) / 2.0
                
                # AABB碰撞检测
                if (abs(world_pos[0] - center[0]) < half_size[0] + self.robot_radius and
                    abs(world_pos[1] - center[1]) < half_size[1] + self.robot_radius):
                    return False
        
        return True
    
    def heuristic(self, node: Node, goal: Node) -> float:
        """启发式函数（欧几里得距离）"""
        dx = abs(node.x - goal.x)
        dy = abs(node.y - goal.y)
        return self.heuristic_weight * np.sqrt(dx * dx + dy * dy) * self.grid_resolution
    
    def get_neighbors(self, node: Node, grid_bounds: Tuple[int, int, int, int]) -> List[Node]:
        """获取邻居节点"""
        neighbors = []
        x_min, y_min, x_max, y_max = grid_bounds
        
        for dx, dy, cost in self.directions:
            new_x = node.x + dx
            new_y = node.y + dy
            
            # 检查边界
            if new_x < x_min or new_x >= x_max or new_y < y_min or new_y >= y_max:
                continue
            
            neighbor = Node(new_x, new_y)
            neighbor.g_cost = node.g_cost + cost * self.grid_resolution
            neighbors.append(neighbor)
        
        return neighbors
    
    def reconstruct_path(self, node: Node) -> List[Node]:
        """重构路径"""
        path = []
        current = node
        
        while current is not None:
            path.append(current)
            current = current.parent
        
        return list(reversed(path))
    
    def search(self, start_pos: np.ndarray, goal_pos: np.ndarray) -> List[np.ndarray]:
        """
        执行A*搜索
        
        Args:
            start_pos: 起点世界坐标 [x, y]
            goal_pos: 终点世界坐标 [x, y]
            
        Returns:
            路径点列表，每个点为世界坐标 [x, y]
        """
        print(f"A*搜索: 从 {start_pos} 到 {goal_pos}")
        
        # 计算搜索区域
        margin = 5.0  # 5米边界
        x_min = min(start_pos[0], goal_pos[0]) - margin
        x_max = max(start_pos[0], goal_pos[0]) + margin
        y_min = min(start_pos[1], goal_pos[1]) - margin
        y_max = max(start_pos[1], goal_pos[1]) + margin
        
        origin = np.array([x_min, y_min])
        
        # 网格大小
        grid_width = int((x_max - x_min) / self.grid_resolution) + 1
        grid_height = int((y_max - y_min) / self.grid_resolution) + 1
        grid_bounds = (0, 0, grid_width, grid_height)
        
        # 转换起点和终点
        start_grid = self.world_to_grid(start_pos, origin)
        goal_grid = self.world_to_grid(goal_pos, origin)
        
        # 检查起点和终点是否可行
        if not self.is_collision_free(start_grid, origin):
            print("错误: 起点位于障碍物内")
            return []
        
        if not self.is_collision_free(goal_grid, origin):
            print("错误: 终点位于障碍物内")
            return []
        
        # 初始化起点和终点节点
        start_node = Node(start_grid[0], start_grid[1])
        goal_node = Node(goal_grid[0], goal_grid[1])
        start_node.h_cost = self.heuristic(start_node, goal_node)
        start_node.f_cost = start_node.g_cost + start_node.h_cost
        
        # A*搜索
        open_list = [start_node]
        closed_set: Set[Tuple[int, int]] = set()
        open_dict = {(start_node.x, start_node.y): start_node}
        
        iterations = 0
        max_iterations = config.planning.max_iterations
        
        while open_list and iterations < max_iterations:
            iterations += 1
            
            # 选择f值最小的节点
            current_node = heapq.heappop(open_list)
            current_pos = (current_node.x, current_node.y)
            
            # 从open字典中移除
            if current_pos in open_dict:
                del open_dict[current_pos]
            
            # 添加到closed集合
            closed_set.add(current_pos)
            
            # 检查是否到达目标
            if current_node.x == goal_node.x and current_node.y == goal_node.y:
                print(f"A*搜索成功! 迭代次数: {iterations}")
                
                # 重构路径
                path_nodes = self.reconstruct_path(current_node)
                
                # 转换为世界坐标
                world_path = []
                for node in path_nodes:
                    world_pos = self.grid_to_world((node.x, node.y), origin)
                    world_path.append(world_pos)
                
                # 路径平滑处理
                smoothed_path = self.smooth_path(world_path, origin)
                
                print(f"路径长度: {len(smoothed_path)} 个点")
                return smoothed_path
            
            # 扩展邻居节点
            neighbors = self.get_neighbors(current_node, grid_bounds)
            
            for neighbor in neighbors:
                neighbor_pos = (neighbor.x, neighbor.y)
                
                # 跳过已访问的节点
                if neighbor_pos in closed_set:
                    continue
                
                # 检查碰撞
                if not self.is_collision_free(neighbor_pos, origin):
                    continue
                
                # 计算启发式代价
                neighbor.h_cost = self.heuristic(neighbor, goal_node)
                neighbor.f_cost = neighbor.g_cost + neighbor.h_cost
                neighbor.parent = current_node
                
                # 检查是否在open列表中
                if neighbor_pos in open_dict:
                    existing_node = open_dict[neighbor_pos]
                    if neighbor.g_cost < existing_node.g_cost:
                        # 找到更好的路径
                        existing_node.g_cost = neighbor.g_cost
                        existing_node.f_cost = neighbor.f_cost
                        existing_node.parent = current_node
                else:
                    # 添加新节点
                    heapq.heappush(open_list, neighbor)
                    open_dict[neighbor_pos] = neighbor
        
        print(f"A*搜索失败: 达到最大迭代次数 {max_iterations}")
        return []
    
    def smooth_path(self, path: List[np.ndarray], origin: np.ndarray) -> List[np.ndarray]:
        """路径平滑处理"""
        if len(path) <= 2:
            return path
        
        smoothed = [path[0]]
        i = 0
        
        while i < len(path) - 1:
            # 尝试直线连接到更远的点
            j = len(path) - 1
            while j > i + 1:
                if self.is_line_collision_free(path[i], path[j], origin):
                    smoothed.append(path[j])
                    i = j
                    break
                j -= 1
            else:
                # 如果无法跳跃，添加下一个点
                smoothed.append(path[i + 1])
                i += 1
        
        return smoothed
    
    def is_line_collision_free(self, start: np.ndarray, end: np.ndarray, origin: np.ndarray) -> bool:
        """检查直线路径是否无碰撞"""
        # 采样直线上的点进行碰撞检测
        distance = MathUtils.euclidean_distance(start, end)
        num_samples = int(distance / (self.grid_resolution * 0.5)) + 1
        
        for i in range(num_samples + 1):
            alpha = i / num_samples if num_samples > 0 else 0
            point = (1 - alpha) * start + alpha * end
            grid_pos = self.world_to_grid(point, origin)
            
            if not self.is_collision_free(grid_pos, origin):
                return False
        
        return True
    
    def visualize_search_result(self, path: List[np.ndarray]) -> dict:
        """可视化搜索结果（返回可视化数据）"""
        if not path:
            return {'path': [], 'length': 0, 'status': 'failed'}
        
        # 计算路径长度
        total_length = 0.0
        for i in range(1, len(path)):
            total_length += MathUtils.euclidean_distance(path[i-1], path[i])
        
        return {
            'path': path,
            'length': total_length,
            'num_points': len(path),
            'status': 'success'
        }