#!/usr/bin/env python3
"""
机器人控制修复脚本
修复SVSDF演示中机器人不移动的问题
确保机器人通过底盘物理驱动而不是瞬移
"""

import numpy as np
import math

class RobotControlFix:
    """机器人控制修复类"""
    
    @staticmethod
    def fix_interactive_loop(demo_instance):
        """修复主循环，添加机器人控制更新"""
        
        def enhanced_interactive_loop(self):
            """增强的交互式主循环 - 包含机器人控制更新"""
            self.running = True
            print("\n🎮 交互模式开始！使用箭头键移动目标，SPACE开始导航，ESC退出")
            print("🔧 已应用机器人控制修复 - 确保物理移动")
            
            try:
                while self.running:
                    # 更新仿真
                    self.world.step(render=True)
                    
                    # 🔧 关键修复：更新机器人控制 - 确保机器人物理移动
                    self.update_robot_control()
                    
                    # 检查是否需要重新规划
                    if self.auto_navigation and self.goal_changed:
                        self.goal_changed = False
                        self.request_replan()
                    
                    # 控制频率：50Hz更新
                    import time
                    time.sleep(0.02)  # 50Hz
                    
            except KeyboardInterrupt:
                print("\n用户中断")
            finally:
                print("退出交互模式")
        
        # 替换原方法
        demo_instance.interactive_loop = enhanced_interactive_loop.__get__(demo_instance, demo_instance.__class__)
        return demo_instance
    
    @staticmethod 
    def fix_robot_control(demo_instance):
        """修复机器人控制方法，确保底盘移动"""
        
        def enhanced_apply_robot_control(self, linear_vel: float, angular_vel: float):
            """增强的机器人控制 - 确保底盘移动而不是机械臂"""
            if not hasattr(self, 'robot_articulation') or self.robot_articulation is None:
                print("⚠️ 机器人articulation未初始化")
                return
            
            try:
                # 方法1：使用差分控制器
                command = np.array([linear_vel, angular_vel])
                articulation_action = self.controller.forward(command)
                
                # 应用控制
                self.robot_articulation.apply_action(articulation_action)
                
                # 调试输出
                if abs(linear_vel) > 0.01 or abs(angular_vel) > 0.01:
                    print(f"🚗 应用控制: 线速度={linear_vel:.3f}, 角速度={angular_vel:.3f}")
                
                # 更新状态
                self.current_linear_vel = linear_vel
                self.current_angular_vel = angular_vel
                
            except Exception as e:
                print(f"❌ 标准控制失败: {e}")
                
                # 方法2：直接关节控制 (备用方案)
                try:
                    self._apply_direct_joint_control(linear_vel, angular_vel)
                except Exception as e2:
                    print(f"❌ 直接关节控制也失败: {e2}")
        
        def _apply_direct_joint_control(self, linear_vel: float, angular_vel: float):
            """直接关节控制方法 - 备用方案"""
            try:
                # 获取关节信息
                dof_names = self.robot_articulation.dof_names
                joint_indices = self.robot_articulation.get_applied_action_space()
                
                print(f"🔧 使用直接关节控制，关节: {dof_names}")
                
                # Create-3机器人参数
                wheel_base = 0.235  # 轮距
                wheel_radius = 0.0508  # 轮子半径
                
                # 差分驱动运动学
                left_wheel_vel = (linear_vel - angular_vel * wheel_base / 2) / wheel_radius
                right_wheel_vel = (linear_vel + angular_vel * wheel_base / 2) / wheel_radius
                
                # 创建速度命令
                velocities = np.zeros(len(dof_names))
                
                # 查找并设置轮子速度
                for i, name in enumerate(dof_names):
                    name_lower = name.lower()
                    if 'left' in name_lower and 'wheel' in name_lower:
                        velocities[i] = left_wheel_vel
                        print(f"  左轮 {name}: {left_wheel_vel:.3f} rad/s")
                    elif 'right' in name_lower and 'wheel' in name_lower:
                        velocities[i] = right_wheel_vel
                        print(f"  右轮 {name}: {right_wheel_vel:.3f} rad/s")
                
                # 应用速度
                self.robot_articulation.set_joint_velocities(velocities)
                
            except Exception as e:
                print(f"直接关节控制错误: {e}")
        
        # 替换原方法
        demo_instance.apply_robot_control = enhanced_apply_robot_control.__get__(demo_instance, demo_instance.__class__)
        demo_instance._apply_direct_joint_control = _apply_direct_joint_control.__get__(demo_instance, demo_instance.__class__)
        return demo_instance
    
    @staticmethod
    def fix_trajectory_execution(demo_instance):
        """修复轨迹执行，确保连续物理移动"""
        
        def enhanced_execute_trajectory(self):
            """增强的轨迹执行"""
            if not self.current_trajectory:
                print("没有可执行的轨迹")
                return False
            
            print("🚀 开始执行物理轨迹跟踪...")
            print(f"📍 轨迹包含 {len(self.current_trajectory)} 个路径点")
            
            self.trajectory_executing = True
            self.trajectory_index = 0
            
            # 立即开始第一个控制周期
            self.update_robot_control()
            
            return True
        
        def enhanced_update_robot_control(self):
            """增强的机器人控制更新 - 确保平滑连续的移动"""
            if not self.trajectory_executing or not self.current_trajectory:
                # 确保机器人完全停止
                self.apply_robot_control(0.0, 0.0)
                return True
            
            if self.trajectory_index >= len(self.current_trajectory):
                print("✅ 轨迹执行完成")
                self.trajectory_executing = False
                self.apply_robot_control(0.0, 0.0)
                return True
            
            # 获取当前机器人位置
            current_pos, current_yaw = self.get_robot_pose()
            
            # 获取目标轨迹点
            target_point = self.current_trajectory[self.trajectory_index]
            target_x = target_point.position[0]
            target_y = target_point.position[1]
            
            # 计算距离和角度
            dx = target_x - current_pos[0]
            dy = target_y - current_pos[1]
            distance = math.sqrt(dx**2 + dy**2)
            target_angle = math.atan2(dy, dx)
            angle_error = target_angle - current_yaw
            
            # 归一化角度
            while angle_error > math.pi:
                angle_error -= 2 * math.pi
            while angle_error < -math.pi:
                angle_error += 2 * math.pi
            
            # 控制逻辑 - 优化的PID控制
            tolerance = 0.15  # 位置容差
            
            if distance > tolerance:
                # 还未到达目标点，继续移动
                
                # 动态调整控制增益
                kp_linear = 1.0 if distance > 1.0 else 0.6
                kp_angular = 2.0 if abs(angle_error) > 0.5 else 1.2
                
                # 计算控制命令
                if abs(angle_error) > 0.3:  # 需要大幅转向
                    angular_vel = np.clip(kp_angular * angle_error, -1.2, 1.2)
                    linear_vel = 0.15  # 转向时保持小速度
                else:
                    # 可以前进
                    linear_vel = min(kp_linear * distance, 0.5)
                    angular_vel = np.clip(kp_angular * angle_error, -0.8, 0.8)
                
                # 应用控制
                self.apply_robot_control(linear_vel, angular_vel)
                
            else:
                # 到达当前目标点
                self.trajectory_index += 1
                progress = (self.trajectory_index / len(self.current_trajectory)) * 100
                print(f"📍 到达轨迹点 {self.trajectory_index-1}/{len(self.current_trajectory)}, 进度: {progress:.1f}%")
                
                # 如果还有下一个点，立即开始移动
                if self.trajectory_index < len(self.current_trajectory):
                    # 预计算下一个目标的控制，避免停顿
                    next_target = self.current_trajectory[self.trajectory_index]
                    next_dx = next_target.position[0] - current_pos[0]
                    next_dy = next_target.position[1] - current_pos[1]
                    next_distance = math.sqrt(next_dx**2 + next_dy**2)
                    
                    if next_distance > 0.1:
                        next_angle = math.atan2(next_dy, next_dx)
                        next_angle_error = next_angle - current_yaw
                        
                        # 归一化
                        while next_angle_error > math.pi:
                            next_angle_error -= 2 * math.pi
                        while next_angle_error < -math.pi:
                            next_angle_error += 2 * math.pi
                        
                        # 开始移动向下一个点
                        linear_vel = min(0.8 * next_distance, 0.4)
                        angular_vel = np.clip(1.5 * next_angle_error, -1.0, 1.0)
                        self.apply_robot_control(linear_vel, angular_vel)
            
            return True
        
        # 替换原方法
        demo_instance.execute_trajectory = enhanced_execute_trajectory.__get__(demo_instance, demo_instance.__class__)
        demo_instance.update_robot_control = enhanced_update_robot_control.__get__(demo_instance, demo_instance.__class__)
        return demo_instance
    
    @staticmethod
    def apply_all_fixes(demo_instance):
        """应用所有修复"""
        print("🔧 正在应用机器人控制修复...")
        
        demo_instance = RobotControlFix.fix_interactive_loop(demo_instance)
        demo_instance = RobotControlFix.fix_robot_control(demo_instance)  
        demo_instance = RobotControlFix.fix_trajectory_execution(demo_instance)
        
        print("✅ 机器人控制修复完成")
        print("   - 主循环已更新，包含机器人控制")
        print("   - 控制方法已增强，确保底盘移动")
        print("   - 轨迹执行已优化，确保连续移动")
        
        return demo_instance

# 使用方法：
# from robot_control_fix import RobotControlFix
# demo = SVSDFDemo()
# demo = RobotControlFix.apply_all_fixes(demo)
