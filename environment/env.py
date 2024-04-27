import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from agents.prey import Prey
from utils.sampling import uniform_circle_sample
from utils.kinematics import bound_positions
import sys

class PredatorPreyEnv(gym.Env):
    """
    Custom environment for Predator-Prey game in a circular arena using gymnasium.
    """
    metadata = {'render_modes': ['console', 'human', 'rgb_array'], 'render_fps': 12, 'render_return': 'rgb_array'}

    def __init__(self, 
                arena_radius=10.0, 
                visibility_radius=2.0, 
                critical_distance=0.5, 
                num_preys=100, 
                predator_speed=0.1,
                predator_D_0=0.01,
                predator_D_theta=0.01,
                prey_D_0=0.05,
                energy_per_step=-0.1, 
                energy_per_capture=1.0, 
                target_capture_ratio=0.3,
                is_reward_home=False, 
                home_reward_ratio=0.5,
                render_mode='human'):
        super(PredatorPreyEnv, self).__init__()
        # Environment parameters
        self.home_position = np.array([0.,0.], dtype=np.float32)  # Center of the arena, home
        self.arena_radius = arena_radius  # Radius of the circular arena
        self.critical_distance = critical_distance  # Distance to consider prey captured
        self.visibility_radius = visibility_radius  # Visibility radius of the predator
        self.num_preys = num_preys  # Number of preys in the arena
        
        # Predator & Prey parameters
        self.predator_speed = predator_speed # Speed of the predator
        self.predator_D_0 = predator_D_0
        self.predator_D_theta = predator_D_theta
        self.prey_D_0 = prey_D_0  # Speed of the preys

        # Reward parameters
        self.energy_per_step = energy_per_step  # Energy cost of moving
        self.energy_per_capture = energy_per_capture  # Energy gained from capturing a prey
        
        # Give reward for returning home
        self.is_reward_home = is_reward_home
        self.home_reward_ratio = home_reward_ratio  # Reward ratio for returning home
        
        # Target capture ratio
        self.target_capture_ratio = target_capture_ratio

        # Random number generator
        self.rng = np.random.default_rng()

        # Action space (bearing angle in radians) 
        self.action_space = spaces.Box(low=-1, high=1, shape=(), dtype=np.float32)

        # Observation space (distance from home, number of visible preys, average position of visible preys, closest prey position)
        self.observation_space = spaces.Dict({
            "distance_to_home": spaces.Box(low=0, high=self.arena_radius, shape=(1,), dtype=np.float32),
            "average_position_of_visible_preys": spaces.Box(low=np.array([0, -np.pi]), high=np.array([self.arena_radius, np.pi]), dtype=np.float32),  # 2D position
            "closest_prey_position": spaces.Box(low=np.array([0, -np.pi]), high=np.array([self.arena_radius, np.pi]), dtype=np.float32)  # 2D position
        })

        # Initialize positions of preys and predator (reset)
        self.preys = Prey(self._initialize_prey_positions(),
                        D_0=self.prey_D_0,
                        radius=self.arena_radius,
                        center=self.home_position
                        ) 
        self.predator_position = self.home_position 
        self.predator_angle = self.rng.uniform(-np.pi, np.pi)
        self.energy = 0.0
        
        # Initialize renderer
        self.render_mode = render_mode
        self.screen = None
        self.clock = None

    def step(self, action):
        self.preys.move()
        self._move_predator(action)
        observation = self._get_observation()
        reward = self._calculate_reward()
        terminated = self._check_done()
        info = {'reward': reward}
        truncated = False
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # Seed not used in this environment
        self.preys = Prey(self._initialize_prey_positions(),
                        D_0=self.prey_D_0,
                        radius=self.arena_radius,
                        center=self.home_position
                        ) 
        self.predator_position = self.home_position 
        self.predator_angle = self.rng.uniform(-np.pi, np.pi)
        self.energy = 0.0
        info = {}
        return self._get_observation(), info

    def render(self):
        mode = self.render_mode
        if mode not in self.metadata['render_modes']:
            raise ValueError("Unsupported render mode.")
        elif mode == 'console':
            # Simple text representation of positions
            print(f"Predator: {self.predator_position}, Preys: {self.preys.positions}")
            return
        else:
            if self.screen is None or self.clock is None:
                self._initialize_renderer()
            self.screen.fill((255, 255, 255))  # 填充背景为白色
            # 绘制活动区域
            pygame.draw.circle(self.screen, (1, 158, 213), 
                            self._pos_to_int(self.home_position),
                            self.arena_radius*self.scale+6, 2)
            # 绘制家
            pygame.draw.circle(self.screen, (244, 222, 41), 
                            self._pos_to_int(self.home_position), 8)
            # 绘制捕食者
            pygame.draw.circle(self.screen, (244, 13, 100), 
                            self._pos_to_int(self.predator_position), 
                            self.critical_distance*self.scale)
            pygame.draw.circle(self.screen, (175, 18, 88), 
                            self._pos_to_int(self.predator_position), 
                            self.visibility_radius*self.scale, 1)
            # 绘制猎物
            for prey in self.preys.positions:
                pygame.draw.circle(self.screen, (179, 197, 135), 
                                self._pos_to_int(prey), 5)
        if mode == 'human':
            pygame.display.flip()  # 更新整个待显示的 Surface 对象到屏幕上
            self.clock.tick(12)  # 限制帧率为60fps
        elif mode == 'rgb_array':
            # 创建基于当前屏幕的rgb数组
            buffer = pygame.surfarray.array3d(self.screen)
            # 转置数组以符合常见的图像格式
            return np.transpose(np.array(buffer), axes=(1, 0, 2))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
    
    def _initialize_renderer(self):
        self.screen_size = 1024
        self.offset = np.array([self.screen_size // 2, self.screen_size // 2])  # 偏移量
        self.scale = (self.screen_size-30) / (2 * self.arena_radius)  # 缩放比例
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
        self.clock = pygame.time.Clock()
        
    def _move_predator(self, action):
        # Update predator position based on the action angle
        # 计算从家到当前捕食者位置的向量
        predator_polar = self._cartesian_to_polar(self.predator_position - self.home_position)
        # 计算家到捕食者连线与正北方向的夹角
        angle_to_home = predator_polar[1]
        # 计算新的移动方向
        action_angle = np.pi * action
        move_angle = angle_to_home + action_angle + self.rng.normal(0, np.sqrt(2 * self.predator_D_theta))
        # 更新捕食者的位置
        self.predator_position += np.array([np.cos(move_angle), np.sin(move_angle)]) * (self.predator_speed 
                                                                                        + self.rng.normal(0, np.sqrt(2 * self.predator_D_0)))
        # 确保捕食者不会离开定义的活动区域
        self.predator_position = bound_positions(np.array([self.predator_position]), self.home_position, self.arena_radius)[0]

    def _get_observation(self):
        # Calculate observation based on current state
        # Calculate distances to preys
        distances = np.linalg.norm(self.predator_position - self.preys.positions, axis=1)
        # Preys within visibility radius
        visible_preys = self.preys.positions[distances < self.visibility_radius]  
        # Number of visible preys
        num_visible_preys = len(visible_preys)  
        # Calculate average position of visible preys and closest prey position
        if num_visible_preys > 0:
            avg_position = np.mean(visible_preys, axis=0)
            visible_distances = np.linalg.norm(self.predator_position - visible_preys, axis=1)
            closest_prey_idx = np.argmin(visible_distances)
            closest_prey = visible_preys[closest_prey_idx]
        else:
            avg_position = self._imagine()
            closest_prey = self._imagine()
        # Polar of average position and closest prey position  of visible preys
        avg_polar = self._cartesian_to_polar(avg_position - self.predator_position)
        closest_polar = self._cartesian_to_polar(closest_prey - self.predator_position)
        # Calculate relative position, distance to home
        predator_polar = self._cartesian_to_polar(self.predator_position - self.home_position)
        distance_to_home = predator_polar[0]
        angle_to_home = predator_polar[1]
        avg_polar[1] -= angle_to_home
        closest_polar[1] -= angle_to_home
        # Return observation
        return {
            "distance_to_home": np.array([distance_to_home], dtype=np.float32),
            "average_position_of_visible_preys": avg_polar,
            "closest_prey_position": closest_polar
        }

    def _cartesian_to_polar(self, cartesian_coords):
        rho = np.linalg.norm(cartesian_coords)
        phi = np.arctan2(cartesian_coords[1], cartesian_coords[0])
        return np.array([rho, phi], dtype=np.float32)
    
    def _imagine(self):
        return uniform_circle_sample(self.predator_position, self.visibility_radius, 1)[0]


    def _check_capture(self):
        # Check and capture preys that are close enough to the predator
        remaining_preys = []
        for prey_position in self.preys.positions:
            if np.linalg.norm(self.predator_position - prey_position) >= self.critical_distance:
                remaining_preys.append(prey_position)
        self.preys.positions = np.array(remaining_preys, dtype=np.float32)

    def _calculate_reward(self):
        # Reward for moving
        reward = self.energy_per_step
        # Number of captured preys
        num_preys_before = len(self.preys.positions)
        self._check_capture()
        num_preys_after = len(self.preys.positions)
        num_captured = num_preys_before - num_preys_after  
        # Reward for capturing preys
        reward += num_captured * self.energy_per_capture  
        # Update energy
        self.energy += reward
        # Reward for returning home
        if self.is_reward_home:
            is_at_home = np.linalg.norm(self.predator_position - self.home_position) < self.critical_distance
            if is_at_home:
                reward += self.energy * self.home_reward_ratio
        return reward

    def _check_done(self):
        # Done if all preys are captured or predator is at home
        current_prey_count = self.preys.get_count()
        # is_at_home = np.linalg.norm(self.predator_position - self.home_position) < self.critical_distance
        # if current_prey_count == 0 or is_at_home:
        #     return True
        if current_prey_count <= int(self.num_preys * (1-self.target_capture_ratio)):
            return True
        return False

    def _initialize_prey_positions(self):
        # This method calls the sampling utility to generate initial positions
        return uniform_circle_sample(self.home_position, self.arena_radius, self.num_preys)
    
    def _pos_to_int(self, position):
        # 应用缩放
        scaled_position = position * self.scale + self.offset
        # 应用偏移并取整
        final_position = (int(scaled_position[0]), int(scaled_position[1]))
        return final_position

        
        

