import gymnasium as gym
from gymnasium import spaces
import numpy as np
from agents.prey import Prey
from utils.sampling import uniform_circle_sample
from utils.kinematics import bound_positions

class PredatorPreyEnv(gym.Env):
    """
    Custom environment for Predator-Prey game in a circular arena using gymnasium.
    """
    metadata = {'render_modes': ['console', 'rgb_array']}

    def __init__(self, 
                arena_radius=10.0, 
                visibility_radius=2.0, 
                critical_distance=0.5, 
                num_preys=100, 
                predator_speed=0.1,
                prey_max_speed=0.1,
                energy_per_step=-0.1, 
                energy_per_capture=1.0, 
                home_reward_ratio=0.5):
        super(PredatorPreyEnv, self).__init__()
        # Environment parameters
        self.home_position = np.array([0.,0.], dtype=np.float32)  # Center of the arena, home
        self.arena_radius = arena_radius  # Radius of the circular arena
        self.critical_distance = critical_distance  # Distance to consider prey captured
        self.visibility_radius = visibility_radius  # Visibility radius of the predator
        self.num_preys = num_preys  # Number of preys in the arena
        
        # Predator & Prey parameters
        self.predator_speed = predator_speed # Speed of the predator
        self.prey_max_speed = prey_max_speed  # Speed of the preys

        # Reward parameters
        self.energy_per_step = energy_per_step  # Energy cost of moving
        self.energy_per_capture = energy_per_capture  # Energy gained from capturing a prey
        self.home_reward_ratio = home_reward_ratio  # Reward ratio for returning home

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
                        max_speed=self.prey_max_speed,
                        radius=self.arena_radius,
                        center=self.home_position
                        ) 
        self.predator_position = self.home_position 
        self.predator_angle = np.random.uniform(-np.pi, np.pi)
        self.energy = 0.0
        

    def step(self, action):
        self.preys.move()
        self._move_predator(action)
        observation = self._get_observation()
        reward = self._calculate_reward()
        terminated = self._check_done()
        info = {}
        truncated = False
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # Seed not used in this environment
        self.preys = Prey(self._initialize_prey_positions(),
                        max_speed=self.prey_max_speed,
                        radius=self.arena_radius,
                        center=self.home_position
                        ) 
        self.predator_position = self.home_position 
        self.predator_angle = np.random.uniform(-np.pi, np.pi)
        info = {}
        return self._get_observation(), info

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError("Supported render mode is console")
        # Simple text representation of positions
        print(f"Predator: {self.predator_position}, Preys: {self.preys.positions}")

    def _move_predator(self, action):
        # Update predator position based on the action angle
        # 计算从家到当前捕食者位置的向量
        predator_polar = self._cartesian_to_polar(self.predator_position - self.home_position)
        # 计算家到捕食者连线与正北方向的夹角
        angle_to_home = predator_polar[1]
        # 计算新的移动方向
        action_angle = np.pi * action
        move_angle = angle_to_home + action_angle
        # 更新捕食者的位置
        self.predator_position += self.predator_speed * np.array([np.cos(move_angle), np.sin(move_angle)])
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
            avg_polar = self._cartesian_to_polar(avg_position - self.predator_position)
        else:
            avg_polar = self._imagine() # !
        # Closest prey position
        visible_distances = np.linalg.norm(self.predator_position - visible_preys, axis=1)
        closest_prey_idx = np.argmin(visible_distances)
        closest_prey = visible_preys[closest_prey_idx] if visible_preys.size > 0 else self._imagine() # !
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
        self.preys.positions = remaining_preys

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
        if current_prey_count == 0:
            return True
        return False

    def _initialize_prey_positions(self):
        # This method calls the sampling utility to generate initial positions
        return uniform_circle_sample(self.home_position, self.arena_radius, self.num_preys)
        
        

