from stable_baselines3.common.policies import BasePolicy
from gymnasium import spaces
import torch as th
import numpy as np


class GreedyPolicy(BasePolicy):
    def __init__(self, 
                observation_space: spaces.Space,
                action_space: spaces.Box):
        super().__init__(observation_space, action_space)
        # self.model = model
    
    # def _predict(self, observation, state=None, episode_start=None, deterministic=True):
    #     # 获取数据
    #     closest_prey_position = observation['closest_prey_position'][0]  # 最近猎物的位置
    #     average_pos_of_preys = observation['average_position_of_visible_preys'][0]   # 可见猎物的平均位置
    #     if (closest_prey_position[0] == 0.0) and (average_pos_of_preys[0] == 0.0):
    #         # action = np.random.randint(0, 5)
    #         action = th.tensor(-0.5)
    #     else:
    #         action = closest_prey_position[1] / np.pi  # 向最近的猎物移动
    #     return action
    # import torch as th

    def _predict(self, observation, state=None, episode_start=None, deterministic=True):
        # 直接从字典中取出张量
        closest_prey_position = observation['closest_prey_position']
        average_pos_of_preys = observation['average_position_of_visible_preys']

        # 检查位置的第0元素是否为零
        zero_condition = (closest_prey_position[:, 0] == 0.0) & (average_pos_of_preys[:, 0] == 0.0)
        
        # 初始化动作张量为-0.5,确保与输入同形状
        action = th.full(closest_prey_position[:, 0].shape, -0.5, device=closest_prey_position.device)
        
        # 计算非零位置的动作
        action[~zero_condition] = closest_prey_position[~zero_condition, 1] / th.tensor(np.pi, device=closest_prey_position.device)
        
        return action
