import gymnasium as gym
import numpy as np
import environment
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv


# def make_env(env_id):
#     def _init():
#         env = gym.make(env_id)
#         return env
#     return _init

# num_envs = 2  # 并行环境的数量
# env = SubprocVecEnv([gym.make('environment/PredatorPrey-v0')])
# env = DummyVecEnv([make_env('environment/PredatorPrey-v0') for _ in range(num_envs)])
if __name__ == '__main__':
    vec_env = make_vec_env('environment/PredatorPrey-v0', n_envs=8,vec_env_cls=SubprocVecEnv)

# env = gym.make('environment/PredatorPrey-v0')

# env = PredatorPreyEnv(num_preys=1)
# env.render()
# env.step(0.)
# env.render()

# from agents.prey import Prey
# prey = Prey([[0, 0], [12., 0]], 10.)
# print(prey.positions)
# prey.move()
# print(prey.positions)