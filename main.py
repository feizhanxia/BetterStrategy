import gymnasium as gym
import numpy as np
import environment
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv


if __name__ == '__main__':
    vec_env = make_vec_env('environment/PredatorPrey-v0', n_envs=6,vec_env_cls=SubprocVecEnv)
