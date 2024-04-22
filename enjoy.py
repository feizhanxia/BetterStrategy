import gymnasium as gym
import numpy as np
import environment
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import PPO
import gymnasium as gym

gym.make("PredatorPrey-v0")
# 加载模型
model = PPO.load("final_model", env=env)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render()
