import gymnasium as gym
import numpy as np
import environment
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import PPO
import gymnasium as gym
import yaml

env_id = 'PredatorPrey-v0'
config_id = 'default_3'
# name_prefix = 'predator_prey'
# 读取配置文件
with open('./params/env_configs.yaml', 'r') as file:
    env_config = yaml.safe_load(file)[env_id][config_id]
# 创建或加载你的Gym环境
env = gym.make(env_id, **env_config)
# 加载模型
model = PPO.load("output/checkpoints/default_2/predator_prey_final_model.zip")

# 初始化环境
obs, _info = env.reset()
done = False

# 打印神经网络结构
print('神经网络结构：')
print(model.policy)

# 创建一个图形窗口以显示环境
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, _truncated, info = env.step(action)
    # 渲染环境的当前状态
    env.render()  # 确保你的环境支持rgb_array模式

