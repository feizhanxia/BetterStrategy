import gymnasium as gym
import time
from stable_baselines3 import PPO
import yaml
import environment


env_id = 'PredatorPrey-v0'
config_id = 'default_5'
# name_prefix = 'predator_prey'

# 读取配置文件
with open('./params/env_configs.yaml', 'r') as file:
    env_config = yaml.safe_load(file)[env_id][config_id]

# 创建或加载你的Gym环境
env = gym.make(env_id, **env_config, render_mode='human')

# 训练模型
t1 = time.time()
model = PPO('MultiInputPolicy', env, verbose=0, device="cuda")
model.learn(total_timesteps=10_000)
print(f"Time train with cuda : {time.time()-t1:.2f}s")

t1 = time.time()
model = PPO('MultiInputPolicy', env, verbose=0, device="cpu")
model.learn(total_timesteps=10_000)
print(f"Time train with cpu : {time.time()-t1:.2f}s")

# 评估模型