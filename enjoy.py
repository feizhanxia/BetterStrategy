import gymnasium as gym
import numpy as np
import environment
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
import gymnasium as gym
import yaml
import cv2

# 设置环境ID和配置ID
env_id = 'PredatorPrey-v0'
config_id = 'default_6'
# name_prefix = 'predator_prey'

# 读取配置文件
with open('./params/env_configs.yaml', 'r') as file:
    env_config = yaml.safe_load(file)[env_id][config_id]
    
# 视频保存路径和格式设置，并加载Gym环境
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用mp4v编码器
out = cv2.VideoWriter('output/videos/evaluation_video.mp4', fourcc, 12, (1024, 1024))
env = gym.make(env_id, **env_config, render_mode='rgb_array')

# 加载Gym环境
# env = gym.make(env_id, **env_config, render_mode='human')

# 加载模型
model = PPO.load("output/checkpoints/default_5/predator_prey_final_model.zip")

# 打印神经网络结构
print('神经网络结构：')
print(model.policy)

# 初始化环境并运行
obs, _info = env.reset()
done = False
while not done:
    # 保存视频，'rgb_array'模式
    frame = env.render()  # 获取当前环境的图像帧
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # 将RGB转换为BGR，适用于OpenCV
    out.write(frame)  # 写入帧到视频文件中
    # 渲染环境的当前状态，'human'模式
    # env.render()
    # 预测动作
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, _truncated, info = env.step(action)

print("视频保存完毕！")

