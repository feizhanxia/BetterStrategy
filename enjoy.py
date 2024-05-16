import gymnasium as gym
import numpy as np
import environment
from stable_baselines3 import PPO
import yaml
import cv2

# 设置环境ID和配置ID
env_id = 'PredatorPrey-v0'
config_id = 'default_14'
name_prefix = 'predator_prey'
model_path = 'output/checkpoints/default_8/put_back_v1_7040000_steps.zip'

# 'record' or 'watch' mode
mode = 'watch' 


# 读取配置文件
with open('./params/env_configs.yaml', 'r') as file:
    env_config = yaml.safe_load(file)[env_id][config_id]
# 设置特殊参数
# env_config['num_preys'] = 1
# env_config['predator_speed'] = 1.5
# env_config['predator_D_0'] = 0.0
# env_config['predator_D_theta'] = 0.0
# env_config['prey_D_0'] = 0.0

# 创建Gym环境
if mode == 'record':
    # 视频保存路径和格式设置,并加载Gym环境
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore # 使用mp4v编码器
    save_path = 'output/videos/eval_{}_{}.mp4'.format(name_prefix, config_id)
    out = cv2.VideoWriter(save_path, fourcc, 12, (1024, 1024))
    env = gym.make(env_id, **env_config, render_mode='rgb_array')
elif mode == 'watch':
    # 加载Gym环境
    env = gym.make(env_id, **env_config, render_mode='human')

# 加载模型
model = PPO.load(model_path)

# 打印神经网络结构
# print('神经网络结构:')
# print(model.policy)

# 初始化环境并运行
obs, info = env.reset()
done = False
while not done:
    if mode == 'record':
        # 保存视频,'rgb_array'模式
        frame = env.render()  # 获取当前环境的图像帧
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # type: ignore # 将RGB转换为BGR,适用于OpenCV
        out.write(frame)  # 写入帧到视频文件中
    elif mode == 'watch':
        # 渲染环境的当前状态,'human'模式
        env.render()
    action, _states = model.predict(obs , deterministic=True)
    obs, reward, done, _truncated, info = env.step(action)

# record track
track = info['track']

if mode == 'record':
    print("视频保存完毕!")

