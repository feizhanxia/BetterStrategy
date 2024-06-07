import numpy as np
import environment
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
import yaml
import cv2


# 设置环境ID和配置ID
env_id = 'PredatorPrey-v0'
config_id = 'put_back_2' # 'eval_1'
name_prefix = 'predator_prey'
# model_path = 'output/checkpoints/default_9/put_back_home_v1_7040000_steps.zip'
models = [   
    "output/checkpoints/default_12/v0_final_model.zip",
    "output/checkpoints/default_12/v0_7680000_steps.zip",
    "output/checkpoints/default_12/v0_7040000_steps.zip",
    "output/checkpoints/default_12/v0_6400000_steps.zip",
]
# 读取配置文件
with open('./params/env_configs.yaml', 'r') as file:
    env_config = yaml.safe_load(file)[env_id][config_id]


if __name__ == '__main__':
    
    # 创建环境
    vec_env = make_vec_env(env_id, n_envs=10, vec_env_cls=SubprocVecEnv, env_kwargs=env_config)
    
    for path in models:
        # 加载模型
        model = PPO.load(path)
        
        # 评估模型
        mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=1000)
        
        # 输出评估结果
        print(f"model:{path} mean_reward:{mean_reward:.2f} ep_reward_std:{std_reward:.2f}")
