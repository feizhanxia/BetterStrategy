import numpy as np
import environment
from agents.my_policies import GreedyPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import yaml


# 设置环境ID和配置ID
env_id = 'PredatorPrey-v0'
config_id = 'default_7'

# 读取配置文件
with open('./params/env_configs.yaml', 'r') as file:
    env_config = yaml.safe_load(file)[env_id][config_id]


if __name__ == '__main__':
    
    # 创建环境
    vec_env = make_vec_env(env_id, n_envs=10, vec_env_cls=SubprocVecEnv, env_kwargs=env_config)
    
    # 加载模型
    model = GreedyPolicy(vec_env.observation_space, vec_env.action_space)
    
    # 评估模型
    mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=1000)
    
    # 输出评估结果
    print(f"greedy model: mean_reward:{mean_reward:.2f} ep_reward_std:{std_reward:.2f}")
