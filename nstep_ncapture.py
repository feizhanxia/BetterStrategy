import gymnasium as gym
import numpy as np
import pandas as pd
import sys
import environment
from agents.my_policies import GreedyPolicy, RandomPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
import gymnasium as gym
import yaml
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback


# 设置环境ID和配置ID
env_id = 'PredatorPrey-v0'
config_id = 'eval_0'
name_prefix = 'smart' # 'smart' or 'greedy'
episodes = 2000
n_envs = 10
save_path = 'analysis/data/nstep_ncap_{}_{}_{}.csv'.format(config_id, name_prefix, episodes)

# 加载模型
model_path = './output/checkpoints/default_8/put_back_v0_7680000_steps.zip'
if name_prefix == 'smart':
    model = PPO.load(model_path)
elif name_prefix == 'greedy':
    model = GreedyPolicy(env.observation_space, env.action_space)
elif name_prefix == 'random':
    model = RandomPolicy(env.observation_space, env.action_space)


# 读取配置文件
with open('./params/env_configs.yaml', 'r') as file:
    env_config = yaml.safe_load(file)[env_id][config_id]

if __name__ == '__main__':
    # 创建环境
    vec_env = make_vec_env(env_id, n_envs=n_envs, vec_env_cls=SubprocVecEnv, env_kwargs=env_config)
    
    # 创建回调函数
    n_capture = np.zeros(n_envs, dtype=int)
    n_steps = np.zeros(n_envs, dtype=int)
    start_episode = np.arange(0, n_envs)*episodes//n_envs+1
    data = []
    def my_callback(local_vars, global_vars):
        infos = local_vars['infos']
        dones = local_vars['dones']
        finished = local_vars['episode_starts']
        for i in range(n_envs):
            if infos[i]['capture_count']>n_capture[i] and not finished[i]:
                n_capture[i] = infos[i]['capture_count']
                n_steps[i] = infos[i]['num_steps']
                # 记录数据
                data.append({
                    'config_id': config_id,
                    'episode': start_episode[i]+local_vars['episode_counts'][i],
                    'n_capture': n_capture[i],
                    'n_steps': n_steps[i]
                })
            elif finished[i]:
                n_capture[i] = 0
                n_steps[i] = 0
                # 记录数据
                data.append({
                    'config_id': config_id,
                    'episode': start_episode[i]+local_vars['episode_counts'][i],
                    'n_capture': n_capture[i],
                    'n_steps': n_steps[i]
                })
        return True

    # 评估模型
    mean_reward, std_reward = evaluate_policy(model, 
                                                vec_env, 
                                                n_eval_episodes = episodes,
                                                callback=my_callback)
    
    # 创建 DataFrame 并保存为 CSV
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    print(f"Data saved to {save_path}")
    
    # 输出评估结果
    print(f"{name_prefix} mean_reward:{mean_reward:.2f} ep_reward_std:{std_reward:.2f}")
