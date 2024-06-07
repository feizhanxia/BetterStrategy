import gymnasium as gym
import numpy as np
import pandas as pd
import sys
import environment
from agents.my_policies import GreedyPolicy, RandomPolicy, SmarterPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
import gymnasium as gym
import yaml
from tqdm import tqdm
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback


# 设置环境ID和配置ID
env_id = 'PredatorPrey-v0'
config_id = 'put_back_count_2' 
name_prefix = 'greedy' # 'smart' or 'greedy' or 'random'
episodes = 1000
n_envs = 10
save_path = 'analysis/data/nstep_ncap_{}_{}_{}.csv'.format(config_id, name_prefix, episodes)
model_path = './output/best_model/default_12_v0_7680000_steps.zip'



# 读取配置文件
with open('./params/env_configs.yaml', 'r') as file:
    env_config = yaml.safe_load(file)[env_id][config_id]

if __name__ == '__main__':
    # 创建环境
    vec_env = make_vec_env(env_id, n_envs=n_envs, vec_env_cls=SubprocVecEnv, env_kwargs=env_config)
    
    # 加载模型
    if name_prefix == 'smart':
        model = PPO.load(model_path)
    elif name_prefix == 'greedy':
        model = GreedyPolicy(vec_env.observation_space, vec_env.action_space)
    elif name_prefix == 'random':
        model = RandomPolicy(vec_env.observation_space, vec_env.action_space)
    elif name_prefix == 'smarter':
        model = SmarterPolicy(vec_env.observation_space, vec_env.action_space)
    
    # 初始化tqdm进度条
    progress_bar = tqdm(total=episodes, desc="Evaluating", ascii=True, unit_scale=True)
    # 创建回调函数
    n_capture = np.zeros(n_envs, dtype=int)
    n_steps = np.zeros(n_envs, dtype=int)
    start_episode = np.arange(0, n_envs)*episodes//n_envs+1
    data = []
    current_episode = 0
    def my_callback(local_vars, global_vars):
        global current_episode
        infos = local_vars['infos']
        dones = local_vars['dones']
        finished = local_vars['episode_starts']
        if sum(local_vars['episode_counts'])>current_episode:
            current_episode = sum(local_vars['episode_counts'])
            progress_bar.update()
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
    
    # 关闭进度条
    progress_bar.close()
    
    # 创建 DataFrame 并保存为 CSV
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    print(f"Data saved to {save_path}")
    
    # 输出评估结果
    print(f"{name_prefix} mean_reward:{mean_reward:.2f} ep_reward_std:{std_reward:.2f}")
