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
config_id = 'random_putback_count_2'
name_prefix = 'random' # 'smart' or 'greedy' or 'random'
episodes = 500
n_envs = 10
save_path = 'analysis/data/ave_steps_{}_{}_{}.csv'.format(config_id, name_prefix, episodes)
model_path = './output/best_model/default_12_v0_7680000_steps.zip'
num_preys_list = list(range(10, 125, 5))


# 读取配置文件
with open('./params/env_configs.yaml', 'r') as file:
    env_config = yaml.safe_load(file)[env_id][config_id]

if __name__ == '__main__':
    # 加载模型
    if name_prefix == 'smart':
        model = PPO.load(model_path)
    elif name_prefix == 'greedy':
        vec_env = make_vec_env(env_id, n_envs=n_envs, vec_env_cls=SubprocVecEnv, env_kwargs=env_config)
        model = GreedyPolicy(vec_env.observation_space, vec_env.action_space)
    elif name_prefix == 'random':
        vec_env = make_vec_env(env_id, n_envs=n_envs, vec_env_cls=SubprocVecEnv, env_kwargs=env_config)
        model = RandomPolicy(vec_env.observation_space, vec_env.action_space)
    elif name_prefix == 'smarter':
        vec_env = make_vec_env(env_id, n_envs=n_envs, vec_env_cls=SubprocVecEnv, env_kwargs=env_config)
        model = SmarterPolicy(vec_env.observation_space, vec_env.action_space)
    
    data = []
    for num_preys in num_preys_list:
        env_config['num_preys'] = num_preys
        print(f"num_preys: {num_preys}")
        # 创建环境
        vec_env = make_vec_env(env_id, n_envs=n_envs, vec_env_cls=SubprocVecEnv, env_kwargs=env_config)
        # 初始化tqdm进度条
        progress_bar = tqdm(total=episodes, desc="Evaluating", ascii=True, unit_scale=True)
        # 创建回调函数
        current_episode = 0
        def pb_callback(local_vars, global_vars):
            global current_episode
            infos = local_vars['infos']
            dones = local_vars['dones']
            finished = local_vars['episode_starts']
            if sum(local_vars['episode_counts'])>current_episode:
                current_episode = sum(local_vars['episode_counts'])
                progress_bar.update()
            return True

        # 评估
        ep_rewards, ep_lengths = evaluate_policy(model, 
                                                vec_env, 
                                                n_eval_episodes=episodes, 
                                                callback=pb_callback, 
                                                return_episode_rewards=True)
        for i in range(episodes):
            data.append({'num_preys': num_preys,
                            'episode': i,
                            'ep_rewards': ep_rewards[i],
                            'ep_lengths': ep_lengths[i],
                            'mean_steps_per_capture': ep_lengths[i]/ep_rewards[i]})
        mean_steps_per_capture = np.mean([np.array(ep_lengths)/np.array(ep_rewards)])
        print(f"Mean steps per capture: {mean_steps_per_capture}")
        # 关闭进度条
        progress_bar.close()
    
    # 创建 DataFrame 并保存为 CSV
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    print(f"Data saved to {save_path}")
