import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
import multiprocessing
import environment

# 设置环境ID和配置ID
ENV_ID = 'PredatorPrey-v0'
CONFIG_ID = 'put_back_0'
NAME_PREFIX = 'smart'
MODEL_PATH = 'output/checkpoints/default_8/put_back_v1_7040000_steps.zip'
NUM_TRAJECTORIES = 5000
OUTPUT_FILENAME = f'analysis/data/trajectories_{CONFIG_ID}_{NAME_PREFIX}_{NUM_TRAJECTORIES}.csv'

# 读取配置文件
with open('./params/env_configs.yaml', 'r') as file:
    env_config = yaml.safe_load(file)[ENV_ID][CONFIG_ID]

# define a global list to store all trajectories df
all_df = []

# define a function to simulate a single trajectory
def simulate_trajectory(index, env, model):
    obs, info = env.reset()
    done = False
    trajectory = []
    while not done:
        action, _states = model.predict(obs , deterministic=True)
        obs, _reward, done, _truncated, info = env.step(action)
        point = {
            "time": info["num_steps"],
            "x": info["predator_position"][0],
            "y": info["predator_position"][1],
            "is_cap": info["is_cap"]
        }
        trajectory.append(point)
    df_trajectory = pd.DataFrame(trajectory, columns=["trajectory_id", "time", "x", "y", "is_cap"])
    df_trajectory["trajectory_id"] = index
    return df_trajectory


# define a function to simulate num_episodes trajectories
def simulate_trajectories(start_index, num_episodes, position=0):
    model = PPO.load(MODEL_PATH)
    env = gym.make(ENV_ID, **env_config)
    trajectories = []
    for i in tqdm(range(start_index, start_index + num_episodes), 
                    desc=f"Core {position}", 
                    position=position, 
                    ascii=True):
        _obs, info = env.reset()
        print(info["predator_position"][0], info["predator_position"][1])
        df_trajectory = simulate_trajectory(i, env, model)
        if not df_trajectory.empty:
            trajectories.append(df_trajectory)
    df_trajectories = pd.concat(trajectories, ignore_index=True)
    return df_trajectories


def handle_result(df):
    # 这个回调函数处理每个任务的结果
    global all_df
    if not df.empty:
        all_df.append(df)
    return True


def main():
    num_cores = multiprocessing.cpu_count()
    if num_cores > NUM_TRAJECTORIES:
        num_cores = NUM_TRAJECTORIES
    # 检查 NUM_TRAJECTORIES 是否是 num_cores 的倍数
    assert NUM_TRAJECTORIES % num_cores == 0, f"NUM_TRAJECTORIES must be a multiple of num_cores ({num_cores}), but it is {NUM_TRAJECTORIES}"
    # then we can split the task into num_cores parts
    num_episodes_per_core = NUM_TRAJECTORIES // num_cores
    start_indices = [i * num_episodes_per_core for i in range(num_cores)]
    with multiprocessing.Pool(num_cores) as pool:
        results = []
        # 异步提交任务
        for i in range(num_cores):
            result = pool.apply_async(simulate_trajectories, 
                                    args=(start_indices[i], num_episodes_per_core, i), 
                                    callback=handle_result)
            results.append(result)
        # 等待所有任务完成,并获取结果
        for result in results:
            result.get()  # 这里我们只是等待任务完成,实际结果处理在回调函数中进行
        # 将所有结果合并
        df_all = pd.concat(all_df, ignore_index=True)
        # 保存结果
        df_all.to_csv(OUTPUT_FILENAME, index=False)

if __name__ == '__main__':
    main()
