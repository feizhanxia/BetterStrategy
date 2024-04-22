import gymnasium as gym
import numpy as np
import environment
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

if __name__ == '__main__':
    vec_env = make_vec_env('PredatorPrey-v0', n_envs=32, vec_env_cls=SubprocVecEnv)
    

    # 创建并配置PPO算法
    model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log="./output/ppo_tensorboard/")

    # Checkpoint每10000步保存一次
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./output/checkpoints/',
                                            name_prefix='predator_prey')

    # 每10000步评估一次，并记录评估结果
    eval_callback = EvalCallback(vec_env, best_model_save_path='./output/best_model/',
                                log_path='./logs/', eval_freq=10000,
                                deterministic=True, render=False)

    # 训练模型
    model.learn(total_timesteps=25000, callback=[checkpoint_callback, eval_callback])
    
    # 保存模型
    model.save("final_model")

