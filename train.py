import environment
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import yaml


env_id = 'PredatorPrey-v0'
env_config_id = 'default'
train_config_id = 'default'
name_prefix = 'predator_prey'
# 读取配置文件
with open('./params/env_configs.yaml', 'r') as file:
    env_config = yaml.safe_load(file)[env_id][env_config_id]
with open('./params/train_configs.yaml', 'r') as file:
    train_config = yaml.safe_load(file)[train_config_id]

if __name__ == '__main__':
    # 创建环境
    vec_env = make_vec_env(env_id, n_envs=train_config['n_envs'], vec_env_cls=SubprocVecEnv, env_kwargs=env_config)
    
    # 创建并配置PPO算法
    model = PPO(train_config['policy'], vec_env, verbose=1, tensorboard_log="./output/ppo_tensorboard/", **train_config["model"])

    # Checkpoint每10000步保存一次
    checkpoint_callback = CheckpointCallback(save_freq=train_config['save_freq'], save_path='./output/checkpoints/',
                                            name_prefix=name_prefix)

    # 每10000步评估一次，并记录评估结果
    eval_callback = EvalCallback(vec_env, best_model_save_path='./output/best_model/',
                                log_path='./output/logs/', eval_freq=train_config['eval_freq'],
                                deterministic=True, render=False)

    # 训练模型
    model.learn(total_timesteps=train_config['total_timesteps'], callback=[checkpoint_callback, eval_callback])
    
    # 保存模型
    model.save("final_model")

 