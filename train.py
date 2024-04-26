import environment
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import yaml
import gymnasium as gym


env_id = 'PredatorPrey-v0'
config_id = 'default_3'
name_prefix = 'predator_prey'
# 读取配置文件
with open('./params/env_configs.yaml', 'r') as file:
    env_config = yaml.safe_load(file)[env_id][config_id]
with open('./params/train_configs.yaml', 'r') as file:
    train_config = yaml.safe_load(file)[config_id]

if __name__ == '__main__':

    # 创建环境
    vec_env = make_vec_env(env_id, n_envs=train_config['n_envs'], vec_env_cls=SubprocVecEnv, env_kwargs=env_config)

    # 创建并配置PPO算法，训练新模型
    model = PPO('MultiInputPolicy', 
                env=vec_env, 
                verbose=3, 
                tensorboard_log="./output/ppo_tensorboard/",
                **train_config["model"])
    
    # 加载训练好的权重，迁移学习
    model = PPO.load("output/checkpoints/default_2/predator_prey_final_model.zip", 
                    env=vec_env, 
                    verbose=3,
                    tensorboard_log="./output/ppo_tensorboard/",
                    **train_config["model"])

    # Checkpoint每n步保存一次
    checkpoint_callback = CheckpointCallback(save_freq=train_config['save_freq'], 
                                            save_path='output/checkpoints/'+config_id+'/',
                                            name_prefix=name_prefix,
                                            save_replay_buffer=True,
                                            save_vecnormalize=True,)

    # 训练模型
    model.learn(total_timesteps=train_config['total_timesteps'], 
                progress_bar=True, 
                callback=[checkpoint_callback])
    
    # 保存模型
    model.save("output/checkpoints/"+config_id+"/"+name_prefix+"_final_model")

