import environment
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.logger import HParam
from stable_baselines3.common.logger import Image
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

# 自定义Callback以实现实时渲染训练过程和保存超参数
class MyCallback(BaseCallback):
    """
    Saves the hyperparameters at the start of the training, and logs them to TensorBoard.
    Renders one environment at each step.
    """

    def _on_training_start(self) -> None:
        hparam_dict = {
            'config_id': config_id,
            'name_prefix': name_prefix,
            'n_envs': train_config['n_envs'],
            'total_timesteps': train_config['total_timesteps'],
            'algorithm': self.model.__class__.__name__,
            'policy': train_config['policy'],
            'learning_rate': self.model.learning_rate,
            'n_steps': self.model.n_steps,
            'gamma': self.model.gamma,
            'env_id': env_id,
        }
        metric_dict = {
            "rollout/ep_len_mean": 0.0,
            "rollout/ep_rew_mean": 0.0,
            "train/value_loss": 0.0,
        }
        hparam_dict.update(env_config)
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        # self.training_env.render(mode='human')
        # image = self.training_env.render(mode="rgb_array")
        # "HWC" specify the dataformat of the image, here channel last
        # (H for height, W for width, C for channel)
        # See https://pytorch.org/docs/stable/tensorboard.html
        # for supported formats
        # self.logger.record("output/trajectory/image", Image(image, "HWC"), exclude=("stdout", "log", "json", "csv"))
        return True

if __name__ == '__main__':
    
    # 创建环境
    vec_env = make_vec_env(env_id, n_envs=train_config['n_envs'], vec_env_cls=SubprocVecEnv, env_kwargs=env_config)

    # 加载训练好的权重
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

