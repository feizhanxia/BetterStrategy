from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.utils import get_device
import gymnasium as gym
import environment

env = gym.make('environment/PredatorPrey-v0')
check_env(env)
print(get_device())