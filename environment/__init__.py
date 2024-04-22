from gymnasium.envs.registration import register
from environment.env import PredatorPreyEnv

register(
    id="PredatorPrey-v0",
    entry_point="environment.env:PredatorPreyEnv",
    
)