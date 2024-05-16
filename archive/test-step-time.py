import time
import environment
import gymnasium as gym
import cProfile
import pstats


env = gym.make('PredatorPrey-v0')
env.reset()

profiler = cProfile.Profile()
profiler.enable()

# start_time = time.time()
for _ in range(1000):  # 运行1000步来获得更准确的平均时间
    action = env.action_space.sample()  # 假设随机动作
    _, _, _, done, _ = env.step(action)
    if done:
        env.reset()
# end_time = time.time()

# average_time_per_step = (end_time - start_time) / 1000
# print(f"平均每步计算时间:{average_time_per_step:.5f}秒")

profiler.disable()
stats = pstats.Stats(profiler).sort_stats('cumtime')
stats.print_stats()