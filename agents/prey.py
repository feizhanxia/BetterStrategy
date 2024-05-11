import numpy as np
from utils.kinematics import bound_positions
from utils.sampling import uniform_circle_sample
from numba import jit

class Prey:
    def __init__(self, num, D_0 = 0.1, radius=10, center=np.zeros(2)):
        # 随机数生成器
        self.rng = np.random.default_rng()
        # 猎物数量, 扩散系数, 猎物半径, 猎物中心
        self.num = num
        self.D_0 = D_0
        self.radius = radius
        self.center = center
        # 初始化猎物
        self.positions = uniform_circle_sample(self.center, self.radius, self.num)

    def move(self):
        # 猎物运动
        self.positions = brownian_motion(self.positions, self.D_0, self.rng)
        self.positions = bound_positions(self.positions, self.center, self.radius)

    def get_count(self):
        # 返回当前存活的猎物数量
        return len(self.positions)
    
    def add(self, num_add = 1):
        # 添加猎物
        add_positions = uniform_circle_sample(self.center, self.radius, num_add)
        self.positions = np.append(self.positions, add_positions, axis=0)
        
    
@jit(nopython=True, parallel=False, cache=True)
def brownian_motion(positions, D_0, rng):
    sigma = np.sqrt(2 * D_0)
    deltas = rng.normal(loc=0, scale=sigma, size=positions.shape)
    return positions + deltas