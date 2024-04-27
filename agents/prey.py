import numpy as np
from utils.kinematics import bound_positions
from numba import jit

class Prey:
    def __init__(self, initial_positions, D_0 = 0.1, radius=10, center=np.zeros(2)):
        self.rng = np.random.default_rng()
        self.positions = np.array(initial_positions, dtype=np.float32)
        self.D_0 = D_0
        self.radius = radius
        self.center = center

    def move(self):
        # sigma = np.sqrt(2 * self.D_0)
        # deltas = self.rng.normal(loc=0, scale=sigma, size=self.positions.shape)
        # self.positions += deltas
        self.positions = brownian_motion(self.positions, self.D_0, self.rng)
        self.positions = bound_positions(self.positions, self.center, self.radius)

    def get_count(self):
        # 返回当前存活的猎物数量
        return len(self.positions)
    
@jit(nopython=True, parallel=False, cache=True)
def brownian_motion(positions, D_0, rng):
    sigma = np.sqrt(2 * D_0)
    deltas = rng.normal(loc=0, scale=sigma, size=positions.shape)
    return positions + deltas