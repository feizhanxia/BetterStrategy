import numpy as np
from utils.kinematics import bound_positions

class Prey:
    def __init__(self, initial_positions, max_speed = 0.1, radius=10, center=np.zeros(2)):
        self.positions = np.array(initial_positions, dtype=np.float32)
        self.max_speed = max_speed
        self.radius = radius
        self.center = center

    def move(self):
        angles = np.random.uniform(-np.pi, np.pi, size=(len(self.positions),))
        distances = np.random.uniform(0, self.max_speed, size=(len(self.positions),),)
        deltas = np.vstack((np.cos(angles) * distances, np.sin(angles) * distances)).T
        self.positions += deltas
        self.positions = bound_positions(self.positions, self.center, self.radius)

    def get_count(self):
        # 返回当前存活的猎物数量
        return len(self.positions)