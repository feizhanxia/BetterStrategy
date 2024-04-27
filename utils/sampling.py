import numpy as np
from numba import jit

@jit(nopython=True, parallel=True)
def uniform_circle_sample(center, radius, num_samples):
    samples = np.empty((num_samples, 2), dtype=np.float32)
    i=0
    while i < num_samples:
        x, y = np.random.uniform(-1, 1, 2) * radius
        if np.sqrt(x**2 + y**2) <= radius:
            samples[i] = center + np.array([x, y])
            i+=1
    return samples