import numpy as np

def uniform_circle_sample(center, radius, num_samples):
    samples = []
    while len(samples) < num_samples:
        x, y = np.random.uniform(-1, 1, 2) * radius
        if np.sqrt(x**2 + y**2) <= radius:
            samples.append(center + np.array([x, y]))
    return np.array(samples, dtype=np.float32)