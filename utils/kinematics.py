import numpy as np

# Bound positions
def bound_positions(positions, center, radius):
    # Calculate distance from center
    dist_from_center = np.linalg.norm(positions-center, axis=1)
    outside = dist_from_center > radius  # Check if positions are outside the circle
    # 超出边界的点,将其位置映射到边界上
    positions[outside] = (positions[outside].T * (radius / dist_from_center[outside])).T
    # 超出边界的点,将其位置关于切线反射
    # positions[outside] = (positions[outside].T * (2 * radius/dist_from_center[outside] - 1)).T
    return positions
