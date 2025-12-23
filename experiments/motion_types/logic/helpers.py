import math
import random
from typing import List

def set_random_seed(seed: int | None = None):
    """
    Set random seed for reproducibility.
    """
    if seed is not None:
        random.seed(seed)

def clamp(value: float, min_value: float, max_value: float) -> float:
    """
    Clamp a value between min and max.
    """
    return max(min_value, min(value, max_value))

def vector_magnitude(x: float, y: float) -> float:
    """
    Compute magnitude of a 2D vector.
    """
    return math.sqrt(x**2 + y**2)

def vector_direction(x: float, y: float) -> float:
    """
    Compute direction (angle in radians) of a 2D vector.
    """
    return math.atan2(y, x)

def normalize_input(value: float, min_value: float, max_value: float) -> float:
    """
    Normalize input to range [0, 1].
    """
    if max_value == min_value:
        return 0.0
    return (value - min_value) / (max_value - min_value)

def denormalize_output(
    value: float,
    min_value: float,
    max_value: float
) -> float:
    """
    Convert normalized value back to original scale.
    """
    return min_value + value * (max_value - min_value)

def smooth_series(
    values: List[float],
    alpha: float = 0.2
) -> List[float]:
    """
    Apply exponential smoothing to a time-series.
    """
    if not values:
        return []

    smoothed = [values[0]]
    for i in range(1, len(values)):
        smoothed.append(alpha * values[i] + (1 - alpha) * smoothed[-1])
    return smoothed

def linear_interpolate(a: float, b: float, t: float) -> float:
    """
    Linear interpolation between a and b.
    """
    return a + t * (b - a)

def damping_factor(
    medium_resistance: float,
    base_damping: float
) -> float:
    """
    Compute damping multiplier based on medium resistance.
    """
    return math.exp(-base_damping * medium_resistance)

def angular_to_linear_velocity(
    angular_velocity: float,
    radius: float
) -> float:
    """
    Convert angular velocity to linear velocity.
    """
    return angular_velocity * radius

def add_gaussian_noise(
    value: float,
    sigma: float
) -> float:
    """
    Add Gaussian noise to a value.
    """
    if sigma <= 0:
        return value
    return value + random.gauss(0, sigma)
