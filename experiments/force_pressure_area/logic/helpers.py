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
    Clamp a value within a specified range.
    """
    return max(min_value, min(value, max_value))

def safe_divide(numerator: float, denominator: float) -> float:
    """
    Safely divide two numbers, avoiding division by zero.
    """
    if denominator == 0:
        return 0.0
    return numerator / denominator

def deg_to_rad(angle_deg: float) -> float:
    """
    Convert degrees to radians.
    """
    return math.radians(angle_deg)

def rad_to_deg(angle_rad: float) -> float:
    """
    Convert radians to degrees.
    """
    return math.degrees(angle_rad)

def normalize_input(value: float, min_value: float, max_value: float) -> float:
    """
    Normalize a value to the range [0, 1].
    """
    if max_value == min_value:
        return 0.0
    return (value - min_value) / (max_value - min_value)

def denormalize_output(value: float, min_value: float, max_value: float) -> float:
    """
    Convert a normalized value back to original scale.
    """
    return min_value + value * (max_value - min_value)

def smooth_series(values: List[float], alpha: float = 0.25) -> List[float]:
    """
    Apply exponential smoothing to a time series.
    """
    if not values:
        return []

    smoothed = [values[0]]
    for i in range(1, len(values)):
        smoothed.append(alpha * values[i] + (1 - alpha) * smoothed[-1])
    return smoothed

def linear_interpolate(start: float, end: float, t: float) -> float:
    """
    Linear interpolation between two values.
    """
    return start + t * (end - start)

def normal_force_component(force: float, angle_deg: float) -> float:
    """
    Compute the normal component of force based on application angle.
    """
    return force * math.cos(deg_to_rad(angle_deg))

def surface_contact_factor(roughness: float) -> float:
    """
    Convert surface roughness into a contact efficiency factor.
    """
    return clamp(1.0 - roughness, 0.0, 1.0)

def shape_concentration_factor(shape: str, shape_map: dict) -> float:
    """
    Get pressure concentration factor based on contact shape.
    """
    return shape_map.get(shape, 1.0)

def add_gaussian_noise(value: float, sigma: float) -> float:
    """
    Add Gaussian noise to simulate real-world variations.
    """
    if sigma <= 0:
        return value
    return value + random.gauss(0, sigma)
