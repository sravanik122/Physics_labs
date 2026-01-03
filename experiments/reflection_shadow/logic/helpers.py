import math
import random
from typing import List

def set_random_seed(seed: int | None = None):
    """
    Set random seed for reproducibility.
    """
    if seed is not None:
        random.seed(seed)

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

def clamp(value: float, min_value: float, max_value: float) -> float:
    """
    Clamp a value within bounds.
    """
    return max(min_value, min(value, max_value))

def safe_divide(numerator: float, denominator: float) -> float:
    """
    Safely divide two numbers.
    """
    if denominator == 0:
        return 0.0
    return numerator / denominator

def normalize_input(value: float, min_value: float, max_value: float) -> float:
    """
    Normalize value to [0, 1].
    """
    if max_value == min_value:
        return 0.0
    return (value - min_value) / (max_value - min_value)

def denormalize_output(value: float, min_value: float, max_value: float) -> float:
    """
    Convert normalized value back to original scale.
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

def linear_interpolate(a: float, b: float, t: float) -> float:
    """
    Linear interpolation between a and b.
    """
    return a + t * (b - a)

def reflection_angle(angle_of_incidence: float) -> float:
    """
    Law of reflection helper: θᵢ = θᵣ
    """
    return angle_of_incidence

def intensity_falloff(intensity: float, distance: float) -> float:
    """
    Inverse square law helper for light intensity.
    """
    return safe_divide(intensity, distance ** 2)

def shadow_scale(object_size: float, object_distance: float, screen_distance: float) -> float:
    """
    Compute relative shadow scaling factor.
    """
    return safe_divide(object_size * screen_distance, object_distance)

def surface_scatter_factor(surface_smoothness: float) -> float:
    """
    Convert surface smoothness to scattering factor.
    """
    return clamp(1.0 - surface_smoothness, 0.0, 1.0)

def add_gaussian_noise(value: float, sigma: float) -> float:
    """
    Add Gaussian noise to simulate real-world variation.
    """
    if sigma <= 0:
        return value
    return value + random.gauss(0, sigma)
