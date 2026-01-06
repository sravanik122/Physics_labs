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
    Clamp a value between min_value and max_value.
    """
    return max(min_value, min(value, max_value))

def safe_divide(numerator: float, denominator: float) -> float:
    """
    Perform division safely, avoiding division by zero.
    """
    if denominator == 0:
        return 0.0
    return numerator / denominator

def normalize_input(value: float, min_value: float, max_value: float) -> float:
    """
    Normalize value to range [0, 1].
    """
    if max_value == min_value:
        return 0.0
    return (value - min_value) / (max_value - min_value)

def denormalize_output(value: float, min_value: float, max_value: float) -> float:
    """
    Convert normalized value back to original scale.
    """
    return min_value + value * (max_value - min_value)

def smooth_series(values: List[float], alpha: float = 0.3) -> List[float]:
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
    Linear interpolation between start and end.
    """
    return start + t * (end - start)

def temperature_factor(
    temperature: float,
    reference_temperature: float,
    change_per_degree: float
) -> float:
    """
    Compute scaling factor based on temperature deviation.
    """
    return max(
        0.0,
        1 + (temperature - reference_temperature) * change_per_degree
    )

def surface_roughness_factor(
    roughness: float,
    min_factor: float,
    max_factor: float
) -> float:
    """
    Map surface roughness to a scaling factor.
    """
    return clamp(
        min_factor + roughness * (max_factor - min_factor),
        min_factor,
        max_factor
    )

def add_gaussian_noise(value: float, sigma: float) -> float:
    """
    Add Gaussian noise to simulate real-world variations.
    """
    if sigma <= 0:
        return value
    return value + random.gauss(0, sigma)
