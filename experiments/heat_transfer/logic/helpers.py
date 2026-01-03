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
    Clamp a value between minimum and maximum limits.
    """
    return max(min_value, min(value, max_value))

def safe_divide(numerator: float, denominator: float) -> float:
    """
    Safely divide two numbers, avoiding division by zero.
    """
    if denominator == 0:
        return 0.0
    return numerator / denominator

def normalize_input(value: float, min_value: float, max_value: float) -> float:
    """
    Normalize a value to range [0, 1].
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

def smooth_series(values: List[float], alpha: float = 0.2) -> List[float]:
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

def temperature_gradient(delta_t: float, distance: float) -> float:
    """
    Compute temperature gradient (ΔT / d).
    """
    return safe_divide(delta_t, distance)

def insulation_factor(level: str, mapping: dict) -> float:
    """
    Map insulation level to numerical reduction factor.
    """
    return mapping.get(level, 1.0)

def convection_mode(airflow_speed: float) -> str:
    """
    Determine convection mode based on airflow speed.
    """
    return "forced" if airflow_speed > 0 else "natural"

def add_gaussian_noise(value: float, sigma: float) -> float:
    """
    Add Gaussian noise to simulate measurement variations.
    """
    if sigma <= 0:
        return value
    return value + random.gauss(0, sigma)
