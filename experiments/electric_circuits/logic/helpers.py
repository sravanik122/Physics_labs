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

def temperature_factor(
    temperature: float,
    reference_temperature: float,
    temperature_coefficient: float
) -> float:
    """
    Compute resistivity scaling factor based on temperature.
    """
    return 1 + temperature_coefficient * (temperature - reference_temperature)

def cross_section_area_mm2_to_m2(area_mm2: float) -> float:
    """
    Convert cross-sectional area from mm² to m².
    """
    return area_mm2 * 1e-6

def rms_value(value: float) -> float:
    """
    Compute RMS value for AC quantities.
    """
    return value / math.sqrt(2)
  
def add_gaussian_noise(value: float, sigma: float) -> float:
    """
    Add Gaussian noise to simulate measurement variations.
    """
    if sigma <= 0:
        return value
    return value + random.gauss(0, sigma)
