import math
import random
from typing import List, Dict, Optional

def set_random_seed(seed: Optional[int] = None) -> None:
    """
    Set random seed for reproducibility.
    """
    if seed is not None:
        random.seed(seed)

def normalize_input(value: float, min_value: float, max_value: float) -> float:
    """
    Normalize a value into [0, 1] range.
    """
    if max_value == min_value:
        return 0.0
    return (value - min_value) / (max_value - min_value)

def denormalize_output(value: float, min_value: float, max_value: float) -> float:
    """
    Convert normalized value back to original range.
    """
    return min_value + value * (max_value - min_value)

def clamp(value: float, min_value: float, max_value: float) -> float:
    """
    Clamp a value within [min_value, max_value].
    """
    return max(min_value, min(value, max_value))

def inverse_distance_decay(distance: float, power: float = 2) -> float:
    """
    Generic inverse-distance decay function.
    Used for magnetic field / force falloff.
    """
    return 1.0 / (distance ** power + 1e-6)

def safe_log(value: float) -> float:
    """
    Safe logarithm to avoid math domain errors.
    """
    return math.log(max(value, 1e-6))

def smooth_series(values: List[float], alpha: float = 0.2) -> List[float]:
    """
    Apply exponential smoothing to a time series.
    """
    if not values:
        return []

    smoothed = [values[0]]
    for v in values[1:]:
        smoothed.append(alpha * v + (1 - alpha) * smoothed[-1])
    return smoothed

def linear_interpolate(a: float, b: float, t: float) -> float:
    """
    Linear interpolation between a and b.
    """
    return a + t * (b - a)

def temperature_factor(
    temperature: float,
    reference_temperature: float,
    loss_per_degree: float
) -> float:
    """
    Compute magnetic strength reduction due to temperature.
    """
    delta = abs(temperature - reference_temperature)
    factor = 1.0 - delta * loss_per_degree
    return clamp(factor, 0.0, 1.0)

def material_factor(material: str, susceptibility_map: Dict[str, float]) -> float:
    """
    Return susceptibility factor for a material.
    """
    return susceptibility_map.get(material, 0.0)

def shape_factor(shape: str, shape_map: Dict[str, float]) -> float:
    """
    Return amplification factor based on magnet shape.
    """
    return shape_map.get(shape, 1.0)

def medium_factor(medium: str, medium_map: Dict[str, float]) -> float:
    """
    Return environmental medium effect factor.
    """
    return medium_map.get(medium, 1.0)

def pole_interaction_factor(orientation: str) -> float:
    """
    Return interaction sign based on pole orientation.
    """
    if orientation == "N-S":
        return 1.0
    if orientation in ("N-N", "S-S"):
        return -1.0
    return 0.0

def angle_between_points(x1: float, y1: float, x2: float, y2: float) -> float:
    """
    Compute angle (radians) from point (x1, y1) to (x2, y2).
    """
    return math.atan2(y2 - y1, x2 - x1)

def vector_magnitude(x: float, y: float) -> float:
    """
    Compute magnitude of a 2D vector.
    """
    return math.sqrt(x * x + y * y)

def add_gaussian_noise(value: float, sigma: float) -> float:
    """
    Add Gaussian noise to a value.
    """
    if sigma <= 0:
        return value
    return value + random.gauss(0, sigma)
