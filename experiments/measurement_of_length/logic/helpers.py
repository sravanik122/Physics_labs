import random
from typing import List

def set_random_seed(seed: int | None = None) -> None:
    """
    Sets random seed for reproducibility.
    """
    if seed is not None:
        random.seed(seed)

def clamp(value: float, min_value: float, max_value: float) -> float:
    """
    Restricts a value within a given range.
    """
    return max(min_value, min(value, max_value))

def root_sum_square(errors: List[float]) -> float:
    """
    Combines independent errors using RSS method.
    """
    return sum(e**2 for e in errors) ** 0.5

def normalize_input(value: float, min_value: float, max_value: float) -> float:
    """
    Normalizes value to range [0, 1].
    """
    if max_value == min_value:
        return 0.0
    return (value - min_value) / (max_value - min_value)

def denormalize_output(value: float, min_value: float, max_value: float) -> float:
    """
    Converts normalized value back to original scale.
    """
    return value * (max_value - min_value) + min_value

def moving_average(values: List[float], window: int = 3) -> List[float]:
    """
    Applies simple moving average smoothing.
    """
    if window <= 1:
        return values

    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        segment = values[start:i + 1]
        smoothed.append(sum(segment) / len(segment))
    return smoothed

def linear_interpolate(start: float, end: float, steps: int) -> List[float]:
    """
    Generates linearly interpolated values.
    """
    if steps <= 1:
        return [start]
    step_size = (end - start) / (steps - 1)
    return [start + i * step_size for i in range(steps)]

def temperature_expansion_factor(
    temperature: float,
    reference_temperature: float,
    expansion_coefficient: float
) -> float:
    """
    Computes linear expansion factor due to temperature.
    """
    return 1 + expansion_coefficient * (temperature - reference_temperature)

def surface_roughness_factor(level: str, mapping: dict) -> float:
    """
    Maps surface roughness level to numeric factor.
    """
    return mapping.get(level, 0.0)

def user_precision_factor(level: str, mapping: dict) -> float:
    """
    Maps user precision level to numeric factor.
    """
    return mapping.get(level, 0.0)
