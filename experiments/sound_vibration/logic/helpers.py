import math
import random
from typing import List

def set_seed(seed: int | None):
    """
    Set random seed for reproducibility.
    """
    if seed is not None:
        random.seed(seed)

def normalize_input(value: float, min_value: float, max_value: float) -> float:
    """
    Normalize value to range [0, 1].
    """
    if max_value == min_value:
        return 0.0
    return (value - min_value) / (max_value - min_value)

def denormalize_value(norm_value: float, min_value: float, max_value: float) -> float:
    """
    Convert normalized value back to original range.
    """
    return min_value + norm_value * (max_value - min_value)

def exponential_decay(initial_value: float, damping: float, t: float) -> float:
    """
    Compute exponentially decaying value.
    """
    return initial_value * math.exp(-damping * t)

def sine_wave(amplitude: float, frequency: float, t: float) -> float:
    """
    Compute sine wave displacement.
    """
    return amplitude * math.sin(2 * math.pi * frequency * t)

def wavelength_from_frequency(speed_of_sound: float, frequency: float) -> float:
    """
    Compute wavelength of sound.
    """
    if frequency <= 0:
        return 0.0
    return speed_of_sound / frequency

def sound_intensity_from_amplitude(amplitude: float) -> float:
    """
    Approximate sound intensity from vibration amplitude.
    """
    return amplitude ** 2

def loudness_db(intensity: float, reference_intensity: float) -> float:
    """
    Convert sound intensity to loudness level in decibels.
    """
    if intensity <= 0:
        return 0.0
    return 10 * math.log10(intensity / reference_intensity)

def distance_attenuation(intensity: float, distance: float) -> float:
    """
    Apply inverse-square law for sound attenuation with distance.
    """
    if distance <= 0:
        return intensity
    return intensity / (distance ** 2)

def resonance_boost(amplitude: float, resonance: bool, factor: float) -> float:
    """
    Amplify amplitude under resonance conditions.
    """
    return amplitude * factor if resonance else amplitude

def smooth_series(values: List[float], alpha: float = 0.1) -> List[float]:
    """
    Apply exponential smoothing to a time series.
    """
    if not values:
        return []

    smoothed = [values[0]]
    for v in values[1:]:
        smoothed.append(alpha * v + (1 - alpha) * smoothed[-1])
    return smoothed
