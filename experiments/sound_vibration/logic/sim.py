import json
import math
import random
from pathlib import Path
from typing import List

from .types import (
    ExperimentInputs,
    FinalState,
    Timeline,
    SimulationResult,
    TrainingRow
)
from .helpers import (
    set_seed,
    sine_wave,
    exponential_decay,
    wavelength_from_frequency,
    sound_intensity_from_amplitude,
    loudness_db,
    distance_attenuation,
    resonance_boost,
    smooth_series
)

CONFIG_PATH = Path(__file__).parents[1] / "config" / "sim_config.json"

with open(CONFIG_PATH, "r") as f:
    CONFIG = json.load(f)

def simulate(inputs: ExperimentInputs) -> SimulationResult:
    """
    Run deterministic sound simulation.
    """

    set_seed(inputs.seed)

    sampling_interval = inputs.sampling_interval
    total_time = inputs.observation_time

    speed_of_sound = CONFIG["constants"]["speed_of_sound"][inputs.medium_type]
    reference_intensity = CONFIG["constants"]["reference_intensity"]
    resonance_factor = CONFIG["constants"]["resonance_amplification_factor"]

    base_amplitude = resonance_boost(
        inputs.amplitude,
        inputs.resonance,
        resonance_factor
    )

    times = []
    displacement = []
    intensities = []
    loudness_levels = []
    wave_amplitudes = []

    t = 0.0
    while t <= total_time:
        disp = sine_wave(base_amplitude, inputs.frequency, t)
        amp = exponential_decay(abs(disp), inputs.damping, t)

        intensity = sound_intensity_from_amplitude(amp)
        intensity = distance_attenuation(intensity, inputs.distance_from_source)

        # Add noise
        intensity += random.gauss(0, CONFIG["noise_sigma"])

        loudness = loudness_db(intensity, reference_intensity)

        times.append(t)
        displacement.append(disp)
        intensities.append(intensity)
        loudness_levels.append(loudness)
        wave_amplitudes.append(amp)

        t += sampling_interval

    # Smooth curves for better visualization
    intensities = smooth_series(intensities)
    loudness_levels = smooth_series(loudness_levels)
    wave_amplitudes = smooth_series(wave_amplitudes)

    wavelength = wavelength_from_frequency(speed_of_sound, inputs.frequency)

    final_state = FinalState(
        sound_intensity=intensities[-1],
        loudness_level=loudness_levels[-1],
        wavelength=wavelength,
        energy_decay=wave_amplitudes[-1]
    )

    timeline = Timeline(
        t=times,
        displacement=displacement,
        sound_intensity=intensities,
        loudness_level=loudness_levels,
        wave_amplitude=wave_amplitudes
    )

    visual = CONFIG["visual"]

    return SimulationResult(
        final_state=final_state,
        timeline=timeline,
        visual=visual
    )

def simulate_at_time(inputs: ExperimentInputs, t: float) -> FinalState:
    """
    Compute sound state at a specific time.
    """

    speed_of_sound = CONFIG["constants"]["speed_of_sound"][inputs.medium_type]
    reference_intensity = CONFIG["constants"]["reference_intensity"]
    resonance_factor = CONFIG["constants"]["resonance_amplification_factor"]

    amplitude = resonance_boost(
        inputs.amplitude,
        inputs.resonance,
        resonance_factor
    )

    disp = sine_wave(amplitude, inputs.frequency, t)
    amp = exponential_decay(abs(disp), inputs.damping, t)

    intensity = sound_intensity_from_amplitude(amp)
    intensity = distance_attenuation(intensity, inputs.distance_from_source)

    loudness = loudness_db(intensity, reference_intensity)
    wavelength = wavelength_from_frequency(speed_of_sound, inputs.frequency)

    return FinalState(
        sound_intensity=intensity,
        loudness_level=loudness,
        wavelength=wavelength,
        energy_decay=amp
    )

def generate_teacher_dataset(
    inputs_list: List[ExperimentInputs]
) -> List[TrainingRow]:
    """
    Generate dataset for ML training.
    """

    dataset = []

    for inputs in inputs_list:
        result = simulate(inputs)

        dataset.append(
            TrainingRow(
                inputs={
                    "frequency": inputs.frequency,
                    "amplitude": inputs.amplitude,
                    "distance_from_source": inputs.distance_from_source,
                    "damping": inputs.damping
                },
                outputs={
                    "sound_intensity": result.final_state.sound_intensity,
                    "loudness_level": result.final_state.loudness_level,
                    "wavelength": result.final_state.wavelength
                }
            )
        )

    return dataset
