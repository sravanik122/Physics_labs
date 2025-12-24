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
    ErrorContributionTimeline,
    TrainingRow
)
from .helpers import (
    set_random_seed,
    root_sum_square,
    temperature_expansion_factor,
    surface_roughness_factor,
    user_precision_factor,
    linear_interpolate
)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "sim_config.json"

with open(CONFIG_PATH, "r") as f:
    SIM_CONFIG = json.load(f)

def simulate(inputs: ExperimentInputs) -> SimulationResult:
    """
    Runs full deterministic measurement simulation
    and returns timeline + final state.
    """
    set_random_seed(inputs.seed)

    sampling_interval = (
        inputs.sampling_interval
        if inputs.sampling_interval is not None
        else SIM_CONFIG["default_sampling_interval"]
    )

    total_time = 5.0
    steps = int(total_time / sampling_interval) + 1
    t_values = [i * sampling_interval for i in range(steps)]

    temp_factor = temperature_expansion_factor(
        inputs.temperature,
        SIM_CONFIG["constants"]["reference_temperature"],
        SIM_CONFIG["constants"]["thermal_expansion_coefficient"]
    )

    roughness_factor = surface_roughness_factor(
        inputs.surface_roughness,
        SIM_CONFIG["constants"]["surface_roughness_factor"]
    )

    precision_factor = user_precision_factor(
        inputs.user_precision,
        SIM_CONFIG["constants"]["user_precision_factor"]
    )
  
    tool_noise_sigma = SIM_CONFIG["noise_levels"][inputs.measuring_tool_type]

    zero_err = inputs.zero_error
    align_err = inputs.alignment_error + roughness_factor
    parallax_err = inputs.parallax_error + precision_factor
    thermal_err = inputs.object_length * (temp_factor - 1)

    A = zero_err + align_err + parallax_err + thermal_err
    k = 1.2  

    observed_series: List[float] = []
    corrected_series: List[float] = []

    for t in t_values:
        deterministic_offset = A * (1 - math.exp(-k * t))
        noise = random.gauss(0, tool_noise_sigma)
        observed = inputs.object_length + deterministic_offset + noise
        corrected = observed - zero_err
        observed_series.append(observed)
        corrected_series.append(corrected)

    zero_series = linear_interpolate(0.0, zero_err, steps)
    align_series = linear_interpolate(0.0, align_err, steps)
    parallax_series = linear_interpolate(0.0, parallax_err, steps)
    thermal_series = linear_interpolate(0.0, thermal_err, steps)

    total_error = root_sum_square(
        [zero_err, align_err, parallax_err, thermal_err]
    )

    final_length = corrected_series[-1]
    accuracy = max(
        0.0,
        100 * (1 - abs(final_length - inputs.object_length) / inputs.object_length)
    )

    final_state = FinalState(
        observed_length=observed_series[-1],
        corrected_length=corrected_series[-1],
        total_measurement_error=total_error,
        final_length=final_length,
        measurement_accuracy=accuracy
    )

    timeline = Timeline(
        t=t_values,
        observed_length=observed_series,
        corrected_length=corrected_series,
        error_contribution=ErrorContributionTimeline(
            zero_error=zero_series,
            alignment_error=align_series,
            parallax_error=parallax_series,
            thermal_error=thermal_series
        )
    )

    visual = {
        "tool_position_offset": A,
        "parallax_angle_deg": SIM_CONFIG["visual"]["max_parallax_angle_deg"]
        * (parallax_err / max(0.001, SIM_CONFIG["input_ranges"]["parallax_error"]["max"])),
        "error_highlight_level": total_error,
        "show_error_overlay": total_error
        > SIM_CONFIG["visual"]["error_highlight_threshold"]
    }

    return SimulationResult(
        final_state=final_state,
        timeline=timeline,
        visual=visual
    )

def simulate_at_time(inputs: ExperimentInputs, t: float) -> FinalState:
    """
    Returns only the state at a specific time t.
    """
    result = simulate(inputs)
    idx = min(range(len(result.timeline.t)), key=lambda i: abs(result.timeline.t[i] - t))

    return FinalState(
        observed_length=result.timeline.observed_length[idx],
        corrected_length=result.timeline.corrected_length[idx],
        total_measurement_error=result.final_state.total_measurement_error,
        final_length=result.timeline.corrected_length[idx],
        measurement_accuracy=result.final_state.measurement_accuracy
    )

def generate_teacher_dataset(
    samples: int = 500
) -> List[TrainingRow]:
    """
    Generates synthetic dataset using deterministic simulation
    for ML model training.
    """
    dataset: List[TrainingRow] = []

    for _ in range(samples):
        inputs = ExperimentInputs(
            object_length=random.uniform(1, 50),
            measuring_tool_type=random.choice(["ruler", "vernier_caliper", "screw_gauge"]),
            least_count=random.choice([0.1, 0.01, 0.001]),
            zero_error=random.uniform(-0.2, 0.2),
            alignment_error=random.uniform(0, 0.1),
            parallax_error=random.uniform(0, 0.1),
            unit_system="cm",
            temperature=random.uniform(10, 40),
            surface_roughness=random.choice(["smooth", "medium", "rough"]),
            user_precision=random.choice(["low", "medium", "high"])
        )

        result = simulate(inputs)

        dataset.append(
            TrainingRow(
                object_length=inputs.object_length,
                least_count=inputs.least_count,
                zero_error=inputs.zero_error,
                alignment_error=inputs.alignment_error,
                parallax_error=inputs.parallax_error,
                temperature=inputs.temperature,
                surface_roughness_factor=SIM_CONFIG["constants"]["surface_roughness_factor"][inputs.surface_roughness],
                user_precision_factor=SIM_CONFIG["constants"]["user_precision_factor"][inputs.user_precision],
                final_length=result.final_state.final_length
            )
        )

    return dataset
