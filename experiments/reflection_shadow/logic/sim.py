import json
from pathlib import Path
from typing import List

from .types import (
    ExperimentInputs,
    FinalState,
    Timeline,
    SimulationResult,
    VisualData,
    TrainingRow
)

from .helpers import (
    set_random_seed,
    reflection_angle,
    intensity_falloff,
    shadow_scale,
    surface_scatter_factor,
    add_gaussian_noise
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "sim_config.json"

with open(CONFIG_PATH, "r") as f:
    SIM_CONFIG = json.load(f)

def simulate(inputs: ExperimentInputs) -> SimulationResult:
    """
    Run deterministic reflection & shadow simulation.
    """

    set_random_seed(inputs.seed)

    dt = inputs.sampling_interval or SIM_CONFIG["default_sampling_interval"]
    total_time = inputs.observation_duration or SIM_CONFIG["default_total_time"]
    noise_sigma = SIM_CONFIG["noise_sigma"]

    t_values = []
    inc_angles = []
    ref_angles = []
    reflected_intensity_series = []
    shadow_size_series = []
    shadow_intensity_series = []

    mirror_props = SIM_CONFIG["constants"]["mirror_types"][inputs.mirror_type]
    reflection_efficiency = mirror_props["reflection_efficiency"]

    t = 0.0
    while t <= total_time:

        angle_i = inputs.angle_of_incidence
        angle_r = reflection_angle(angle_i)

        intensity_at_surface = intensity_falloff(
            inputs.light_intensity,
            inputs.object_distance
        )

        reflected_intensity = (
            intensity_at_surface *
            reflection_efficiency *
            inputs.surface_smoothness
        )

        shadow_size_val = shadow_scale(
            inputs.object_size,
            inputs.object_distance,
            inputs.screen_distance
        )

        shadow_intensity = (
            reflected_intensity *
            (1 - inputs.ambient_light)
        )

        # Add noise
        reflected_intensity = add_gaussian_noise(reflected_intensity, noise_sigma)
        shadow_size_val = add_gaussian_noise(shadow_size_val, noise_sigma)
        shadow_intensity = add_gaussian_noise(shadow_intensity, noise_sigma)

        t_values.append(t)
        inc_angles.append(angle_i)
        ref_angles.append(angle_r)
        reflected_intensity_series.append(reflected_intensity)
        shadow_size_series.append(shadow_size_val)
        shadow_intensity_series.append(shadow_intensity)

        t += dt

    final_state = FinalState(
        angle_of_reflection=ref_angles[-1],
        image_distance=inputs.object_distance,
        shadow_size=shadow_size_series[-1],
        shadow_intensity=shadow_intensity_series[-1],
        reflected_intensity=reflected_intensity_series[-1]
    )

    visual = VisualData(
        rays={
            "incident_angle": inputs.angle_of_incidence,
            "reflected_angle": ref_angles[-1],
            "ray_count": SIM_CONFIG["visual"]["max_rays"],
            "scattering_factor": surface_scatter_factor(inputs.surface_smoothness)
        },
        shadow={
            "size": shadow_size_series[-1],
            "blur_radius": SIM_CONFIG["visual"]["shadow"]["max_blur_radius"]
                          * (1 - inputs.surface_smoothness),
            "opacity": max(0.0, 1 - inputs.ambient_light)
        },
        mirror={
            "type": inputs.mirror_type
        },
        light={
            "wavelength": inputs.wavelength,
            "color_hint": "visible"
        },
        animation={
            "show_incident_ray": True,
            "show_reflected_ray": True,
            "show_normal": True,
            "show_shadow": True,
            "recommended_fps": SIM_CONFIG["visual"]["animation"]["recommended_fps"]
        }
    )

    timeline = Timeline(
        t=t_values,
        angle_of_incidence=inc_angles,
        angle_of_reflection=ref_angles,
        reflected_intensity=reflected_intensity_series,
        shadow_size=shadow_size_series,
        shadow_intensity=shadow_intensity_series
    )

    return SimulationResult(
        final_state=final_state,
        timeline=timeline,
        visual=visual
    )

def simulate_at_time(inputs: ExperimentInputs, t: float) -> FinalState:
    """
    Return final state snapshot at time t.
    """
    result = simulate(inputs)
    index = min(
        int(t / inputs.sampling_interval),
        len(result.timeline.t) - 1
    )
    return result.final_state

def generate_teacher_dataset(
    input_samples: List[ExperimentInputs]
) -> List[TrainingRow]:
    """
    Generate dataset for ML training using deterministic simulation.
    """

    dataset: List[TrainingRow] = []

    for inp in input_samples:
        result = simulate(inp)

        dataset.append(
            TrainingRow(
                inputs=inp.dict(exclude={"seed"}, exclude_none=True),
                outputs={
                    "effective_reflection_coefficient": result.final_state.reflected_intensity,
                    "shadow_size_factor": result.final_state.shadow_size,
                    "shadow_blur_factor": surface_scatter_factor(inp.surface_smoothness)
                }
            )
        )

    return dataset
