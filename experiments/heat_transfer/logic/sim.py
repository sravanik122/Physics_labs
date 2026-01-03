import json
from pathlib import Path
from typing import List

from .types import (
    ExperimentInputs,
    FinalState,
    Timeline,
    SimulationResult,
    VisualData,
    TrainingRow,
)
from .helpers import (
    set_random_seed,
    temperature_gradient,
    insulation_factor,
    convection_mode,
    add_gaussian_noise,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "sim_config.json"

with open(CONFIG_PATH, "r") as f:
    SIM_CONFIG = json.load(f)

def simulate(inputs: ExperimentInputs) -> SimulationResult:
    """
    Run deterministic heat-transfer simulation.
    """

    set_random_seed(inputs.seed)

    dt = inputs.sampling_interval or SIM_CONFIG["default_sampling_interval"]
    total_time = inputs.observation_duration or SIM_CONFIG["default_total_time"]
    noise_sigma = SIM_CONFIG["noise_sigma"]

    t_values: List[float] = []
    temperature_series: List[float] = []
    heat_rate_series: List[float] = []
    cond_series: List[float] = []
    conv_series: List[float] = []
    rad_series: List[float] = []

    k_material = SIM_CONFIG["constants"]["material_conductivity"][inputs.material_type]
    emissivity = inputs.emissivity
    sigma = SIM_CONFIG["constants"]["stefan_boltzmann_constant"]

    convection_env = SIM_CONFIG["constants"]["convection_coefficients"][inputs.medium_type]
    mode = convection_mode(inputs.airflow_speed)
    h = convection_env[mode]

    insulation_map = SIM_CONFIG["constants"]["insulation_factor"]
    insulation = insulation_factor(inputs.insulation_level, insulation_map)

    A_cond = k_material * inputs.surface_area * insulation
    A_conv = h * inputs.surface_area * insulation
    A_rad = emissivity * sigma * inputs.surface_area

    k_decay = 0.02 + (inputs.airflow_speed * 0.01)

    t = 0.0
    delta_t = inputs.temperature_difference

    while t <= total_time:

        grad = temperature_gradient(delta_t, inputs.distance_from_source)

        q_cond = A_cond * grad
        q_conv = A_conv * delta_t
        q_rad = A_rad * ((delta_t + 273.15) ** 4 - 273.15 ** 4)

        q_total = q_cond + q_conv + q_rad

        # Add noise
        q_cond = add_gaussian_noise(q_cond, noise_sigma)
        q_conv = add_gaussian_noise(q_conv, noise_sigma)
        q_rad = add_gaussian_noise(q_rad, noise_sigma)
        q_total = add_gaussian_noise(q_total, noise_sigma)

        temperature = delta_t * (1 - (t / total_time))

        t_values.append(t)
        temperature_series.append(temperature)
        heat_rate_series.append(q_total)
        cond_series.append(q_cond)
        conv_series.append(q_conv)
        rad_series.append(q_rad)

        t += dt

    final_state = FinalState(
        final_temperature=temperature_series[-1],
        total_heat_transfer=sum(heat_rate_series) * dt,
        conduction_heat=sum(cond_series) * dt,
        convection_heat=sum(conv_series) * dt,
        radiation_heat=sum(rad_series) * dt,
    )

    visual = VisualData(
        conduction={
            "heat_gradient_intensity": abs(cond_series[-1]),
            "color_map": SIM_CONFIG["visual"]["heat_color_map"],
        },
        convection={
            "flow_strength": inputs.airflow_speed,
            "arrow_density": SIM_CONFIG["visual"]["convection"]["max_arrow_count"],
        },
        radiation={
            "wave_intensity": emissivity,
            "wave_count": SIM_CONFIG["visual"]["radiation"]["wave_density"],
        },
        animation={
            "show_conduction": True,
            "show_convection": True,
            "show_radiation": True,
            "recommended_fps": SIM_CONFIG["visual"]["animation"]["recommended_fps"],
        },
    )

    timeline = Timeline(
        t=t_values,
        temperature=temperature_series,
        heat_transfer_rate=heat_rate_series,
        conduction_rate=cond_series,
        convection_rate=conv_series,
        radiation_rate=rad_series,
    )

    return SimulationResult(
        final_state=final_state,
        timeline=timeline,
        visual=visual,
    )

def simulate_at_time(inputs: ExperimentInputs, t: float) -> FinalState:
    """
    Return heat-transfer state at a specific time.
    """
    result = simulate(inputs)
    index = min(
        int(t / inputs.sampling_interval),
        len(result.timeline.t) - 1,
    )
    return result.final_state

def generate_teacher_dataset(
    input_samples: List[ExperimentInputs],
) -> List[TrainingRow]:
    """
    Generate labeled dataset using deterministic simulation.
    """

    dataset: List[TrainingRow] = []

    for inp in input_samples:
        result = simulate(inp)

        dataset.append(
            TrainingRow(
                inputs=inp.dict(exclude={"seed"}, exclude_none=True),
                outputs={
                    "effective_conductivity": result.final_state.conduction_heat,
                    "effective_heat_transfer_coefficient": result.final_state.convection_heat,
                    "effective_emissivity": result.final_state.radiation_heat,
                },
            )
        )

    return dataset
