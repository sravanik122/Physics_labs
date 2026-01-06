import json
import math
from pathlib import Path
from typing import List

import numpy as np

from .types import (
    ExperimentInputs,
    FinalState,
    Timeline,
    SimulationResult,
    TrainingRow
)
from .helpers import (
    set_random_seed,
    safe_divide,
    temperature_factor,
    surface_roughness_factor,
    add_gaussian_noise
)

CONFIG_PATH = Path(__file__).parents[1] / "config" / "sim_config.json"

with open(CONFIG_PATH, "r") as f:
    SIM_CONFIG = json.load(f)

def simulate(inputs: ExperimentInputs) -> SimulationResult:
    """
    Run deterministic friction simulation.
    """

    set_random_seed(inputs.seed)

    # Time setup 
    total_time = inputs.time
    dt = inputs.sampling_interval
    t_values = np.arange(0, total_time + dt, dt)

    noise_sigma = SIM_CONFIG["noise_sigma"]
    constants = SIM_CONFIG["constants"]

    # Base friction coefficient 
    base_mu = constants["base_friction_coefficients"][
        inputs.material_pair
    ][inputs.motion_type]

    # Apply scaling factors
    mu = base_mu

    mu *= constants["lubrication_factor"][inputs.lubrication_level]
    mu *= constants["surface_contamination_factor"][inputs.surface_contamination]
    mu *= constants["environment_factor"][inputs.environmental_conditions]
    mu *= constants["wear_factor"][inputs.wear_factor]

    mu *= surface_roughness_factor(
        inputs.surface_roughness,
        constants["surface_roughness_factor"]["min"],
        constants["surface_roughness_factor"]["max"]
    )

    mu *= temperature_factor(
        inputs.temperature,
        constants["temperature_effect"]["reference_temperature"],
        constants["temperature_effect"]["friction_change_per_degree"]
    )

    # Parametric curve parameters 
    A_mu = mu
    k = safe_divide(inputs.speed_of_motion + 0.1, inputs.contact_area + 0.001)

    # Time series generation 
    friction_force_series = []
    mu_series = []
    heat_series = []
    motion_series = []

    for t in t_values:
        mu_t = A_mu * (1 - math.exp(-k * t))
        friction_force = mu_t * inputs.normal_force
        heat_generated = friction_force * inputs.speed_of_motion * dt

        mu_t = add_gaussian_noise(mu_t, noise_sigma)
        friction_force = add_gaussian_noise(friction_force, noise_sigma)
        heat_generated = add_gaussian_noise(heat_generated, noise_sigma)

        mu_series.append(mu_t)
        friction_force_series.append(friction_force)
        heat_series.append(heat_generated)
        motion_series.append(inputs.speed_of_motion)

    # Final values 
    final_mu = mu_series[-1]
    final_friction = friction_force_series[-1]
    total_heat = sum(heat_series)
    energy_loss = total_heat

    final_state = FinalState(
        friction_force=final_friction,
        coefficient_of_friction=final_mu,
        heat_generated=total_heat,
        energy_loss=energy_loss
    )

    timeline = Timeline(
        t=t_values.tolist(),
        friction_force=friction_force_series,
        coefficient_of_friction=mu_series,
        heat_generated=heat_series,
        relative_motion=motion_series
    )

    visual = SIM_CONFIG["visual"]

    return SimulationResult(
        final_state=final_state,
        timeline=timeline,
        visual=visual
    )

def simulate_at_time(inputs: ExperimentInputs, t: float) -> FinalState:
    """
    Compute friction state at a specific time t.
    """
    inputs = inputs.copy()
    inputs.time = t
    return simulate(inputs).final_state

def generate_teacher_dataset(
    inputs_list: List[ExperimentInputs]
) -> List[TrainingRow]:
    """
    Generate ML training dataset using deterministic simulation.
    """

    dataset = []

    for inp in inputs_list:
        result = simulate(inp)

        dataset.append(
            TrainingRow(
                inputs={
                    "normal_force": inp.normal_force,
                    "surface_roughness": inp.surface_roughness,
                    "speed_of_motion": inp.speed_of_motion,
                    "contact_area": inp.contact_area,
                    "temperature": inp.temperature
                },
                outputs={
                    "friction_force": result.final_state.friction_force,
                    "coefficient_of_friction": result.final_state.coefficient_of_friction
                }
            )
        )

    return dataset
