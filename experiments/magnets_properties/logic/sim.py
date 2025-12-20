import json
import math
from pathlib import Path
from typing import List

from .types import (
    ExperimentInputs,
    FinalState,
    Timeline,
    SimulationResult,
    FieldLine,
    ForceVector,
    PolePosition,
    VisualData,
    TrainingRow,
)
from .helpers import (
    set_random_seed,
    inverse_distance_decay,
    temperature_factor,
    material_factor,
    shape_factor,
    medium_factor,
    pole_interaction_factor,
    add_gaussian_noise,
    angle_between_points,
    clamp,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "sim_config.json"

with open(CONFIG_PATH, "r") as f:
    SIM_CONFIG = json.load(f)

def simulate(inputs: ExperimentInputs) -> SimulationResult:
    """
    Run full magnetic simulation and return timeline + visuals.
    """

    set_random_seed(inputs.seed)

    #Resolve time parameters
    total_time = inputs.total_time or SIM_CONFIG.get("default_total_time", 10)
    dt = inputs.sampling_interval or SIM_CONFIG.get("default_sampling_interval", 0.1)

    t_values: List[float] = []
    force_series: List[float] = []
    field_series: List[float] = []

    #Read constants 
    constants = SIM_CONFIG["constants"]
    noise_sigma = SIM_CONFIG.get("noise_sigma", 0.0)

    #Domain factors
    temp_factor = temperature_factor(
        inputs.temperature,
        constants["temperature_effect"]["reference_temperature"],
        constants["temperature_effect"]["strength_loss_per_degree"],
    )

    mat_factor = material_factor(
        inputs.material_type,
        constants["material_susceptibility"],
    )

    shp_factor = shape_factor(
        inputs.shape_of_magnet,
        constants["shape_factor"],
    )

    med_factor = medium_factor(
        inputs.surrounding_medium,
        constants["surrounding_medium_factor"],
    )

    pole_factor = pole_interaction_factor(inputs.pole_orientation)

    #Base coefficients
    A = (
        inputs.magnet_strength
        * inputs.magnetic_field_intensity
        * mat_factor
        * shp_factor
        * med_factor
        * temp_factor
        * inputs.number_of_magnets
    )

    k = inverse_distance_decay(
        max(inputs.distance_from_object, 0.001),
        constants["distance_decay_power"],
    )

    t = 0.0
    while t <= total_time:
        # Magnetic field intensity model
        field = A * math.exp(-k * t)

        # Force proportional to field gradient
        force = pole_factor * field * k

        # Add noise
        field = add_gaussian_noise(field, noise_sigma)
        force = add_gaussian_noise(force, noise_sigma)

        t_values.append(t)
        field_series.append(field)
        force_series.append(force)

        t += dt

    final_field = field_series[-1]
    final_force = force_series[-1]

    attraction_type = (
        "attract" if final_force > 0 else
        "repel" if final_force < 0 else
        "none"
    )

    final_state = FinalState(
        attraction_force=final_force,
        field_intensity=final_field,
        attraction_type=attraction_type,
        effective_range=1 / (k + 1e-6),
        temperature_loss_factor=temp_factor,
    )
    # Field lines (simple radial pattern)
    field_lines: List[FieldLine] = []
    max_lines = SIM_CONFIG["visual"]["max_field_lines"]

    for i in range(max_lines):
        angle = (2 * math.pi / max_lines) * i
        r = 1 + i * 0.1
        strength = clamp(final_field / (r + 1e-6), 0.0, 1.0)

        field_lines.append(
            FieldLine(
                x=r * math.cos(angle),
                y=r * math.sin(angle),
                strength=strength,
            )
        )

    # Force vectors 
    force_vectors = [
        ForceVector(
            x=0.0,
            y=0.0,
            magnitude=abs(final_force),
            direction=0.0,
        )
    ]

    # Pole positions 
    pole_positions = [
        PolePosition(pole="N", x=-0.5, y=0.0),
        PolePosition(pole="S", x=0.5, y=0.0),
    ]

    visual = VisualData(
        field_lines=field_lines,
        force_vectors=force_vectors,
        pole_positions=pole_positions,
        animation={
            "field_line_speed": 1.0,
            "force_arrow_scale": 1.0,
            "show_repulsion": attraction_type == "repel",
        },
    )

    timeline = Timeline(
        t=t_values,
        attraction_force=force_series,
        field_intensity=field_series,
    )

    return SimulationResult(
        final_state=final_state,
        timeline=timeline,
        visual=visual,
    )

def simulate_at_time(inputs: ExperimentInputs, t: float) -> FinalState:
    """
    Return final state at a specific time t.
    """
    result = simulate(inputs)
    index = min(int(t / (inputs.sampling_interval or 0.1)), len(result.timeline.t) - 1)

    return FinalState(
        attraction_force=result.timeline.attraction_force[index],
        field_intensity=result.timeline.field_intensity[index],
        attraction_type=result.final_state.attraction_type,
        effective_range=result.final_state.effective_range,
        temperature_loss_factor=result.final_state.temperature_loss_factor,
    )
  
def generate_teacher_dataset(
    input_samples: List[ExperimentInputs]
) -> List[TrainingRow]:
    """
    Generate labeled dataset using deterministic simulation.
    """

    dataset: List[TrainingRow] = []

    for inp in input_samples:
        result = simulate(inp)

        dataset.append(
            TrainingRow(
                inputs=inp.dict(exclude={"seed"}),
                outputs={
                    "attraction_force": result.final_state.attraction_force,
                    "field_intensity": result.final_state.field_intensity,
                    "effective_range": result.final_state.effective_range,
                },
            )
        )

    return dataset
