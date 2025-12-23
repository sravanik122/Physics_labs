import json
import math
from pathlib import Path
from typing import List

from .types import (
    ExperimentInputs,
    FinalState,
    Timeline,
    SimulationResult,
    VisualData,
    VelocityVector,
    TrainingRow,
)
from .helpers import (
    set_random_seed,
    clamp,
    vector_magnitude,
    vector_direction,
    damping_factor,
    add_gaussian_noise,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "sim_config.json"

with open(CONFIG_PATH, "r") as f:
    SIM_CONFIG = json.load(f)

def simulate(inputs: ExperimentInputs) -> SimulationResult:
    """
    Run deterministic motion simulation.
    """

    set_random_seed(inputs.seed)

    dt = inputs.sampling_interval or SIM_CONFIG["default_sampling_interval"]
    total_time = inputs.observation_duration or SIM_CONFIG["default_total_time"]
    noise_sigma = SIM_CONFIG["noise_sigma"]

    t_values: List[float] = []
    pos_series = []
    vel_series = []
    acc_series = []
    x_series: List[float] = []
    y_series: List[float] = []
    speed_series: List[float] = []

    base_damping = SIM_CONFIG["constants"]["damping"]["linear_damping_factor"]
    damping = damping_factor(inputs.medium_resistance, base_damping)

    t = 0.0
    total_distance = 0.0
    prev_x, prev_y = inputs.initial_position, 0.0

    while t <= total_time:
        if inputs.motion_type == "linear":
            x = inputs.initial_position + inputs.speed * t * damping
            y = 0.0
            vx = inputs.speed * damping
            vy = 0.0
            ax = 0.0
            ay = 0.0

        elif inputs.motion_type == "circular":
            x = inputs.radius_of_path * math.cos(inputs.angular_velocity * t)
            y = inputs.radius_of_path * math.sin(inputs.angular_velocity * t)
            vx = -inputs.radius_of_path * inputs.angular_velocity * math.sin(inputs.angular_velocity * t)
            vy = inputs.radius_of_path * inputs.angular_velocity * math.cos(inputs.angular_velocity * t)
            ax = 0.0
            ay = 0.0

        elif inputs.motion_type == "periodic":
            omega = 2 * math.pi * inputs.frequency
            x = inputs.amplitude * math.sin(omega * t) * damping
            y = 0.0
            vx = inputs.amplitude * omega * math.cos(omega * t) * damping
            vy = 0.0
            ax = -inputs.amplitude * omega**2 * math.sin(omega * t) * damping
            ay = 0.0

        else:
            raise ValueError("Invalid motion type")

        # Add noise
        x = add_gaussian_noise(x, noise_sigma)
        y = add_gaussian_noise(y, noise_sigma)

        distance_step = math.sqrt((x - prev_x)**2 + (y - prev_y)**2)
        total_distance += distance_step
        prev_x, prev_y = x, y

        speed = vector_magnitude(vx, vy)

        t_values.append(t)
        pos_series.append({"x": x, "y": y})
        vel_series.append({"x": vx, "y": vy})
        acc_series.append({"x": ax, "y": ay})
        x_series.append(x)
        y_series.append(y)
        speed_series.append(speed)

        t += dt

    avg_speed = total_distance / total_time if total_time > 0 else 0.0

    final_state = FinalState(
        final_position={"x": x_series[-1], "y": y_series[-1]},
        final_velocity=vel_series[-1],
        total_distance=total_distance,
        average_speed=avg_speed,
        motion_type=inputs.motion_type,
    )

    velocity_vectors = [
        VelocityVector(
            x=vel_series[-1]["x"],
            y=vel_series[-1]["y"],
            magnitude=vector_magnitude(
                vel_series[-1]["x"], vel_series[-1]["y"]
            ),
            direction=vector_direction(
                vel_series[-1]["x"], vel_series[-1]["y"]
            ),
        )
    ]

    visual = VisualData(
        trajectory={"x": x_series, "y": y_series},
        velocity_vectors=velocity_vectors,
        path_type=inputs.motion_type,
        animation={
            "show_path": True,
            "show_velocity_vector": True,
            "trail_length": SIM_CONFIG["visual"]["max_trail_points"],
            "recommended_fps": SIM_CONFIG["visual"]["animation"]["recommended_fps"],
        },
    )

    timeline = Timeline(
        t=t_values,
        position=pos_series,
        velocity=vel_series,
        acceleration=acc_series,
        x=x_series,
        y=y_series,
        speed=speed_series,
    )

    return SimulationResult(
        final_state=final_state,
        timeline=timeline,
        visual=visual,
    )

def simulate_at_time(inputs: ExperimentInputs, t: float) -> FinalState:
    """
    Return motion state at a specific time.
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
                    "average_speed": result.final_state.average_speed,
                    "total_distance": result.final_state.total_distance,
                },
            )
        )

    return dataset
