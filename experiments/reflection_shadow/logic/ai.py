import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import TensorBoard

from .types import (
    ExperimentInputs,
    FinalState,
    Timeline,
    VisualData,
    SimulationResult,
    TrainingRow,
)

from .helpers import (
    reflection_angle,
    intensity_falloff,
    shadow_scale,
    surface_scatter_factor,
    clamp,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "model_config.json"

with open(CONFIG_PATH, "r") as f:
    MODEL_CONFIG = json.load(f)

def build_model(input_dim: int, output_dim: int) -> tf.keras.Model:
    model = models.Sequential()
    model.add(layers.Input(shape=(input_dim,)))

    for units in MODEL_CONFIG["model_layers"]:
        model.add(layers.Dense(units, activation="relu"))

    model.add(layers.Dense(output_dim, activation="linear"))

    optimizer = optimizers.Adam(
        learning_rate=MODEL_CONFIG["optimizer"]["learning_rate"]
    )

    model.compile(
        optimizer=optimizer,
        loss=MODEL_CONFIG["loss"],
        metrics=MODEL_CONFIG["metrics"],
    )

    return model

def train_model(
    training_data: List[TrainingRow],
    input_keys: List[str],
    output_keys: List[str],
) -> Dict:
    X, y = [], []

    for row in training_data:
        X.append([float(row.inputs[k]) for k in input_keys])
        y.append([float(row.outputs[k]) for k in output_keys])

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    model = build_model(X.shape[1], y.shape[1])

    log_dir = MODEL_CONFIG["tensorboard_logdir"]
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    tensorboard_cb = TensorBoard(log_dir=log_dir)

    history = model.fit(
        X,
        y,
        epochs=MODEL_CONFIG["epochs"],
        batch_size=MODEL_CONFIG["batch_size"],
        validation_split=MODEL_CONFIG["validation_split"],
        callbacks=[tensorboard_cb],
        verbose=0,
    )

    model.save(Path(log_dir) / "model.keras")

    return {
        "loss": history.history["loss"],
        "val_loss": history.history.get("val_loss", []),
        "epochs": MODEL_CONFIG["epochs"],
    }

def load_model(model_path: Path) -> tf.keras.Model:
    return tf.keras.models.load_model(model_path)

def params_to_timeline(
    params: Dict[str, float],
    inputs: ExperimentInputs,
    total_time: float,
    sampling_interval: float,
) -> Timeline:
    t_values = []
    inc_angles = []
    ref_angles = []
    reflected_intensity_series = []
    shadow_size_series = []
    shadow_intensity_series = []

    t = 0.0
    while t <= total_time:

        angle_i = inputs.angle_of_incidence
        angle_r = reflection_angle(angle_i)

        intensity_surface = intensity_falloff(
            inputs.light_intensity,
            inputs.object_distance
        )

        reflected_intensity = (
            intensity_surface *
            params["effective_reflection_coefficient"]
        )

        shadow_size_val = (
            params["shadow_size_factor"] *
            shadow_scale(
                inputs.object_size,
                inputs.object_distance,
                inputs.screen_distance
            )
        )

        shadow_intensity = reflected_intensity * (1 - inputs.ambient_light)

        t_values.append(t)
        inc_angles.append(angle_i)
        ref_angles.append(angle_r)
        reflected_intensity_series.append(reflected_intensity)
        shadow_size_series.append(shadow_size_val)
        shadow_intensity_series.append(shadow_intensity)

        t += sampling_interval

    return Timeline(
        t=t_values,
        angle_of_incidence=inc_angles,
        angle_of_reflection=ref_angles,
        reflected_intensity=reflected_intensity_series,
        shadow_size=shadow_size_series,
        shadow_intensity=shadow_intensity_series,
    )

def predict(
    model: tf.keras.Model,
    inputs: ExperimentInputs,
    input_keys: List[str],
    total_time: float,
    sampling_interval: float,
) -> SimulationResult:
    X = np.asarray(
        [[float(getattr(inputs, k)) for k in input_keys]],
        dtype=np.float32,
    )

    preds = model.predict(X, verbose=0)[0]

    params = {
        "effective_reflection_coefficient": clamp(preds[0], 0.0, 1.0),
        "shadow_size_factor": clamp(preds[1], 0.0, 10.0),
        "shadow_blur_factor": clamp(preds[2], 0.0, 1.0),
    }

    timeline = params_to_timeline(
        params=params,
        inputs=inputs,
        total_time=total_time,
        sampling_interval=sampling_interval,
    )

    final_state = FinalState(
        angle_of_reflection=timeline.angle_of_reflection[-1],
        image_distance=inputs.object_distance,
        shadow_size=timeline.shadow_size[-1],
        shadow_intensity=timeline.shadow_intensity[-1],
        reflected_intensity=timeline.reflected_intensity[-1],
    )

    visual = VisualData(
        rays={
            "incident_angle": inputs.angle_of_incidence,
            "reflected_angle": final_state.angle_of_reflection,
            "ray_count": 10,
            "scattering_factor": params["shadow_blur_factor"],
        },
        shadow={
            "size": final_state.shadow_size,
            "blur_radius": params["shadow_blur_factor"] * 30,
            "opacity": max(0.0, 1 - inputs.ambient_light),
        },
        mirror={
            "type": inputs.mirror_type
        },
        light={
            "wavelength": inputs.wavelength,
            "color_hint": "visible",
        },
        animation={
            "predicted": True
        }
    )

    return SimulationResult(
        final_state=final_state,
        timeline=timeline,
        visual=visual,
    )
