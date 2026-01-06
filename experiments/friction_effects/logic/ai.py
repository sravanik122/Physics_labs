import json
import math
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
    SimulationResult,
    TrainingRow,
    ModelPrediction
)
from .helpers import safe_divide, add_gaussian_noise

CONFIG_PATH = Path(__file__).parents[1] / "config" / "model_config.json"

with open(CONFIG_PATH, "r") as f:
    MODEL_CONFIG = json.load(f)

def build_model(input_dim: int, output_dim: int) -> tf.keras.Model:
    """
    Build lightweight regression model for friction parameters.
    """
    model = models.Sequential()
    model.add(layers.Input(shape=(input_dim,)))

    for units in MODEL_CONFIG["model_layers"]:
        model.add(layers.Dense(units, activation="relu"))

    model.add(layers.Dense(output_dim, activation="linear"))

    optimizer = optimizers.Adam(
        learning_rate=MODEL_CONFIG["learning_rate"]
    )

    model.compile(
        optimizer=optimizer,
        loss=MODEL_CONFIG["loss"],
        metrics=MODEL_CONFIG["metrics"]
    )

    return model

def train_model(
    training_data: List[TrainingRow],
    input_keys: List[str],
    output_keys: List[str]
) -> Dict:
    """
    Train ML model using teacher dataset.
    """

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
        verbose=0
    )

    model.save(Path(log_dir) / "model.keras")

    return {
        "epochs": MODEL_CONFIG["epochs"],
        "loss": history.history["loss"],
        "val_loss": history.history.get("val_loss", [])
    }

def load_model(model_path: Path) -> tf.keras.Model:
    """
    Load trained friction ML model.
    """
    return tf.keras.models.load_model(model_path)

def params_to_timeline(
    predicted_params: Dict[str, float],
    total_time: float,
    sampling_interval: float,
    normal_force: float,
    speed_of_motion: float,
    noise_sigma: float = 0.0
) -> Timeline:
    """
    Reconstruct friction behavior using A*(1 - exp(-k*t)).
    """

    t_values = np.arange(0, total_time + sampling_interval, sampling_interval)

    A_mu = predicted_params["A_mu"]
    k = predicted_params["k_rate"]

    mu_series = []
    friction_series = []
    heat_series = []
    motion_series = []

    for t in t_values:
        mu_t = A_mu * (1 - math.exp(-k * t))
        friction_force = mu_t * normal_force
        heat_generated = friction_force * speed_of_motion * sampling_interval

        mu_t = add_gaussian_noise(mu_t, noise_sigma)
        friction_force = add_gaussian_noise(friction_force, noise_sigma)
        heat_generated = add_gaussian_noise(heat_generated, noise_sigma)

        mu_series.append(mu_t)
        friction_series.append(friction_force)
        heat_series.append(heat_generated)
        motion_series.append(speed_of_motion)

    return Timeline(
        t=t_values.tolist(),
        friction_force=friction_series,
        coefficient_of_friction=mu_series,
        heat_generated=heat_series,
        relative_motion=motion_series
    )

def predict(
    model: tf.keras.Model,
    inputs: ExperimentInputs,
    input_keys: List[str]
) -> SimulationResult:
    """
    Predict friction behavior using trained ML model.
    """

    X = np.asarray(
        [[float(getattr(inputs, k)) for k in input_keys]],
        dtype=np.float32
    )

    preds = model.predict(X, verbose=0)[0]

    # Compact physical parameters
    predicted_params = {
        "A_mu": max(0.0, preds[0]),
        "k_rate": max(1e-4, preds[1])
    }

    timeline = params_to_timeline(
        predicted_params=predicted_params,
        total_time=inputs.time,
        sampling_interval=inputs.sampling_interval,
        normal_force=inputs.normal_force,
        speed_of_motion=inputs.speed_of_motion
    )

    final_state = FinalState(
        friction_force=timeline.friction_force[-1],
        coefficient_of_friction=timeline.coefficient_of_friction[-1],
        heat_generated=sum(timeline.heat_generated),
        energy_loss=sum(timeline.heat_generated)
    )

    visual = {
        "show_friction_arrows": True,
        "show_heat_map": True,
        "show_surface_texture_change": True,
        "predicted": True
    }

    return SimulationResult(
        final_state=final_state,
        timeline=timeline,
        visual=visual
    )
