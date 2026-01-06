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
    ModelPrediction,
)

from .helpers import (
    safe_divide,
    add_gaussian_noise,
)

CONFIG_PATH = Path(__file__).parents[1] / "config" / "model_config.json"

with open(CONFIG_PATH, "r") as f:
    MODEL_CONFIG = json.load(f)

def build_model(input_dim: int, output_dim: int) -> tf.keras.Model:
    """
    Build a lightweight regression model.
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
    output_keys: List[str],
) -> Dict:
    """
    Train ML model using simulation-generated dataset.
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
        verbose=0,
    )

    model.save(Path(log_dir) / "model.keras")

    return {
        "loss": history.history["loss"],
        "val_loss": history.history.get("val_loss", []),
        "epochs": MODEL_CONFIG["epochs"],
    }

def load_model(model_path: Path) -> tf.keras.Model:
    """
    Load trained ML model.
    """
    return tf.keras.models.load_model(model_path)

def params_to_timeline(
    params: Dict[str, float],
    total_time: float,
    sampling_interval: float,
    noise_sigma: float = 0.0
) -> Timeline:
    """
    Reconstruct pressure and deformation timeline
    using A*(1 - exp(-k*t)).
    """

    t_values = np.arange(0, total_time + sampling_interval, sampling_interval)

    pressure_series = []
    deformation_series = []

    A = params["A_pressure"]
    k = params["k_rate"]

    for t in t_values:
        pressure_t = A * (1 - math.exp(-k * t))
        deformation_t = safe_divide(pressure_t, 1e6)

        pressure_t = add_gaussian_noise(pressure_t, noise_sigma)
        deformation_t = add_gaussian_noise(deformation_t, noise_sigma)

        pressure_series.append(pressure_t)
        deformation_series.append(deformation_t)

    return Timeline(
        t=t_values.tolist(),
        applied_force=[None] * len(t_values),
        normal_force=[None] * len(t_values),
        pressure=pressure_series,
        deformation_depth=deformation_series,
    )

def predict(
    model: tf.keras.Model,
    inputs: ExperimentInputs,
    input_keys: List[str],
    total_time: float,
    sampling_interval: float,
) -> SimulationResult:
    """
    Predict pressure behavior using trained ML model.
    """

    X = np.asarray(
        [[float(getattr(inputs, k)) for k in input_keys]],
        dtype=np.float32,
    )

    preds = model.predict(X, verbose=0)[0]

    params = {
        "A_pressure": max(0.0, preds[0]),
        "k_rate": max(1e-4, preds[1]),
    }

    timeline = params_to_timeline(
        params=params,
        total_time=total_time,
        sampling_interval=sampling_interval,
    )

    final_state = FinalState(
        pressure=timeline.pressure[-1],
        normal_force=safe_divide(params["A_pressure"], 1.0),
        stress=timeline.pressure[-1],
        deformation_depth=timeline.deformation_depth[-1],
    )

    visual = {
        "show_force_arrows": True,
        "show_pressure_heatmap": True,
        "show_surface_deformation": True,
        "predicted": True
    }

    return SimulationResult(
        final_state=final_state,
        timeline=timeline,
        visual=visual,
    )
