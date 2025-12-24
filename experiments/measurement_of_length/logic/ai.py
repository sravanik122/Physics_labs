import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

from .types import (
    TrainingRow,
    ModelPrediction,
    Timeline,
    ErrorContributionTimeline
)
from .helpers import normalize_input, denormalize_output, linear_interpolate

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "model_config.json"

with open(CONFIG_PATH, "r") as f:
    MODEL_CONFIG = json.load(f)

def build_model(input_dim: int, output_dim: int) -> tf.keras.Model:
    """
    Builds a lightweight regression model.
    Output predicts compact parameters [A, k].
    """
    model = models.Sequential()
    model.add(layers.Input(shape=(input_dim,)))

    for units in MODEL_CONFIG["model_layers"]:
        model.add(layers.Dense(units, activation="relu"))

    model.add(layers.Dense(output_dim, activation="linear"))

    optimizer_cfg = MODEL_CONFIG["optimizer"]
    optimizer = optimizers.Adam(
        learning_rate=MODEL_CONFIG["learning_rate"],
        beta_1=optimizer_cfg["beta_1"],
        beta_2=optimizer_cfg["beta_2"],
        epsilon=optimizer_cfg["epsilon"]
    )

    model.compile(
        optimizer=optimizer,
        loss=MODEL_CONFIG["loss"],
        metrics=MODEL_CONFIG["metrics"]
    )

    return model

def train_model(
    dataset: List[TrainingRow]
) -> Dict:
    """
    Trains the ML model using teacher-generated data.
    Returns training history as dict.
    """
    X = np.array([
        [
            row.object_length,
            row.least_count,
            row.zero_error,
            row.alignment_error,
            row.parallax_error,
            row.temperature,
            row.surface_roughness_factor,
            row.user_precision_factor
        ]
        for row in dataset
    ])

    y = np.array([
        [
            row.final_length - row.object_length,  
            1.2                                    
        ]
        for row in dataset
    ])

    model = build_model(input_dim=X.shape[1], output_dim=2)

    log_dir = MODEL_CONFIG["tensorboard_logdir"]
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    history = model.fit(
        X,
        y,
        epochs=MODEL_CONFIG["epochs"],
        batch_size=MODEL_CONFIG["batch_size"],
        validation_split=MODEL_CONFIG["validation_split"],
        callbacks=[tb_callback],
        verbose=0
    )

    model.save(Path(log_dir) / "model.keras")

    return {
        "loss": history.history.get("loss", []),
        "val_loss": history.history.get("val_loss", []),
        "metrics": history.history
    }

def load_model() -> tf.keras.Model:
    """
    Loads trained model from disk.
    """
    model_path = Path(MODEL_CONFIG["tensorboard_logdir"]) / "model.keras"
    return tf.keras.models.load_model(model_path)

def predict(
    model: tf.keras.Model,
    features: List[float]
) -> ModelPrediction:
    """
    Predicts compact parameters (A, k).
    """
    features_np = np.array([features])
    pred = model.predict(features_np, verbose=0)[0]

    return ModelPrediction(
        predicted_length=pred[0]
    )

def params_to_timeline(
    predicted_params: List[float],
    base_length: float,
    total_time: float,
    sampling_interval: float
) -> Timeline:
    """
    Converts predicted (A, k) into full measurement timeline.
    """
    A, k = predicted_params

    steps = int(total_time / sampling_interval) + 1
    t_values = [i * sampling_interval for i in range(steps)]

    observed = []
    corrected = []

    for t in t_values:
        offset = A * (1 - math.exp(-k * t))
        obs = base_length + offset
        corr = obs
        observed.append(obs)
        corrected.append(corr)

    error_series = linear_interpolate(0.0, A, steps)

    return Timeline(
        t=t_values,
        observed_length=observed,
        corrected_length=corrected,
        error_contribution=ErrorContributionTimeline(
            zero_error=error_series,
            alignment_error=error_series,
            parallax_error=error_series,
            thermal_error=error_series
        )
    )
