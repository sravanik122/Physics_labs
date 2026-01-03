from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .logic.types import (
    ExperimentInputs,
    SimulationResult,
    TrainingRow,
)
from .logic import sim as sim_engine
from .logic import ai as ai_engine

app = FastAPI(
    title="Heat Transfer: Conduction, Convection and Radiation",
    version="1.0.0",
    description="Virtual lab backend for heat transfer experiments",
)

class TrainRequest(BaseModel):
    training_inputs: List[ExperimentInputs]
    input_keys: List[str]
    output_keys: List[str]

class PredictRequest(BaseModel):
    inputs: ExperimentInputs
    input_keys: List[str]
    total_time: float
    sampling_interval: float

@app.post("/simulate", response_model=SimulationResult)
def simulate(inputs: ExperimentInputs):
    """
    Run deterministic heat-transfer simulation.
    """
    return sim_engine.simulate(inputs)

@app.post("/train")
def train_model(request: TrainRequest):
    """
    Train ML model using simulation-generated data.
    """
    training_rows: List[TrainingRow] = (
        sim_engine.generate_teacher_dataset(request.training_inputs)
    )
    history = ai_engine.train_model(
        training_data=training_rows,
        input_keys=request.input_keys,
        output_keys=request.output_keys,
    )
    return {
        "status": "training_completed",
        "history": history,
    }

@app.post("/predict", response_model=SimulationResult)
def predict(request: PredictRequest):
    """
    Predict heat-transfer behavior using trained ML model.
    """

    model_path = (
        Path(ai_engine.MODEL_CONFIG["tensorboard_logdir"]) / "model.keras"
    )

    if not model_path.exists():
        raise HTTPException(
            status_code=400,
            detail="Model not trained yet. Please call /train first."
        )

    model = ai_engine.load_model(model_path)

    return ai_engine.predict(
        model=model,
        inputs=request.inputs,
        input_keys=request.input_keys,
        total_time=request.total_time,
        sampling_interval=request.sampling_interval,
    )
