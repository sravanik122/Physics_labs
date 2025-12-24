from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from .logic.types import (
    ExperimentInputs,
    SimulationResult,
    FinalState,
    TrainingRow,
    ModelPrediction
)
from .logic import sim, ai


app = FastAPI(
    title="Measurement of Length using Different Tools",
    version="1.0.0",
    description="Thin HTTP wrapper for measurement_of_length experiment"
)

class SimulateRequest(BaseModel):
    inputs: ExperimentInputs

class SimulateAtTimeRequest(BaseModel):
    inputs: ExperimentInputs
    t: float

class TrainRequest(BaseModel):
    samples: int = 500

class PredictRequest(BaseModel):
    features: List[float]

@app.post("/simulate", response_model=SimulationResult)
def simulate_endpoint(request: SimulateRequest):
    """
    Run full deterministic simulation.
    """
    return sim.simulate(request.inputs)

@app.post("/simulate_at_time", response_model=FinalState)
def simulate_at_time_endpoint(request: SimulateAtTimeRequest):
    """
    Get simulation state at a specific time.
    """
    return sim.simulate_at_time(request.inputs, request.t)

@app.post("/train")
def train_endpoint(request: TrainRequest):
    """
    Train ML model using teacher-generated dataset.
    """
    dataset = sim.generate_teacher_dataset(samples=request.samples)
    history = ai.train_model(dataset)
    return {"status": "training_completed", "history": history}

@app.post("/predict", response_model=ModelPrediction)
def predict_endpoint(request: PredictRequest):
    """
    Predict measurement parameters using ML model.
    """
    model = ai.load_model()
    return ai.predict(model, request.features)

@app.get("/health")
def health_check():
    return {"status": "ok"}
