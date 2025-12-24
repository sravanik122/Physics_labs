from pydantic import BaseModel
from typing import List, Dict, Optional

class ExperimentInputs(BaseModel):
    object_length: float
    measuring_tool_type: str
    least_count: float
    zero_error: float
    alignment_error: float
    parallax_error: float
    unit_system: str
    temperature: float
    surface_roughness: str
    user_precision: str

    sampling_interval: Optional[float] = None
    seed: Optional[int] = None

class FinalState(BaseModel):
    observed_length: float
    corrected_length: float
    total_measurement_error: float
    final_length: float
    measurement_accuracy: float

class ErrorContributionTimeline(BaseModel):
    zero_error: List[float]
    alignment_error: List[float]
    parallax_error: List[float]
    thermal_error: List[float]

class Timeline(BaseModel):
    t: List[float]
    observed_length: List[float]
    corrected_length: List[float]
    error_contribution: ErrorContributionTimeline

class SimulationResult(BaseModel):
    final_state: FinalState
    timeline: Timeline
    visual: Dict
  
class TrainingRow(BaseModel):
    object_length: float
    least_count: float
    zero_error: float
    alignment_error: float
    parallax_error: float
    temperature: float
    surface_roughness_factor: float
    user_precision_factor: float
    final_length: float

class ModelPrediction(BaseModel):
    predicted_length: float
