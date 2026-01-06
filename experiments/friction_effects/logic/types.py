from typing import List, Dict, Optional
from pydantic import BaseModel

class ExperimentInputs(BaseModel):
    surface_roughness: float
    normal_force: float
    material_pair: str
    lubrication_level: str
    speed_of_motion: float
    temperature: float
    contact_area: float
    wear_factor: str
    time: float
    environmental_conditions: str
    motion_type: str
    surface_contamination: str

    sampling_interval: float
    seed: Optional[int] = None

class FinalState(BaseModel):
    friction_force: float
    coefficient_of_friction: float
    heat_generated: float
    energy_loss: float

class Timeline(BaseModel):
    t: List[float]
    friction_force: List[float]
    coefficient_of_friction: List[float]
    heat_generated: List[float]
    relative_motion: List[float]

class SimulationResult(BaseModel):
    final_state: FinalState
    timeline: Timeline
    visual: Dict

class TrainingRow(BaseModel):
    inputs: Dict[str, float]
    outputs: Dict[str, float]

class ModelPrediction(BaseModel):
    predicted_params: Dict[str, float]
