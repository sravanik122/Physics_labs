from pydantic import BaseModel
from typing import List, Dict, Optional, Union

class ExperimentInputs(BaseModel):
    temperature_difference: float
    material_type: str
    surface_area: float
    distance_from_source: float

    medium_type: str
    airflow_speed: float

    emissivity: float
    insulation_level: str

    observation_duration: float
    sampling_interval: float

    seed: Optional[int] = None

class FinalState(BaseModel):
    final_temperature: float
    total_heat_transfer: float

    conduction_heat: float
    convection_heat: float
    radiation_heat: float

class Timeline(BaseModel):
    t: List[float]

    temperature: List[float]
    heat_transfer_rate: List[float]

    conduction_rate: List[float]
    convection_rate: List[float]
    radiation_rate: List[float]

class VisualData(BaseModel):
    conduction: Dict[str, Union[str, float]]
    convection: Dict[str, Union[int, float]]
    radiation: Dict[str, Union[int, float]]
    animation: Dict[str, Union[bool, int]]

class SimulationResult(BaseModel):
    final_state: FinalState
    timeline: Timeline
    visual: VisualData

class TrainingRow(BaseModel):
    inputs: Dict[str, Union[float, int, str]]
    outputs: Dict[str, float]

class ModelPrediction(BaseModel):
    parameters: Dict[str, float]
    confidence: Optional[float] = None
