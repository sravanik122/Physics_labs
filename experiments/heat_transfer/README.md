# Heat Transfer: Conduction, Convection and Radiation

## Input Parameters

- **temperature_difference** – Temperature difference between hot and cold regions (°C)  
- **material_type** – Material used for heat conduction  
- **surface_area** – Surface area available for heat transfer (m²)  
- **distance_from_source** – Distance between heat source and receiving surface (m)  
- **medium_type** – Surrounding medium (air or water)  
- **airflow_speed** – Speed of air or fluid affecting convection (m/s)  
- **emissivity** – Emissivity of the surface for radiation heat transfer  
- **insulation_level** – Level of insulation reducing heat loss  
- **sampling_interval** – Time interval between observations (s)  
- **observation_duration** – Total duration of heat transfer observation (s)  

---

## Output Parameters

### Final State
- **final_temperature** – Temperature at the end of the simulation (°C)  
- **total_heat_transfer** – Total heat transferred (J)  
- **conduction_heat** – Heat transferred by conduction (J)  
- **convection_heat** – Heat transferred by convection (J)  
- **radiation_heat** – Heat transferred by radiation (J)  

### Timeline
- **t** – Time values (s)  
- **temperature** – Temperature over time (°C)  
- **heat_transfer_rate** – Total heat transfer rate over time (W)  
- **conduction_rate** – Heat transfer rate by conduction (W)  
- **convection_rate** – Heat transfer rate by convection (W)  
- **radiation_rate** – Heat transfer rate by radiation (W)  
