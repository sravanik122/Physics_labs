# Electric Circuits and Conductors

## Input Parameters

- **voltage** – Applied voltage from the power source (V)  
- **wire_material** – Material of the conducting wire (copper, aluminum, iron, nichrome, silver)  
- **wire_length** – Length of the wire (m)  
- **wire_thickness** – Cross-sectional area of the wire (mm²)  
- **temperature** – Temperature of the conductor (°C)  
- **switch_state** – Switch ON/OFF state  
- **power_source_type** – Type of power source (DC or AC)  
- **contact_resistance** – Resistance at wire joints and contacts (Ω)  
- **sampling_interval** – Time interval between observations (s)  
- **observation_duration** – Total duration of observation (s)  

---

## Output Parameters

### Final State
- **current** – Electric current in the circuit (A)  
- **voltage** – Voltage across the circuit (V)  
- **total_resistance** – Total resistance of the circuit (Ω)  
- **power** – Power dissipated in the circuit (W)  
- **switch_state** – Final switch state  
- **power_source_type** – Type of power source used  

### Timeline
- **t** – Time values (s)  
- **current** – Current over time (A)  
- **voltage** – Voltage over time (V)  
- **resistance** – Resistance over time (Ω)  
- **power** – Power over time (W)  
- **temperature** – Temperature over time (°C)  

### Visual Outputs
- **current_flow** – Indicator of current flow and intensity  
- **bulb_glow** – Glow intensity of the bulb  
- **wire_heating** – Heating level of the wire  
- **circuit_state** – Switch ON/OFF state for visualization  
