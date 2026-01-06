# Sound: Vibration, Pitch and Loudness

## Input Parameters

- **frequency (Hz)** – Frequency of vibration determining the pitch of sound  
- **amplitude** – Amplitude of vibration determining the loudness  
- **medium_type** – Medium of sound propagation (air / water / solid)  
- **tension (N)** – Tension in the vibrating body  
- **length_of_vibrating_body (m)** – Length of the vibrating source  
- **density (kg/m³)** – Density of the propagation medium  
- **distance_from_source (m)** – Distance between sound source and observer  
- **air_pressure (Pa)** – Air pressure affecting sound propagation  
- **damping** – Energy loss factor reducing vibration over time  
- **resonance** – Indicates whether resonance condition is present  
- **source_type** – Type of sound source (string / membrane / tuning_fork)  
- **observation_time (s)** – Duration of sound observation  

---

## Output Parameters

### Final State
- **sound_intensity** – Intensity of sound at the observer location  
- **loudness_level** – Perceived loudness of sound (dB)  
- **wavelength** – Wavelength of the sound wave  
- **energy_decay** – Remaining vibrational energy after damping  

### Timeline
- **t** – Time values  
- **displacement** – Vibration displacement over time  
- **sound_intensity** – Intensity variation over time  
- **loudness_level** – Loudness variation over time  
- **wave_amplitude** – Decaying wave amplitude over time  

---

## Visualization Guidance (Frontend)

The frontend should animate the vibrating sound source and display sound waves as expanding circular wavefronts.  
Wave spacing represents pitch (frequency), while wave height or brightness represents loudness (amplitude). Sound waves should fade with distance and damping, and resonance can be highlighted through amplified vibrations.
Graphs of sound intensity and loudness versus time can be shown alongside the animation to help students understand sound behavior visually.
