# Reflection of Light and Shadow Formation

## Input Parameters

- **angle_of_incidence** – Angle between the incident ray and the normal (degrees)  
- **object_distance** – Distance of the object from the mirror (m)  
- **light_intensity** – Intensity of the incident light  
- **mirror_type** – Type of mirror (plane / concave / convex)  
- **surface_smoothness** – Smoothness of reflecting surface (0–1)  
- **object_size** – Size of the object casting a shadow (m)  
- **screen_distance** – Distance between object and screen (m)  
- **ambient_light** – Ambient light level affecting shadow visibility  
- **wavelength** – Wavelength of light determining color (nm)  
- **sampling_interval** – Time interval between observations (s)  
- **observation_duration** – Total duration of observation (s)  

---

## Output Parameters

### Final State
- **angle_of_reflection** – Angle of reflected ray (degrees)  
- **image_distance** – Distance of image from mirror (m)  
- **shadow_size** – Size of the shadow formed on the screen (m)  
- **shadow_intensity** – Intensity of the shadow  
- **reflected_intensity** – Intensity of the reflected light  

### Timeline
- **t** – Time values (s)  
- **angle_of_incidence** – Angle of incidence over time (degrees)  
- **angle_of_reflection** – Angle of reflection over time (degrees)  
- **reflected_intensity** – Reflected light intensity over time  
- **shadow_size** – Shadow size over time (m)  
- **shadow_intensity** – Shadow intensity over time  
