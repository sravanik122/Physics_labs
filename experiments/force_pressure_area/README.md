# Force, Pressure and Area Relationship

## Input Parameters

- **applied_force** – Force applied on the surface (N)
- **contact_area** – Area over which the force is applied (m²)
- **object_mass** – Mass of the object applying the force (kg)
- **surface_type** – Type of surface (soft / hard / elastic)
- **gravity** – Gravitational acceleration (m/s²)
- **deformation** – Initial deformation of the surface (m)
- **material_hardness** – Hardness of the contact material (low / medium / high)
- **pressure_distribution** – Pressure distribution pattern (uniform / non-uniform)
- **duration** – Time duration for which force is applied (s)
- **external_resistance** – Resistance opposing the applied force (low / medium / high)
- **contact_shape** – Shape of the contact area (flat / circular / edge / pointed)
- **surface_roughness** – Roughness of the surface (0–1)
- **force_application_angle** – Angle at which force is applied (degrees)

---

## Output Parameters

### Final State
- **pressure** – Pressure exerted on the surface (Pa)
- **normal_force** – Effective normal component of the applied force (N)
- **stress** – Stress experienced by the surface (Pa)
- **deformation_depth** – Final depth of surface deformation (m)

### Timeline
- **t** – Time values (s)
- **applied_force** – Applied force variation over time (N)
- **normal_force** – Normal force variation over time (N)
- **pressure** – Pressure variation over time (Pa)
- **deformation_depth** – Surface deformation variation over time (m)

---

## Frontend Visualization 

The frontend should visualize the applied force using arrows, where arrow length represents force magnitude and arrow orientation represents the angle of application.  
Pressure should be displayed as a color heatmap on the contact surface, transitioning from cooler colors (low pressure) to warmer colors (high pressure) to clearly show pressure distribution.
Surface deformation should be animated as a gradual indentation proportional to pressure, allowing students to observe how material hardness, contact area, and force direction influence deformation.  
Synchronized graphs of pressure and deformation versus time can be displayed alongside the animation to help students connect visual changes with numerical trends.
