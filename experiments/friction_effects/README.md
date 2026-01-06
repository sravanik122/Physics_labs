# Friction and Its Effects

## Input Parameters

- **surface_roughness** – Roughness of the contacting surfaces  
- **normal_force** – Normal force between the surfaces (N)  
- **material_pair** – Pair of materials in contact (e.g., rubber–steel, wood–wood)  
- **lubrication_level** – Level of lubrication (none / low / high)  
- **speed_of_motion** – Relative sliding speed between surfaces (m/s)  
- **temperature** – Temperature at the contact surface (°C)  
- **contact_area** – Area of contact between surfaces (m²)  
- **wear_factor** – Wear tendency of the materials (low / medium / high)  
- **time** – Duration of motion (s)  
- **environmental_conditions** – Surrounding conditions (dry / humid / wet / dusty)  
- **motion_type** – Type of friction (static / kinetic)  
- **surface_contamination** – Contamination on the surface (clean / dusty / oily)  

---

## Output Parameters

### Final State
- **friction_force** – Force opposing motion due to friction (N)  
- **coefficient_of_friction** – Effective coefficient of friction  
- **heat_generated** – Heat produced due to friction (J)  
- **energy_loss** – Total energy lost due to friction (J)  

### Timeline
- **t** – Time values (s)  
- **friction_force** – Friction force variation over time (N)  
- **coefficient_of_friction** – Change in coefficient of friction over time  
- **heat_generated** – Heat generation over time (J)  
- **relative_motion** – Relative motion of the surfaces over time  

---

## Frontend Visualization

The frontend should visualize friction using arrows opposite to the direction of motion, where arrow thickness represents the magnitude of the friction force.  
Surface appearance can be modified to reflect roughness, lubrication, and contamination, while heat generation may be shown using a color glow that intensifies as friction increases.
Additionally, synchronized graphs of friction force and heat generation versus time can be displayed alongside the animation to help students understand how friction evolves under different conditions.
