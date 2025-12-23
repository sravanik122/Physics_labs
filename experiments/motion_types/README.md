# Types of Motion: Linear, Circular and Periodic

## Input Parameters

- **motion_type** – Type of motion (linear, circular, periodic)  
- **displacement** – Initial displacement (m)  
- **speed** – Linear speed of the object (m/s)  
- **radius_of_path** – Radius of circular path (m)  
- **angular_velocity** – Angular velocity (rad/s)  
- **amplitude** – Maximum displacement in periodic motion (m)  
- **frequency** – Frequency of oscillation (Hz)  
- **medium_resistance** – Resistance of the surrounding medium  
- **initial_position** – Initial position of the object (m)  
- **observation_duration** – Total duration of observation (s)  
- **sampling_interval** – Time interval between observations (s)  

---

## Output Parameters

### Final State
- **final_position** – Position of the object at the end of simulation  
- **final_velocity** – Velocity of the object at the end of simulation  
- **total_distance** – Total distance travelled (m)  
- **average_speed** – Average speed over the motion (m/s)  
- **motion_type** – Type of motion simulated  

### Timeline
- **t** – Time values (s)  
- **position** – Position of the object over time  
- **velocity** – Velocity of the object over time  
- **acceleration** – Acceleration of the object over time  
- **x, y** – Trajectory coordinates  
- **speed** – Speed of the object over time  

### Visual Outputs
- **trajectory** – Path followed by the object  
- **velocity_vectors** – Direction and magnitude of velocity  
- **path_type** – Linear, circular, or periodic motion  
