# agent.py

import numpy as np
from numpy.linalg import norm

class Agent:
    """Base class for all agents in the simulation."""
    
    def __init__(self, dimension='2D'):
        # Position
        self.px = None
        self.py = None
        self.pz = None if dimension == '3D' else 0  # Z-coordinate for 3D
        
        # Goal position
        self.gx = None
        self.gy = None
        self.gz = None if dimension == '3D' else 0  # Z-coordinate goal for 3D
        
        # Velocity
        self.vx = None
        self.vy = None
        self.vz = None if dimension == '3D' else 0  # Z-velocity for 3D
        
        # Other properties
        self.theta = None  # XY-plane rotation
        self.phi = None if dimension == '3D' else 0  # Elevation angle for 3D
        self.radius = None
        self.v_pref = None
        self.time_step = None
        self.visible = None
        self.dimension = dimension

    def observable_state(self):
        """Return the observable state for other agents."""
        if self.dimension == '3D':
            return np.array([self.px, self.py, self.pz, self.vx, self.vy, self.vz, self.radius])
        else:
            return np.array([self.px, self.py, self.vx, self.vy, self.radius])

    def full_state(self):
        """Return the full state of the agent."""
        if self.dimension == '3D':
            return np.array([self.px, self.py, self.pz, self.vx, self.vy, self.vz, self.radius, 
                           self.gx, self.gy, self.gz, self.v_pref, self.theta, self.phi])
        else:
            return np.array([self.px, self.py, self.vx, self.vy, self.radius, 
                           self.gx, self.gy, self.v_pref, self.theta])

    def step(self, action):
        """Update agent position and velocity based on action."""
        self.px = self.px + action[0] * self.time_step
        self.py = self.py + action[1] * self.time_step
        if self.dimension == '3D' and len(action) > 2:
            self.pz = self.pz + action[2] * self.time_step
            self.vz = action[2]
        self.vx = action[0]
        self.vy = action[1]

    def set(self, px, py, gx, gy, vx, vy, theta, pz=0, gz=0, vz=0, phi=0):
        """Set agent properties."""
        self.px = px
        self.py = py
        self.pz = pz if self.dimension == '3D' else 0
        self.gx = gx
        self.gy = gy
        self.gz = gz if self.dimension == '3D' else 0
        self.vx = vx
        self.vy = vy
        self.vz = vz if self.dimension == '3D' else 0
        self.theta = theta
        self.phi = phi if self.dimension == '3D' else 0

    def set_position(self, position):
        """Set position from a tuple."""
        self.px = position[0]
        self.py = position[1]
        if self.dimension == '3D' and len(position) > 2:
            self.pz = position[2]

    def set_velocity(self, velocity):
        """Set velocity from a tuple."""
        self.vx = velocity[0]
        self.vy = velocity[1]
        if self.dimension == '3D' and len(velocity) > 2:
            self.vz = velocity[2]

    def get_position(self):
        """Get current position as a tuple."""
        if self.dimension == '3D':
            return self.px, self.py, self.pz
        else:
            return self.px, self.py

    def get_goal_position(self):
        """Get goal position as a tuple."""
        if self.dimension == '3D':
            return self.gx, self.gy, self.gz
        else:
            return self.gx, self.gy

class Obstacle(Agent):
    """Dynamic obstacle agent that can move based on different movement patterns."""
    
    def __init__(self, time_step, velocity_scale=0.3, dimension='2D', z_velocity_scale=0.2):
        super().__init__(dimension)
        self.visible = True
        # Random radius for variety - between 0.2 and 0.8
        self.radius = 0.2 + np.random.random() * 0.6
        self.time_step = time_step
        
        # Velocity parameters - now obstacles can move
        self.v_pref = velocity_scale * (0.3 + np.random.random() * 0.4)  # Random speed between 0.3-0.7 × scale
        self.z_velocity_scale = z_velocity_scale
        
        # Movement pattern parameters
        if dimension == '3D':
            self.movement_type = np.random.choice(['linear', 'circular', 'helical', 'random'])
        else:
            self.movement_type = np.random.choice(['linear', 'circular', 'random'])
            
        self.movement_timer = 0
        self.movement_period = 10.0  # Time before changing direction for random movement
        self.direction_angle = np.random.random() * 2 * np.pi  # Initial direction
        
        if dimension == '3D':
            self.elevation_angle = (np.random.random() - 0.5) * np.pi  # Elevation angle for 3D
        
        # For circular/helical movement
        self.center_x = 0
        self.center_y = 0
        self.center_z = 0 if dimension == '3D' else 0
        self.orbit_radius = 0
        self.angular_velocity = 0
        self.angle = 0
        self.vertical_speed = 0  # For helical movement
    
    def set_circular_params(self, center_x, center_y, center_z=0):
        """Setup parameters for circular movement."""
        self.movement_type = 'circular' if self.dimension == '2D' else 'helical'
        self.center_x = center_x
        self.center_y = center_y
        if self.dimension == '3D':
            self.center_z = center_z
            # Random vertical speed for helical movement
            self.vertical_speed = self.z_velocity_scale * (np.random.random() * 0.4 - 0.2)
        
        # Calculate orbit radius based on current position
        dx = self.px - center_x
        dy = self.py - center_y
        self.orbit_radius = np.sqrt(dx*dx + dy*dy)
        # Set initial angle
        self.angle = np.arctan2(dy, dx)
        # Random angular velocity (radians per time step)
        self.angular_velocity = (0.02 + np.random.random() * 0.03) * (1 if np.random.random() > 0.5 else -1)
    
    def calculate_velocity(self):
        """Calculate velocity based on movement pattern."""
        if self.movement_type == 'linear':
            # Linear movement with occasional bouncing off boundaries
            vx = self.v_pref * np.cos(self.direction_angle)
            vy = self.v_pref * np.sin(self.direction_angle)
            vz = 0
            
            if self.dimension == '3D':
                # Add z-component for 3D
                elevation_factor = np.sin(self.elevation_angle)
                xy_factor = np.cos(self.elevation_angle)
                vx = self.v_pref * xy_factor * np.cos(self.direction_angle)
                vy = self.v_pref * xy_factor * np.sin(self.direction_angle)
                vz = self.v_pref * elevation_factor * self.z_velocity_scale
            
            # Boundary check (simple bounce)
            boundary = 5.0  # Boundary of the environment
            if abs(self.px + vx * self.time_step) > boundary or abs(self.py + vy * self.time_step) > boundary:
                # Reflect direction when hitting boundary
                self.direction_angle = np.random.random() * 2 * np.pi
                if self.dimension == '3D':
                    self.elevation_angle = (np.random.random() - 0.5) * np.pi
                
                # Recalculate velocity after bounce
                if self.dimension == '3D':
                    elevation_factor = np.sin(self.elevation_angle)
                    xy_factor = np.cos(self.elevation_angle)
                    vx = self.v_pref * xy_factor * np.cos(self.direction_angle)
                    vy = self.v_pref * xy_factor * np.sin(self.direction_angle)
                    vz = self.v_pref * elevation_factor * self.z_velocity_scale
                else:
                    vx = self.v_pref * np.cos(self.direction_angle)
                    vy = self.v_pref * np.sin(self.direction_angle)
            
            # Additional z-boundary check for 3D
            if self.dimension == '3D':
                height_limit = 5.0  # Z-boundary
                if abs(self.pz + vz * self.time_step) > height_limit:
                    # Reflect elevation when hitting z-boundary
                    self.elevation_angle = -self.elevation_angle
                    elevation_factor = np.sin(self.elevation_angle)
                    vz = self.v_pref * elevation_factor * self.z_velocity_scale
                
        elif self.movement_type == 'circular' or self.movement_type == 'helical':
            # Circular/helical movement
            self.angle += self.angular_velocity
            # Calculate position on the circle
            target_x = self.center_x + self.orbit_radius * np.cos(self.angle)
            target_y = self.center_y + self.orbit_radius * np.sin(self.angle)
            
            # For helical movement, add z-component
            if self.dimension == '3D' and self.movement_type == 'helical':
                # Simply move up or down, with bounce at height limits
                target_z = self.pz + self.vertical_speed
                height_limit = 5.0
                if abs(target_z) > height_limit:
                    self.vertical_speed = -self.vertical_speed
                    target_z = self.pz + self.vertical_speed
            else:
                target_z = 0
            
            # Calculate velocity to move toward the target position
            dx = target_x - self.px
            dy = target_y - self.py
            dz = target_z - self.pz if self.dimension == '3D' else 0
            
            if self.dimension == '3D':
                dist = np.sqrt(dx*dx + dy*dy + dz*dz)
                
                if dist > 0.001:  # Avoid division by zero
                    vx = dx / dist * self.v_pref
                    vy = dy / dist * self.v_pref
                    vz = dz / dist * self.v_pref * self.z_velocity_scale
                else:
                    vx, vy, vz = 0, 0, 0
            else:
                dist = np.sqrt(dx*dx + dy*dy)
                
                if dist > 0.001:  # Avoid division by zero
                    vx = dx / dist * self.v_pref
                    vy = dy / dist * self.v_pref
                else:
                    vx, vy = 0, 0
                vz = 0
                
        elif self.movement_type == 'random':
            # Random directional changes
            self.movement_timer += self.time_step
            if self.movement_timer >= self.movement_period:
                self.movement_timer = 0
                self.direction_angle = np.random.random() * 2 * np.pi
                if self.dimension == '3D':
                    self.elevation_angle = (np.random.random() - 0.5) * np.pi
                
            if self.dimension == '3D':
                elevation_factor = np.sin(self.elevation_angle)
                xy_factor = np.cos(self.elevation_angle)
                vx = self.v_pref * xy_factor * np.cos(self.direction_angle)
                vy = self.v_pref * xy_factor * np.sin(self.direction_angle)
                vz = self.v_pref * elevation_factor * self.z_velocity_scale
            else:
                vx = self.v_pref * np.cos(self.direction_angle)
                vy = self.v_pref * np.sin(self.direction_angle)
                vz = 0
        
        if self.dimension == '3D':
            return vx, vy, vz
        else:
            return vx, vy
    
    def step(self, action=None):
        """Update obstacle position based on its velocity."""
        if self.dimension == '3D':
            vx, vy, vz = self.calculate_velocity()
            
            # Update position and velocity
            self.px += vx * self.time_step
            self.py += vy * self.time_step
            self.pz += vz * self.time_step
            self.vx = vx
            self.vy = vy
            self.vz = vz
            
            # Update orientation based on movement direction
            if vx != 0 or vy != 0:
                self.theta = np.arctan2(vy, vx)
            if vx != 0 or vy != 0 or vz != 0:
                self.phi = np.arctan2(vz, np.sqrt(vx*vx + vy*vy))
        else:
            vx, vy = self.calculate_velocity()
            
            # Update position and velocity
            self.px += vx * self.time_step
            self.py += vy * self.time_step
            self.vx = vx
            self.vy = vy
            
            # Update orientation based on movement direction
            if vx != 0 or vy != 0:
                self.theta = np.arctan2(vy, vx)

class Robot(Agent):
    """Robot agent controlled by the learning algorithm."""
    
    def __init__(self, time_step, dimension='2D'):
        super().__init__(dimension)
        self.visible = False
        self.radius = 0.3
        self.time_step = time_step
        self.v_pref = 1.0
        
        # Default start and goal positions
        self.px = 0
        self.py = -4
        self.pz = 0 if dimension == '3D' else 0
        self.gx = 0
        self.gy = 4
        self.gz = 0 if dimension == '3D' else 0
        self.vx = 0
        self.vy = 0
        self.vz = 0 if dimension == '3D' else 0
        self.theta = 0
        self.phi = 0 if dimension == '3D' else 0
        
        # Policy for decision making (can be set later)
        self.policy = None
        
    def act(self, ob):
        """
        Determine action based on observation.
        
        Args:
            ob: Observation
            
        Returns:
            Action
        """
        if self.policy is None:
            # Default navigation policy if no policy is set
            state = ob[0]
            if self.dimension == '3D':
                goal_px = state[7]  # Index shifts for 3D
                goal_py = state[8]
                goal_pz = state[9]
                
                # Get vector to goal
                vector_to_goal = np.array([goal_px - self.px, goal_py - self.py, goal_pz - self.pz])
            else:
                goal_px = state[5]
                goal_py = state[6]
                
                # Get vector to goal
                vector_to_goal = np.array([goal_px - self.px, goal_py - self.py])
            
            # Normalize
            if norm(vector_to_goal) > 1e-5:
                vector_to_goal = vector_to_goal / norm(vector_to_goal)
                
            # Scale to preferred velocity
            action = self.v_pref * vector_to_goal
            
            # Safety check for speed limit
            speed = norm(action)
            if speed > self.v_pref:
                action = action / speed * self.v_pref
        else:
            # Use learned policy if available
            action = self.policy.predict(ob)
            
        return action
    
    def step(self, action):
        """
        Step forward with action.
        
        Args:
            action: velocity action (vx, vy) in 2D or (vx, vy, vz) in 3D
        """
        # Call parent step method to update position and velocity
        super().step(action)
        
        # Update orientation (face direction of movement)
        if self.dimension == '3D':
            if action[0] != 0 or action[1] != 0:
                self.theta = np.arctan2(action[1], action[0])
            if action[0] != 0 or action[1] != 0 or action[2] != 0:
                self.phi = np.arctan2(action[2], np.sqrt(action[0]*action[0] + action[1]*action[1]))
        else:
            if action[0] != 0 or action[1] != 0:
                self.theta = np.arctan2(action[1], action[0])