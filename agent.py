# agent.py

import numpy as np
from numpy.linalg import norm

class Agent:
    """Base class for all agents in the simulation."""
    
    def __init__(self):
        # Position
        self.px = None
        self.py = None
        
        # Goal position
        self.gx = None
        self.gy = None
        
        # Velocity
        self.vx = None
        self.vy = None
        
        # Other properties
        self.theta = None
        self.radius = None
        self.v_pref = None
        self.time_step = None
        self.visible = None

    def observable_state(self):
        """Return the observable state for other agents."""
        return np.array([self.px, self.py, self.vx, self.vy, self.radius])

    def full_state(self):
        """Return the full state of the agent."""
        return np.array([self.px, self.py, self.vx, self.vy, self.radius, 
                         self.gx, self.gy, self.v_pref, self.theta])

    def step(self, action):
        """Update agent position and velocity based on action."""
        self.px = self.px + action[0] * self.time_step
        self.py = self.py + action[1] * self.time_step
        self.vx = action[0]
        self.vy = action[1]

    def set(self, px, py, gx, gy, vx, vy, theta):
        """Set agent properties."""
        self.px = px
        self.py = py
        self.gx = gx
        self.gy = gy
        self.vx = vx
        self.vy = vy
        self.theta = theta

    def set_position(self, position):
        """Set position from a tuple."""
        self.px = position[0]
        self.py = position[1]

    def set_velocity(self, velocity):
        """Set velocity from a tuple."""
        self.vx = velocity[0]
        self.vy = velocity[1]

    def get_position(self):
        """Get current position as a tuple."""
        return self.px, self.py

    def get_goal_position(self):
        """Get goal position as a tuple."""
        return self.gx, self.gy

class Obstacle(Agent):
    """Dynamic obstacle agent that can move based on different movement patterns."""
    
    def __init__(self, time_step, velocity_scale=0.3):
        super().__init__()
        self.visible = True
        # Random radius for variety - between 0.2 and 0.8
        self.radius = 0.2 + np.random.random() * 0.6
        self.time_step = time_step
        
        # Velocity parameters - now obstacles can move
        self.v_pref = velocity_scale * (0.3 + np.random.random() * 0.4)  # Random speed between 0.3-0.7 × scale
        
        # Movement pattern parameters
        self.movement_type = np.random.choice(['linear', 'circular', 'random'])
        self.movement_timer = 0
        self.movement_period = 10.0  # Time before changing direction for random movement
        self.direction_angle = np.random.random() * 2 * np.pi  # Initial direction
        
        # For circular movement
        self.center_x = 0
        self.center_y = 0
        self.orbit_radius = 0
        self.angular_velocity = 0
        self.angle = 0
    
    def set_circular_params(self, center_x, center_y):
        """Setup parameters for circular movement."""
        self.movement_type = 'circular'
        self.center_x = center_x
        self.center_y = center_y
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
            
            # Boundary check (simple bounce)
            boundary = 5.0  # Boundary of the environment
            if abs(self.px + vx * self.time_step) > boundary or abs(self.py + vy * self.time_step) > boundary:
                # Reflect direction when hitting boundary
                self.direction_angle = np.random.random() * 2 * np.pi
                vx = self.v_pref * np.cos(self.direction_angle)
                vy = self.v_pref * np.sin(self.direction_angle)
                
        elif self.movement_type == 'circular':
            # Circular movement around a center point
            self.angle += self.angular_velocity
            # Calculate position on the circle
            target_x = self.center_x + self.orbit_radius * np.cos(self.angle)
            target_y = self.center_y + self.orbit_radius * np.sin(self.angle)
            # Calculate velocity to move toward the target position
            dx = target_x - self.px
            dy = target_y - self.py
            dist = np.sqrt(dx*dx + dy*dy)
            
            if dist > 0.001:  # Avoid division by zero
                vx = dx / dist * self.v_pref
                vy = dy / dist * self.v_pref
            else:
                vx, vy = 0, 0
                
        elif self.movement_type == 'random':
            # Random directional changes
            self.movement_timer += self.time_step
            if self.movement_timer >= self.movement_period:
                self.movement_timer = 0
                self.direction_angle = np.random.random() * 2 * np.pi
                
            vx = self.v_pref * np.cos(self.direction_angle)
            vy = self.v_pref * np.sin(self.direction_angle)
        
        return vx, vy
    
    def step(self, action=None):
        """Update obstacle position based on its velocity."""
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
    
    def __init__(self, time_step):
        super().__init__()
        self.visible = False
        self.radius = 0.3
        self.time_step = time_step
        self.v_pref = 1.0
        
        # Default start and goal positions
        self.px = 0
        self.py = -4
        self.gx = 0
        self.gy = 4
        self.vx = 0
        self.vy = 0
        self.theta = 0
        
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
            action: (vx, vy) action
        """
        # Call parent step method to update position and velocity
        super().step(action)
        
        # Update orientation (face direction of movement)
        if action[0] != 0 or action[1] != 0:
            self.theta = np.arctan2(action[1], action[0])