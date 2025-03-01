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
    """Static obstacle agent that doesn't move."""
    
    def __init__(self, time_step):
        super().__init__()
        self.visible = True
        # Random radius for variety - between 0.2 and 0.8
        self.radius = 0.2 + np.random.random() * 0.6
        self.time_step = time_step
        self.v_pref = 0  # Obstacles don't move
    
    def step(self, action=None):
        """Obstacles don't move, so override step to do nothing."""
        pass

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