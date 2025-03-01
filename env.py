import rvo2
import torch
import itertools
import numpy as np
from numpy.linalg import norm
from agent import Robot, Obstacle
from matplotlib import animation
import matplotlib.pyplot as plt
plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

class ObstacleEnv:
    """
    Environment for robot navigation around static obstacles.
    Uses RVO2 library for collision avoidance simulation.
    """
    def __init__(self):
        # Environment parameters
        self.obstacle_list = []
        self.circle_radius = 4.0
        self.square_width = 10.0
        self.discomfort_dist = 0.2
        self.sim_time = 0.0
        self.time_out_duration = 25.0
        
        # Action space parameters
        self.speed_samples = 5
        self.rotation_samples = 16
        
        # RVO2 simulation parameters
        self.safety_space = 0.1  # Small safety buffer for obstacles
        self.neighbor_dist = 10.0
        self.max_neighbors = 10
        self.time_horizon = 5.0
        self.time_horizon_obst = 5.0
        self.time_step = 0.25
        self.radius = 0.3
        self.max_speed = 1.0
        
        # Initialize RVO2 simulator
        params = (self.neighbor_dist, self.max_neighbors, 
                  self.time_horizon, self.time_horizon_obst)
        self.sim = rvo2.PyRVOSimulator(self.time_step, *params, 
                                       self.radius, self.max_speed)
        
        # Pre-compute velocity samples for action space
        self.vel_samples = self.action_space()

    def generate_obstacle_positions(self, obstacle_num, layout):
        """Generate obstacle positions based on specified layout pattern."""
        if layout == "square":
            self._square_layout(obstacle_num)
        elif layout == "circle":
            self._circle_layout(obstacle_num)
        else:
            self._random_layout(obstacle_num)

    def _circle_layout(self, obstacle_num):
        """Position obstacles in a circular pattern around the origin."""
        while len(self.obstacle_list) < obstacle_num:
            obstacle = Obstacle(self.time_step)
            angle = np.random.random() * np.pi * 2
            px_noise = (np.random.random() - 0.5) * 0.5
            py_noise = (np.random.random() - 0.5) * 0.5
            px = self.circle_radius * np.cos(angle) + px_noise
            py = self.circle_radius * np.sin(angle) + py_noise

            if self._is_valid_position(px, py, obstacle):
                # Static obstacle (zero velocity and no goal)
                obstacle.set(px, py, 0, 0, 0, 0, 0)
                self.obstacle_list.append(obstacle)

    def _square_layout(self, obstacle_num):
        """Position obstacles in a square pattern."""
        while len(self.obstacle_list) < obstacle_num:
            obstacle = Obstacle(self.time_step)
            px = (np.random.random() - 0.5) * self.square_width
            py = (np.random.random() - 0.5) * self.square_width
            
            if self._is_valid_position(px, py, obstacle):
                # Static obstacle (zero velocity and no goal)
                obstacle.set(px, py, 0, 0, 0, 0, 0)
                self.obstacle_list.append(obstacle)
    
    def _random_layout(self, obstacle_num):
        """Position obstacles randomly in the environment."""
        while len(self.obstacle_list) < obstacle_num:
            obstacle = Obstacle(self.time_step)
            px = (np.random.random() - 0.5) * self.square_width * 1.5
            py = (np.random.random() - 0.5) * self.square_width * 1.5
            
            if self._is_valid_position(px, py, obstacle):
                # Static obstacle (zero velocity and no goal)
                obstacle.set(px, py, 0, 0, 0, 0, 0)
                self.obstacle_list.append(obstacle)
    
    def _is_valid_position(self, px, py, obstacle):
        """Check if position is valid (not colliding with other entities)."""
        # Get all entities to check against
        entities = [self.robot] + self.obstacle_list if hasattr(self, 'robot') else self.obstacle_list
        
        # Check for collisions with existing entities
        for entity in entities:
            min_dist = obstacle.radius + entity.radius + self.discomfort_dist
            if norm((px - entity.px, py - entity.py)) < min_dist:
                return False
        
        # Check if too close to robot's goal (if robot exists)
        if hasattr(self, 'robot'):
            if norm((px - self.robot.gx, py - self.robot.gy)) < min_dist:
                return False
                
        return True

    def action_space(self):
        """Generate discretized velocity actions for the robot."""
        # Generate speeds with exponential scaling for better control
        speeds = [(np.exp((i + 1) / self.speed_samples) - 1) / (np.e - 1) * self.max_speed 
                  for i in range(self.speed_samples)]
                  
        # Generate rotations (directions) uniformly around the circle
        rotations = np.linspace(0, 2 * np.pi, self.rotation_samples, endpoint=False)
        
        # Create all combinations of speed and direction
        action_space = []
        for rotation, speed in itertools.product(rotations, speeds):
            action_space.append([speed * np.cos(rotation), speed * np.sin(rotation)])
            
        return action_space

    def reset(self, obstacle_num, layout="circle", test_phase=False, counter=None):
        """
        Reset the environment with new obstacles.
        
        Args:
            obstacle_num: Number of obstacles to generate
            layout: Layout pattern for obstacles ("circle", "square", or "random")
            test_phase: Flag for reproducible testing
            counter: Seed value for reproducible testing
            
        Returns:
            Initial observation state
        """
        # Initialize robot agent
        self.robot = Robot(self.time_step)
        self.obstacle_list = []
        
        # Set seed for reproducibility if in test phase
        if test_phase and counter is not None:
            np.random.seed(counter)
            
        # Generate obstacles according to specified layout
        self.generate_obstacle_positions(obstacle_num=obstacle_num, layout=layout)
        
        # Create initial observation
        obs = [self.robot.full_state()] + [obstacle.observable_state() for obstacle in self.obstacle_list]
        
        # Initialize RVO2 simulation with obstacles
        params = (self.neighbor_dist, self.max_neighbors, 
                  self.time_horizon, self.time_horizon_obst)
                  
        for obstacle in self.obstacle_list:
            self.sim.addAgent(
                (obstacle.px, obstacle.py), 
                *params, 
                obstacle.radius + 0.01 + self.safety_space,
                0.0,  # Zero preferred velocity for static obstacles
                (0.0, 0.0)  # Zero initial velocity
            )
            
        # Reset simulation time and distance to goal
        self.sim_time = 0.0
        self.dg = norm(np.array(self.robot.get_position()) - np.array(self.robot.get_goal_position()))
        
        return obs

    def step(self, action):
        """
        Take a step in the environment with the given action.
        
        Args:
            action: Robot velocity action
            
        Returns:
            obs: New observation
            reward: Reward for the action
            done: Whether episode is done
            info: Additional information
        """
        # Static obstacles have zero preferred velocity
        for i in range(len(self.obstacle_list)):
            self.sim.setAgentPrefVelocity(i, (0.0, 0.0))

        # Advance simulation
        self.sim.doStep()
        
        # Update obstacle positions
        for i, obstacle in enumerate(self.obstacle_list):
            obstacle.set_position(self.sim.getAgentPosition(i))
            obstacle.set_velocity(self.sim.getAgentVelocity(i))
            
        # Update robot with action
        self.robot.step(action)
        
        # Update simulation time
        self.sim_time += self.time_step
        
        # Compute reward and check terminal conditions
        distance_list = []
        for obstacle in self.obstacle_list:
            distance = norm(np.array(obstacle.get_position()) - np.array(self.robot.get_position())) - obstacle.radius - self.robot.radius
            distance_list.append(distance)
            
        d_min = min(distance_list) if distance_list else float('inf')
        current_dg = norm(np.array(self.robot.get_position()) - np.array(self.robot.get_goal_position()))
        reaching_goal = current_dg < self.robot.radius

        # Calculate progress toward goal
        delta_d = self.dg - current_dg
        self.dg = current_dg

        # Determine reward and completion status
        if self.sim_time >= self.time_out_duration:
            reward = 0
            done = True
            info = "timeout"
        elif d_min < 0:
            reward = -10
            done = True
            info = "collision"
        elif d_min < self.discomfort_dist:
            reward = 10 * (d_min - self.discomfort_dist)
            done = False
            info = "close"
        elif reaching_goal:
            reward = 10
            done = True
            info = f"Goal reached, time {self.sim_time:.2f}"
        else:
            reward = delta_d
            done = False
            info = "Moving"

        # Create new observation
        obs = [self.robot.full_state()] + [obstacle.observable_state() for obstacle in self.obstacle_list]
        
        return obs, reward, done, info
    
    def convert_coord(self, obs):
        """
        Convert observation to robot-centric coordinate frame.
        
        Args:
            obs: Raw observation
            
        Returns:
            Tensor of transformed observations
        """
        # Extract robot and obstacle states
        robot_state = torch.tensor(obs[0], dtype=torch.float32)
        obstacle_states = torch.tensor(np.array(obs[1:]), dtype=torch.float32)
        
        # Check obstacle state dimensions
        num_obstacles = obstacle_states.shape[0]
        
        # Calculate goal direction
        dx = robot_state[5] - robot_state[0]
        dy = robot_state[6] - robot_state[1]
        dg = torch.tensor(norm((dx.item(), dy.item())), dtype=torch.float32).expand(num_obstacles, 1)
        
        # Calculate rotation angle (angle to goal)
        rot = torch.atan2(dy, dx)
        rot_expand = rot.expand(num_obstacles, 1)
        
        # Robot properties in transformed frame
        v_pref = robot_state[7].expand(num_obstacles, 1)
        vx = (robot_state[2] * torch.cos(rot) + robot_state[3] * torch.sin(rot)).expand(num_obstacles, 1)
        vy = (robot_state[3] * torch.cos(rot) - robot_state[2] * torch.sin(rot)).expand(num_obstacles, 1)
        radius = robot_state[4].expand(num_obstacles, 1)
        
        # Obstacle properties in transformed frame
        vx_obs = (obstacle_states[:, 2] * torch.cos(rot) + obstacle_states[:, 3] * torch.sin(rot)).unsqueeze(1)
        vy_obs = (obstacle_states[:, 3] * torch.cos(rot) - obstacle_states[:, 2] * torch.sin(rot)).unsqueeze(1)
        px_obs = ((obstacle_states[:, 0] - robot_state[0]) * torch.cos(rot) + 
                 (obstacle_states[:, 1] - robot_state[1]) * torch.sin(rot)).unsqueeze(1)
        py_obs = ((obstacle_states[:, 1] - robot_state[1]) * torch.cos(rot) - 
                 (obstacle_states[:, 0] - robot_state[0]) * torch.sin(rot)).unsqueeze(1)
        radius_obs = obstacle_states[:, 4].unsqueeze(1)
        
        # Calculate sum of radii (for collision detection)
        radius_sum = radius + radius_obs
        
        # Calculate distances between robot and obstacles
        distances = []
        for i in range(obstacle_states.shape[0]):
            dist = norm((obstacle_states[i, 0].item() - robot_state[0].item(), 
                         obstacle_states[i, 1].item() - robot_state[1].item()))
            distances.append(dist)
        
        da = torch.tensor(distances, dtype=torch.float32).unsqueeze(1)
        
        # Concatenate all features into a single state tensor
        new_state = torch.cat([
            dg, rot_expand, vx, vy, v_pref, radius, 
            px_obs, py_obs, vx_obs, vy_obs, 
            radius_obs, da, radius_sum
        ], dim=1).unsqueeze(0)  # Add batch dimension
        
        return new_state
    
    def render(self):
        """Render the environment with matplotlib animation."""
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        
        def init():
            """Initialize animation with obstacles and robot."""
            # Draw obstacles
            for obstacle in self.obstacle_list:
                obstacle_circle = plt.Circle(
                    obstacle.get_position(), 
                    obstacle.radius, 
                    fill=True, 
                    color='gray'
                )
                ax.add_artist(obstacle_circle)
                
            # Draw robot
            robot_circle = plt.Circle(
                self.robot.get_position(), 
                self.robot.radius, 
                fill=True, 
                color='r'
            )
            ax.add_artist(robot_circle)
            
            # Draw goal
            goal_circle = plt.Circle(
                self.robot.get_goal_position(), 
                0.2, 
                fill=True, 
                color='g'
            )
            ax.add_artist(goal_circle)
            
            return ax

        def update(i):
            """Update animation frame."""
            ax.clear()
            ax.set_xlim(-6, 6)
            ax.set_ylim(-6, 6)
            
            # Redraw obstacles
            for obstacle in self.obstacle_list:
                obstacle_circle = plt.Circle(
                    obstacle.get_position(), 
                    obstacle.radius, 
                    fill=True, 
                    color='gray'
                )
                ax.add_artist(obstacle_circle)
                
            # Redraw robot
            robot_circle = plt.Circle(
                self.robot.get_position(), 
                self.robot.radius, 
                fill=True, 
                color='r'
            )
            ax.add_artist(robot_circle)
            
            # Redraw goal
            goal_circle = plt.Circle(
                self.robot.get_goal_position(), 
                0.2, 
                fill=True, 
                color='g'
            )
            ax.add_artist(goal_circle)
            
            return ax
            
        # Create and display animation
        anim = animation.FuncAnimation(
            fig, 
            update, 
            init_func=init,
            frames=int(self.time_out_duration/self.time_step), 
            blit=True
        )
        
        plt.show()
        plt.pause(0.0001)