# env.py

import rvo2
import torch
import itertools
import numpy as np
from numpy.linalg import norm
from agent import Robot, Obstacle
from matplotlib import animation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

class ObstacleEnv:
    """
    Environment for robot navigation around static and dynamic obstacles.
    Supports both 2D and 3D scenarios.
    Uses RVO2 library for collision avoidance simulation.
    """
    def __init__(self, dimension='2D'):
        # Environment parameters
        self.dimension = dimension
        self.obstacle_list = []
        self.circle_radius = 4.0
        self.square_width = 10.0
        self.discomfort_dist = 0.2
        self.sim_time = 0.0
        self.time_out_duration = 25.0
        
        # 3D specific parameters
        self.height_limit = 5.0  # Z-coordinate limit
        self.enable_flying = True  # Whether obstacles can move in z-direction
        self.z_velocity_scale = 0.2  # Scale for z-direction velocity (typically lower)
        
        # Action space parameters
        self.speed_samples = 5
        self.rotation_samples = 16
        self.elevation_samples = 4 if dimension == '3D' else 1  # For 3D
        
        # RVO2 simulation parameters
        self.safety_space = 0.1  # Small safety buffer for obstacles
        self.neighbor_dist = 10.0
        self.max_neighbors = 10
        self.time_horizon = 5.0
        self.time_horizon_obst = 5.0
        self.time_step = 0.25
        self.radius = 0.3
        self.max_speed = 1.0
        
        # Dynamic obstacle parameters
        self.moving_obstacle_ratio = 0.8  # Percentage of obstacles that move
        self.obstacle_velocity_scale = 0.3  # Scale factor for obstacle velocities
        
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
            obstacle = Obstacle(self.time_step, self.obstacle_velocity_scale, 
                               self.dimension, self.z_velocity_scale)
            angle = np.random.random() * np.pi * 2
            px_noise = (np.random.random() - 0.5) * 0.5
            py_noise = (np.random.random() - 0.5) * 0.5
            px = self.circle_radius * np.cos(angle) + px_noise
            py = self.circle_radius * np.sin(angle) + py_noise
            
            # Add z-coordinate for 3D
            if self.dimension == '3D':
                pz_noise = (np.random.random() - 0.5) * 2.0
                pz = pz_noise  # Random height, can be adjusted
            else:
                pz = 0

            if self._is_valid_position(px, py, pz, obstacle):
                # Initialize obstacle with position and zero initial velocity
                if self.dimension == '3D':
                    obstacle.set(px, py, 0, 0, 0, 0, 0, pz, 0, 0)
                else:
                    obstacle.set(px, py, 0, 0, 0, 0, 0)
                
                # For circular obstacles, set center at origin
                if obstacle.movement_type in ['circular', 'helical']:
                    if self.dimension == '3D':
                        obstacle.set_circular_params(0, 0, 0)
                    else:
                        obstacle.set_circular_params(0, 0)
                    
                # Make some obstacles stationary
                if np.random.random() > self.moving_obstacle_ratio:
                    obstacle.v_pref = 0
                    obstacle.movement_type = 'static'
                    
                self.obstacle_list.append(obstacle)

    def _square_layout(self, obstacle_num):
        """Position obstacles in a square pattern."""
        while len(self.obstacle_list) < obstacle_num:
            obstacle = Obstacle(self.time_step, self.obstacle_velocity_scale, 
                               self.dimension, self.z_velocity_scale)
            px = (np.random.random() - 0.5) * self.square_width
            py = (np.random.random() - 0.5) * self.square_width
            
            # Add z-coordinate for 3D
            if self.dimension == '3D':
                pz = (np.random.random() - 0.5) * self.height_limit
            else:
                pz = 0
            
            if self._is_valid_position(px, py, pz, obstacle):
                # Initialize obstacle
                if self.dimension == '3D':
                    obstacle.set(px, py, 0, 0, 0, 0, 0, pz, 0, 0)
                else:
                    obstacle.set(px, py, 0, 0, 0, 0, 0)
                self.obstacle_list.append(obstacle)
    
    def _random_layout(self, obstacle_num):
        """Position obstacles randomly in the environment."""
        while len(self.obstacle_list) < obstacle_num:
            obstacle = Obstacle(self.time_step, self.obstacle_velocity_scale,
                              self.dimension, self.z_velocity_scale)
            px = (np.random.random() - 0.5) * self.square_width * 1.5
            py = (np.random.random() - 0.5) * self.square_width * 1.5
            
            # Add z-coordinate for 3D
            if self.dimension == '3D':
                pz = (np.random.random() - 0.5) * self.height_limit
            else:
                pz = 0
            
            if self._is_valid_position(px, py, pz, obstacle):
                # Initialize obstacle
                if self.dimension == '3D':
                    obstacle.set(px, py, 0, 0, 0, 0, 0, pz, 0, 0)
                else:
                    obstacle.set(px, py, 0, 0, 0, 0, 0)
                self.obstacle_list.append(obstacle)
    
    def _is_valid_position(self, px, py, pz, obstacle):
        """Check if position is valid (not colliding with other entities)."""
        # Get all entities to check against
        entities = [self.robot] + self.obstacle_list if hasattr(self, 'robot') else self.obstacle_list
        
        # Check for collisions with existing entities
        for entity in entities:
            if self.dimension == '3D':
                min_dist = obstacle.radius + entity.radius + self.discomfort_dist
                if norm((px - entity.px, py - entity.py, pz - entity.pz)) < min_dist:
                    return False
            else:
                min_dist = obstacle.radius + entity.radius + self.discomfort_dist
                if norm((px - entity.px, py - entity.py)) < min_dist:
                    return False
        
        # Check if too close to robot's goal (if robot exists)
        if hasattr(self, 'robot'):
            if self.dimension == '3D':
                if norm((px - self.robot.gx, py - self.robot.gy, pz - self.robot.gz)) < min_dist:
                    return False
            else:
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
        
        if self.dimension == '3D':
            # Generate elevation angles for 3D
            elevations = np.linspace(-np.pi/4, np.pi/4, self.elevation_samples, endpoint=True)
            
            # Create all combinations of speed, direction, and elevation
            action_space = []
            for rotation, speed, elevation in itertools.product(rotations, speeds, elevations):
                # Convert from spherical to cartesian coordinates
                xy_component = speed * np.cos(elevation)
                z_component = speed * np.sin(elevation)
                action_space.append([
                    xy_component * np.cos(rotation), 
                    xy_component * np.sin(rotation),
                    z_component
                ])
        else:
            # 2D case: Create all combinations of speed and direction
            action_space = []
            for rotation, speed in itertools.product(rotations, speeds):
                action_space.append([speed * np.cos(rotation), speed * np.sin(rotation)])
                
        return action_space

    def reset(self, obstacle_num, layout="circle", test_phase=False, counter=None, 
          moving_obstacle_ratio=None, obstacle_velocity_scale=None, dimension=None):
        """
        Reset the environment with new obstacles.
        
        Args:
            obstacle_num: Number of obstacles to generate
            layout: Layout pattern for obstacles ("circle", "square", or "random")
            test_phase: Flag for reproducible testing
            counter: Seed value for reproducible testing
            moving_obstacle_ratio: Ratio of moving obstacles (0-1)
            obstacle_velocity_scale: Velocity scale factor for obstacles
            dimension: Dimension of the environment ('2D' or '3D')
            
        Returns:
            Initial observation state
        """
        # Update dimension if specified
        if dimension is not None:
            self.dimension = dimension
            # Reset action space
            self.vel_samples = self.action_space()
            
        # Initialize robot agent with current dimension
        self.robot = Robot(self.time_step, self.dimension)
        self.obstacle_list = []
        
        # Update dynamic obstacle parameters if provided
        if moving_obstacle_ratio is not None:
            self.moving_obstacle_ratio = moving_obstacle_ratio
        if obstacle_velocity_scale is not None:
            self.obstacle_velocity_scale = obstacle_velocity_scale
        
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
            # For 3D, we'll handle the z-coordinate separately as RVO2 is primarily for 2D
            # (Here we'll project to 2D for collision detection)
            self.sim.addAgent(
                (obstacle.px, obstacle.py), 
                *params, 
                obstacle.radius + 0.01 + self.safety_space,
                obstacle.v_pref,  # Now using actual preferred velocity
                (obstacle.vx, obstacle.vy)  # Initial velocity
            )
            
        # Reset simulation time and distance to goal
        self.sim_time = 0.0
        if self.dimension == '3D':
            self.dg = norm(np.array([self.robot.px, self.robot.py, self.robot.pz]) - 
                          np.array([self.robot.gx, self.robot.gy, self.robot.gz]))
        else:
            self.dg = norm(np.array([self.robot.px, self.robot.py]) - 
                          np.array([self.robot.gx, self.robot.gy]))
        
        return obs

    def step(self, action):
        """
        Take a step in the environment with the given action.
        
        Args:
            action: Robot velocity action (2D or 3D)
            
        Returns:
            obs: New observation
            reward: Reward for the action
            done: Whether episode is done
            info: Additional information
        """
        # Update preferred velocities for obstacles in RVO2 simulator
        for i, obstacle in enumerate(self.obstacle_list):
            if obstacle.movement_type != 'static':
                if self.dimension == '3D':
                    vx, vy, vz = obstacle.calculate_velocity()
                    # RVO2 is 2D, so we only update x,y components
                    self.sim.setAgentPrefVelocity(i, (vx, vy))
                else:
                    vx, vy = obstacle.calculate_velocity()
                    self.sim.setAgentPrefVelocity(i, (vx, vy))
            else:
                # Static obstacles have zero preferred velocity
                self.sim.setAgentPrefVelocity(i, (0.0, 0.0))

        # Advance simulation
        self.sim.doStep()
        
        # Update obstacle positions and velocities from simulation
        for i, obstacle in enumerate(self.obstacle_list):
            if self.dimension == '3D':
                # Update x,y from RVO2, keep z from obstacle's own movement
                px, py = self.sim.getAgentPosition(i)
                vx, vy = self.sim.getAgentVelocity(i)
                
                # Step the z-coordinate manually
                pz = obstacle.pz
                vz = obstacle.vz
                
                # Set position and velocity
                obstacle.set_position((px, py, pz))
                obstacle.set_velocity((vx, vy, vz))
            else:
                obstacle.set_position(self.sim.getAgentPosition(i))
                obstacle.set_velocity(self.sim.getAgentVelocity(i))
            
        # Update robot with action
        self.robot.step(action)
        
        # Update simulation time
        self.sim_time += self.time_step
        
        # Compute reward and check terminal conditions
        distance_list = []
        for obstacle in self.obstacle_list:
            if self.dimension == '3D':
                distance = norm(np.array([obstacle.px, obstacle.py, obstacle.pz]) - 
                              np.array([self.robot.px, self.robot.py, self.robot.pz])) - obstacle.radius - self.robot.radius
            else:
                distance = norm(np.array([obstacle.px, obstacle.py]) - 
                              np.array([self.robot.px, self.robot.py])) - obstacle.radius - self.robot.radius
            distance_list.append(distance)
                
        d_min = min(distance_list) if distance_list else float('inf')
        
        if self.dimension == '3D':
            current_dg = norm(np.array([self.robot.px, self.robot.py, self.robot.pz]) - 
                             np.array([self.robot.gx, self.robot.gy, self.robot.gz]))
        else:
            current_dg = norm(np.array([self.robot.px, self.robot.py]) - 
                             np.array([self.robot.gx, self.robot.gy]))
                             
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
            info = f"Goal reached, time {self.sim_time:.2f}"  # Keep this format consistent
            # Add a debug print to confirm when goals are reached during evaluation
            print(f"Goal reached at time {self.sim_time:.2f}!")
        else:
            # Add small bonus for avoiding dynamic obstacles
            num_moving_obstacles = sum(1 for obs in self.obstacle_list if obs.v_pref > 0)
            avoidance_bonus = 0.05 * num_moving_obstacles if num_moving_obstacles > 0 else 0
            reward = delta_d + avoidance_bonus
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
        
        # Handle case when there are no obstacles
        if len(obs) > 1:
            obstacle_states = torch.tensor(np.array(obs[1:]), dtype=torch.float32)
            num_obstacles = obstacle_states.shape[0]
        else:
            # Create empty tensor with appropriate dimensions
            if self.dimension == '3D':
                obstacle_states = torch.zeros((0, 7), dtype=torch.float32)  # 3D obstacle observable has 7 dims
            else:
                obstacle_states = torch.zeros((0, 5), dtype=torch.float32)  # 2D obstacle observable has 5 dims
            num_obstacles = 0
        
        # Adjust for 2D vs 3D
        if self.dimension == '3D':
            # 3D conversion
            # Calculate goal direction in 3D
            dx = robot_state[7] - robot_state[0]  # gx - px
            dy = robot_state[8] - robot_state[1]  # gy - py
            dz = robot_state[9] - robot_state[2]  # gz - pz
            
            # Calculate distance to goal
            dg = torch.tensor(norm((dx.item(), dy.item(), dz.item())), dtype=torch.float32)
            
            # Calculate angles (azimuth and elevation to goal)
            rot_xy = torch.atan2(dy, dx)  # Rotation in xy-plane (azimuth)
            dxy = torch.sqrt(dx*dx + dy*dy)
            rot_z = torch.atan2(dz, dxy)  # Elevation angle
            
            # Robot properties in transformed frame
            v_pref = robot_state[10]  # Preferred velocity
            radius = robot_state[6]   # Robot radius
            
            # Transform obstacles to robot-centric frame
            transformed_obstacles = []
            
            for i in range(num_obstacles):
                obstacle = obstacle_states[i]
                
                # Calculate relative position
                rel_x = obstacle[0] - robot_state[0]
                rel_y = obstacle[1] - robot_state[1]
                rel_z = obstacle[2] - robot_state[2]
                
                # Rotate to align with goal direction
                # First rotate around z-axis (azimuth)
                cos_rot_xy = torch.cos(rot_xy)
                sin_rot_xy = torch.sin(rot_xy)
                x1 = rel_x * cos_rot_xy + rel_y * sin_rot_xy
                y1 = -rel_x * sin_rot_xy + rel_y * cos_rot_xy
                
                # Then rotate around y-axis (elevation)
                cos_rot_z = torch.cos(rot_z)
                sin_rot_z = torch.sin(rot_z)
                x2 = x1 * cos_rot_z - rel_z * sin_rot_z
                z2 = x1 * sin_rot_z + rel_z * cos_rot_z
                
                # Apply similar rotations to velocity
                vel_x = obstacle[3] - robot_state[3]
                vel_y = obstacle[4] - robot_state[4]
                vel_z = obstacle[5] - robot_state[5]
                
                vx1 = vel_x * cos_rot_xy + vel_y * sin_rot_xy
                vy1 = -vel_x * sin_rot_xy + vel_y * cos_rot_xy
                
                vx2 = vx1 * cos_rot_z - vel_z * sin_rot_z
                vz2 = vx1 * sin_rot_z + vel_z * cos_rot_z
                
                # Calculate distance to obstacle
                dist = norm((rel_x.item(), rel_y.item(), rel_z.item()))
                
                # Combine all features - exactly 10 feature dimensions for obstacle_dim
                obstacle_feature = torch.tensor([
                    x2, y1, z2,             # Transformed position (3)
                    vx2, vy1, vz2,          # Transformed velocity (3)
                    obstacle[6],            # Obstacle radius (1)
                    dist,                   # Distance to obstacle (1)
                    radius + obstacle[6],   # Sum of radii (1)
                    0.0                     # Extra padding to make it 10 features total
                ], dtype=torch.float32)
                
                transformed_obstacles.append(obstacle_feature)
            
            # Handle empty obstacle list
            if not transformed_obstacles:
                # Create a dummy obstacle far away
                dummy_obstacle = torch.tensor([
                    100.0, 100.0, 100.0,    # Far away position (3)
                    0.0, 0.0, 0.0,          # Zero velocity (3)
                    0.1,                    # Small radius (1)
                    150.0,                  # Large distance (1)
                    radius.item() + 0.1,    # Sum of radii (1)
                    0.0                     # Padding (1)
                ], dtype=torch.float32).unsqueeze(0)
                
                obstacle_tensor = dummy_obstacle
                num_obstacles = 1
            else:
                obstacle_tensor = torch.stack(transformed_obstacles)
            
            # Create robot features - need exactly 9 dimensions to match robot_dim_3d
            robot_features = torch.tensor([
                dg,                # Distance to goal (1)
                rot_xy,            # Azimuth angle (1)
                rot_z,             # Elevation angle (1)
                v_pref,            # Preferred velocity (1)
                radius,            # Robot radius (1)
                0.0, 0.0, 0.0, 0.0 # Padding to make exactly 9 dimensions
            ], dtype=torch.float32).unsqueeze(0).expand(num_obstacles, -1)
                    
            # The key is to put the 9-dimensional robot tensor FIRST, followed by the
            # 10-dimensional obstacle tensor, so that the model's slicing works correctly
            robot_dim_expanded = robot_features
            
            # Combine into state tensor with shape [batch_size, num_obstacles, robot_dim+obstacle_dim]
            state_tensor = torch.cat([robot_dim_expanded, obstacle_tensor], dim=1).unsqueeze(0)
            
        else:
            # Original 2D implementation - UNCHANGED
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
            
            # Handle no obstacles case
            if num_obstacles == 0:
                # Create dummy data with one obstacle far away
                dg = torch.tensor([[norm((dx.item(), dy.item()))]], dtype=torch.float32)
                rot_expand = torch.tensor([[rot]], dtype=torch.float32)
                vx = torch.tensor([[robot_state[2] * torch.cos(rot) + robot_state[3] * torch.sin(rot)]], dtype=torch.float32)
                vy = torch.tensor([[robot_state[3] * torch.cos(rot) - robot_state[2] * torch.sin(rot)]], dtype=torch.float32)
                v_pref = torch.tensor([[robot_state[7]]], dtype=torch.float32)
                radius = torch.tensor([[robot_state[4]]], dtype=torch.float32)
                
                # Dummy obstacle values
                px_obs = torch.tensor([[10.0]])  # Far away
                py_obs = torch.tensor([[10.0]])  # Far away
                vx_obs = torch.tensor([[0.0]])   # Not moving
                vy_obs = torch.tensor([[0.0]])   # Not moving
                radius_obs = torch.tensor([[0.1]])  # Small radius
                da = torch.tensor([[15.0]])  # Far distance
                radius_sum = torch.tensor([[radius[0, 0].item() + 0.1]])  # Sum of radii
                
                num_obstacles = 1
            
            # Concatenate all features into a single state tensor
            state_tensor = torch.cat([
                dg, rot_expand, vx, vy, v_pref, radius, 
                px_obs, py_obs, vx_obs, vy_obs, 
                radius_obs, da, radius_sum
            ], dim=1).unsqueeze(0)  # Add batch dimension
        
        return state_tensor
    
    def render(self):
        """Render the environment with matplotlib animation."""
        if self.dimension == '3D':
            # 3D rendering
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlim(-6, 6)
            ax.set_ylim(-6, 6)
            ax.set_zlim(-6, 6)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            def init():
                """Initialize animation with obstacles and robot."""
                artists = []
                
                # Draw obstacles as spheres using scatter
                for obstacle in self.obstacle_list:
                    # Scale point size based on radius
                    size = obstacle.radius * 100
                    color = 'gray' if obstacle.v_pref == 0 else 'blue'
                    
                    scatter = ax.scatter(
                        obstacle.px, obstacle.py, obstacle.pz,
                        s=size, c=color, alpha=0.7
                    )
                    artists.append(scatter)
                    
                # Draw robot
                robot_scatter = ax.scatter(
                    self.robot.px, self.robot.py, self.robot.pz,
                    s=self.robot.radius * 100, c='r', alpha=0.7
                )
                artists.append(robot_scatter)
                
                # Draw goal
                goal_scatter = ax.scatter(
                    self.robot.gx, self.robot.gy, self.robot.gz,
                    s=20, c='g', alpha=0.7
                )
                artists.append(goal_scatter)
                
                return artists

            def update(i):
                """Update animation frame."""
                ax.clear()
                ax.set_xlim(-6, 6)
                ax.set_ylim(-6, 6)
                ax.set_zlim(-6, 6)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                
                artists = []
                
                # Redraw obstacles
                for obstacle in self.obstacle_list:
                    size = obstacle.radius * 100
                    color = 'gray' if obstacle.v_pref == 0 else 'blue'
                    
                    scatter = ax.scatter(
                        obstacle.px, obstacle.py, obstacle.pz,
                        s=size, c=color, alpha=0.7
                    )
                    artists.append(scatter)
                    
                # Redraw robot
                robot_scatter = ax.scatter(
                    self.robot.px, self.robot.py, self.robot.pz,
                    s=self.robot.radius * 100, c='r', alpha=0.7
                )
                artists.append(robot_scatter)
                
                # Redraw goal
                goal_scatter = ax.scatter(
                    self.robot.gx, self.robot.gy, self.robot.gz,
                    s=20, c='g', alpha=0.7
                )
                artists.append(goal_scatter)
                
                ax.set_title(f'Frame: {i}')
                
                return artists
                
            # Create and display animation
            anim = animation.FuncAnimation(
                fig, 
                update, 
                init_func=init,
                frames=int(self.time_out_duration/self.time_step), 
                interval=50,
                blit=False
            )
            
            plt.show()
            plt.pause(0.0001)
            
        else:
            # Original 2D rendering
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