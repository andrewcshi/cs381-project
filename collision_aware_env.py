import os
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pybullet as p
import pybullet_data
import torch

from collision_constraint import CollisionConstraint, Control
from collision_utils import create_collision_shape_array, get_collision_info
from state import EnvState
from wbc_controller import CollisionAwareController, WholeBodyController


class CollisionAwareEnv:
    """
    PyBullet environment with collision awareness for whole body control.
    
    This environment loads a robot URDF and allows for whole body control
    with collision awareness capabilities.
    """
    
    def __init__(
        self,
        robot_urdf: str,
        use_gui: bool = True,
        dt: float = 1/240.0,
        substeps: int = 1,
        device: str = "cpu",
        collision_margin: float = 0.05,
        add_ground: bool = True,
    ):
        """
        Initialize the collision-aware environment.
        
        Args:
            robot_urdf: Path to the robot URDF file
            use_gui: Whether to use the PyBullet GUI
            dt: Simulation time step
            substeps: Number of simulation steps per control step
            device: Device for PyTorch tensors
            collision_margin: Minimum distance for collision avoidance
            add_ground: Whether to add a ground plane
        """
        self.use_gui = use_gui
        self.dt = dt
        self.substeps = substeps
        self.device = device
        self.collision_margin = collision_margin
        
        # Initialize PyBullet
        if use_gui:
            self.client_id = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            self.client_id = p.connect(p.DIRECT)
        
        # Setup simulation
        p.setTimeStep(dt)
        p.setGravity(0, 0, -9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Add ground if requested
        if add_ground:
            self.ground_id = p.loadURDF("plane.urdf")
        
        # Load robot
        self.robot_id = p.loadURDF(
            robot_urdf,
            basePosition=[0, 0, 0],
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION
        )
        
        # Get robot information
        self.num_joints = p.getNumJoints(self.robot_id)
        self.joint_indices = []
        self.controllable_joint_indices = []
        self.link_indices = []
        self.joint_limits_lower = []
        self.joint_limits_upper = []
        self.joint_names = []
        self.link_names = []
        
        # Collect joint and link information
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_type = joint_info[2]
            
            # Add to joint indices if it's a controllable joint
            if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                self.joint_indices.append(i)
                self.controllable_joint_indices.append(i)
                self.joint_limits_lower.append(joint_info[8])
                self.joint_limits_upper.append(joint_info[9])
                self.joint_names.append(joint_info[1].decode('utf-8'))
            
            # Add to link indices
            self.link_indices.append(i)
            self.link_names.append(joint_info[12].decode('utf-8'))
        
        # Convert joint limits to tensors
        self.joint_limits_lower = torch.tensor(self.joint_limits_lower, device=device)
        self.joint_limits_upper = torch.tensor(self.joint_limits_upper, device=device)
        
        # Create collision pairs for all links
        self.collision_pairs = []
        for i in range(len(self.link_indices)):
            for j in range(i+1, len(self.link_indices)):
                # Skip adjacent links as they are usually connected by a joint
                if abs(self.link_indices[i] - self.link_indices[j]) <= 1:
                    continue
                self.collision_pairs.append((self.link_indices[i], self.link_indices[j]))
        
        # Initialize state
        self.state = EnvState.initialize(
            robot_id=self.robot_id,
            joint_indices=self.joint_indices,
            link_indices=self.link_indices,
            device=device
        )
        
        # Store the gravity value we set
        self.state.gravity = torch.tensor([0, 0, -9.81], device=device).unsqueeze(0).repeat(1, 1)
        
        # Initialize control
        torque_limit = torch.ones(len(self.controllable_joint_indices), device=device) * 100.0
        kp = torch.ones(len(self.controllable_joint_indices), device=device) * 100.0
        kd = torch.ones(len(self.controllable_joint_indices), device=device) * 10.0
        
        # Create controller
        self.controller = CollisionAwareController(
            control_dim=len(self.controllable_joint_indices),
            device=device,
            torque_limit=torque_limit,
            kp=kp,
            kd=kd,
            num_envs=1,
            robot_id=self.robot_id,
            joint_indices=self.controllable_joint_indices,
            collision_margin=collision_margin
        )
        
        # Create whole body controller
        self.wbc = WholeBodyController(
            robot_id=self.robot_id,
            joint_indices=self.controllable_joint_indices,
            link_indices=self.link_indices,
            device=device,
            dt=dt,
            collision_margin=collision_margin
        )
        
        # Create control buffer
        self.control = Control(
            num_envs=1,
            control_dim=len(self.controllable_joint_indices),
            buffer_len=10,
            device=device
        )
        
        # Create collision constraint
        self.collision_constraint = CollisionConstraint(
            robot_id=self.robot_id,
            device=device,
            violation_weight=1.0,
            penalty_weight=1.0,
            terminate_on_violation=False,
            violation_distance=0.01,
            penalty_distance=collision_margin,
            collision_pairs=self.collision_pairs,
            self_collision_only=True
        )
        
        # Setup initial position
        self.reset()
    
    def reset(self):
        """Reset the environment to initial state."""
        # Reset robot to default position
        for i, joint_idx in enumerate(self.controllable_joint_indices):
            # Set joint position to middle of range
            mid_pos = (self.joint_limits_lower[i] + self.joint_limits_upper[i]) / 2.0
            p.resetJointState(self.robot_id, joint_idx, mid_pos)
        
        # Reset simulation
        for _ in range(100):
            p.stepSimulation()
        
        # Update state
        self.state.update(self.dt)
        
        return self.state
    
    def step(self, action: torch.Tensor) -> Tuple[EnvState, Dict[str, torch.Tensor]]:
        """
        Step the environment with the given action.
        
        Args:
            action: Joint position targets (num_envs, num_joints)
            
        Returns:
            state: Updated environment state
            info: Dictionary with additional information
        """
        # Push action to control buffer
        self.control.push(action)
        
        # Compute torques using controller
        torque = self.controller(action, self.state)
        self.control.torque = torque
        
        # Apply torques to the robot
        for i, joint_idx in enumerate(self.controllable_joint_indices):
            p.setJointMotorControl2(
                self.robot_id,
                joint_idx,
                p.TORQUE_CONTROL,
                force=torque[0, i].item()
            )
        
        # Step simulation
        for _ in range(self.substeps):
            p.stepSimulation()
        
        # Update state
        self.state.update(self.dt)
        
        # Get collision info
        distances, nearest_points = get_collision_info(self.state, 1.0)
        self.state.nearest_distances = distances
        self.state.nearest_points = nearest_points
        
        # Compute constraint violation and penalty
        info = {}
        constraint_info = self.collision_constraint.step(self.state, self.control)
        for key, value in constraint_info.items():
            info[f"collision/{key}"] = value
        
        return self.state, info
    
    def set_wbc_targets(self, targets: Dict[str, np.ndarray]) -> torch.Tensor:
        """
        Set targets for whole body controller and get joint commands.
        
        Args:
            targets: Dictionary mapping task names to target positions
            
        Returns:
            Joint position commands
        """
        return self.wbc.compute_control(self.state, targets)
    
    def add_wbc_task(self, task_name: str, priority: int):
        """Add a task to the whole body controller."""
        self.wbc.add_task(task_name, priority)
    
    def close(self):
        """Close the environment."""
        p.disconnect(self.client_id)


def demo():
    """Run a simple demo of the collision-aware environment."""
    # Create environment
    env = CollisionAwareEnv(
        robot_urdf="panda.urdf",
        use_gui=True,
        dt=1/240.0,
        substeps=1,
        device="cpu",
        collision_margin=0.05
    )
    
    # Add WBC tasks
    env.add_wbc_task("ee_pos_panda_link7", priority=2)  # End-effector position
    env.add_wbc_task("joint_pos", priority=1)  # Joint positions
    
    # Reset environment
    state = env.reset()
    
    # Run simulation
    for i in range(1000):
        # Set target for end-effector position that moves in a circle
        radius = 0.3
        angle = i * 0.01
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = 0.5
        
        targets = {
            "ee_pos_panda_link7": np.array([[x, y, z]]),
            "joint_pos": state.dof_pos.cpu().numpy()  # Current joint positions as targets
        }
        
        # Get joint commands from WBC
        joint_cmds = env.set_wbc_targets(targets)
        
        # Step environment
        state, info = env.step(joint_cmds)
        
        # Print collision info
        if i % 100 == 0 and "collision/min_distance" in info:
            print(f"Step {i}, Min distance: {info['collision/min_distance'].item():.4f}")
        
        time.sleep(env.dt)
    
    env.close()


if __name__ == "__main__":
    demo()