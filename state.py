import dataclasses
from typing import Dict, List, Optional, Tuple

import numpy as np
import pybullet as p
import torch


@dataclasses.dataclass
class EnvState:
    """
    Represents the state of the environment in PyBullet.
    Similar to IsaacGym's EnvState but adapted for PyBullet.
    """
    robot_id: int
    num_bodies: int
    joint_indices: List[int]
    link_indices: List[int]
    
    # Base state
    root_pos: torch.Tensor  # (num_envs, 3)
    root_quat: torch.Tensor  # (num_envs, 4) [x,y,z,w]
    root_lin_vel: torch.Tensor  # (num_envs, 3)
    root_ang_vel: torch.Tensor  # (num_envs, 3)
    
    # Joint state
    dof_pos: torch.Tensor  # (num_envs, num_dof)
    dof_vel: torch.Tensor  # (num_envs, num_dof)
    
    # Previous state (for computing changes)
    prev_dof_pos: torch.Tensor  # (num_envs, num_dof)
    prev_dof_vel: torch.Tensor  # (num_envs, num_dof)

    # Link states
    rigid_body_pos: torch.Tensor  # (num_envs, num_bodies, 3)
    rigid_body_quat: torch.Tensor  # (num_envs, num_bodies, 4)
    rigid_body_lin_vel: torch.Tensor  # (num_envs, num_bodies, 3)
    rigid_body_ang_vel: torch.Tensor  # (num_envs, num_bodies, 3)

    # Contact information
    contact_forces: torch.Tensor  # (num_envs, num_bodies, 3)
    
    # Collision information
    nearest_distances: torch.Tensor  # (num_envs, num_collision_pairs)
    nearest_points: torch.Tensor  # (num_envs, num_collision_pairs, 2, 3)
    collision_pairs: List[Tuple[int, int]]  # [(body_id_1, body_id_2), ...]
    
    # Environment info
    episode_time: torch.Tensor  # (num_envs,)
    gravity: torch.Tensor  # (num_envs, 3)
    
    # Timing
    time: float
    prev_time: float
    
    # Device config
    device: str
    
    @property
    def sim_dt(self) -> float:
        """Return the time step since the last update."""
        return self.time - self.prev_time
    
    @classmethod
    def initialize(cls, 
                  robot_id: int, 
                  joint_indices: List[int], 
                  link_indices: List[int],
                  num_envs: int = 1, 
                  device: str = "cpu") -> "EnvState":
        """Initialize EnvState from PyBullet state."""
        num_bodies = len(link_indices)
        num_dof = len(joint_indices)
        
        # Get base position and orientation
        base_pos, base_orn = p.getBasePositionAndOrientation(robot_id)
        base_vel = p.getBaseVelocity(robot_id)
        
        # Convert to tensors and repeat for each environment
        root_pos = torch.tensor(base_pos, device=device, dtype=torch.float32).unsqueeze(0).repeat(num_envs, 1)
        root_quat = torch.tensor(base_orn, device=device, dtype=torch.float32).unsqueeze(0).repeat(num_envs, 1)
        root_lin_vel = torch.tensor(base_vel[0], device=device, dtype=torch.float32).unsqueeze(0).repeat(num_envs, 1)
        root_ang_vel = torch.tensor(base_vel[1], device=device, dtype=torch.float32).unsqueeze(0).repeat(num_envs, 1)
        
        # Get joint states
        dof_pos = torch.zeros((num_envs, num_dof), device=device)
        dof_vel = torch.zeros((num_envs, num_dof), device=device)
        
        for i, joint_idx in enumerate(joint_indices):
            joint_state = p.getJointState(robot_id, joint_idx)
            dof_pos[:, i] = joint_state[0]
            dof_vel[:, i] = joint_state[1]
        
        # Get link states for all bodies
        rigid_body_pos = torch.zeros((num_envs, num_bodies, 3), device=device)
        rigid_body_quat = torch.zeros((num_envs, num_bodies, 4), device=device)
        rigid_body_lin_vel = torch.zeros((num_envs, num_bodies, 3), device=device)
        rigid_body_ang_vel = torch.zeros((num_envs, num_bodies, 3), device=device)
        
        for i, link_idx in enumerate(link_indices):
            link_state = p.getLinkState(robot_id, link_idx, computeLinkVelocity=1)
            rigid_body_pos[:, i] = torch.tensor(link_state[0], device=device)
            rigid_body_quat[:, i] = torch.tensor(link_state[1], device=device)
            rigid_body_lin_vel[:, i] = torch.tensor(link_state[6], device=device)
            rigid_body_ang_vel[:, i] = torch.tensor(link_state[7], device=device)
        
        # Initialize contact forces
        contact_forces = torch.zeros((num_envs, num_bodies, 3), device=device)
        
        # Get collision pairs
        collision_pairs = []
        for i in range(num_bodies):
            for j in range(i+1, num_bodies):
                collision_pairs.append((link_indices[i], link_indices[j]))
                
        # Initialize nearest distances and points
        num_collision_pairs = len(collision_pairs)
        nearest_distances = torch.ones((num_envs, num_collision_pairs), device=device) * 1000.0
        nearest_points = torch.zeros((num_envs, num_collision_pairs, 2, 3), device=device)
        
        # Use default gravity (PyBullet doesn't have getGravity())
        # Standard gravity is [0, 0, -9.81]
        gravity = torch.tensor([0, 0, -9.81], device=device).unsqueeze(0).repeat(num_envs, 1)
        
        # Get time
        time = 0.0
        
        return cls(
            robot_id=robot_id,
            num_bodies=num_bodies,
            joint_indices=joint_indices,
            link_indices=link_indices,
            root_pos=root_pos,
            root_quat=root_quat,
            root_lin_vel=root_lin_vel,
            root_ang_vel=root_ang_vel,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            prev_dof_pos=dof_pos.clone(),
            prev_dof_vel=dof_vel.clone(),
            rigid_body_pos=rigid_body_pos,
            rigid_body_quat=rigid_body_quat,
            rigid_body_lin_vel=rigid_body_lin_vel,
            rigid_body_ang_vel=rigid_body_ang_vel,
            contact_forces=contact_forces,
            nearest_distances=nearest_distances,
            nearest_points=nearest_points,
            collision_pairs=collision_pairs,
            episode_time=torch.zeros(num_envs, device=device),
            gravity=gravity,
            time=time,
            prev_time=time,
            device=device,
        )
    
    def update(self, dt: float) -> None:
        """Update the state from PyBullet."""
        self.prev_dof_pos = self.dof_pos.clone()
        self.prev_dof_vel = self.dof_vel.clone()
        self.prev_time = self.time
        
        # Update base state
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
        base_vel = p.getBaseVelocity(self.robot_id)
        
        self.root_pos[0] = torch.tensor(base_pos, device=self.device)
        self.root_quat[0] = torch.tensor(base_orn, device=self.device)
        self.root_lin_vel[0] = torch.tensor(base_vel[0], device=self.device)
        self.root_ang_vel[0] = torch.tensor(base_vel[1], device=self.device)
        
        # Update joint states
        for i, joint_idx in enumerate(self.joint_indices):
            joint_state = p.getJointState(self.robot_id, joint_idx)
            self.dof_pos[0, i] = joint_state[0]
            self.dof_vel[0, i] = joint_state[1]
        
        # Update link states
        for i, link_idx in enumerate(self.link_indices):
            link_state = p.getLinkState(self.robot_id, link_idx, computeLinkVelocity=1)
            self.rigid_body_pos[0, i] = torch.tensor(link_state[0], device=self.device)
            self.rigid_body_quat[0, i] = torch.tensor(link_state[1], device=self.device)
            self.rigid_body_lin_vel[0, i] = torch.tensor(link_state[6], device=self.device)
            self.rigid_body_ang_vel[0, i] = torch.tensor(link_state[7], device=self.device)
        
        # Update contact forces
        self.contact_forces.zero_()
        contact_points = p.getContactPoints(self.robot_id)
        for contact in contact_points:
            link_index = contact[3]  # linkIndexA
            if link_index >= 0:  # -1 is for the base
                idx = self.link_indices.index(link_index) if link_index in self.link_indices else -1
                if idx >= 0:
                    force_vector = torch.tensor(
                        [contact[9] * n for n in contact[7]],  # normal_force * normal_direction
                        device=self.device
                    )
                    self.contact_forces[0, idx] += force_vector
        
        # Update collision information
        self._update_collision_info()
        
        # Update time
        self.time += dt
        self.episode_time += dt
    
    def _update_collision_info(self):
        """Update collision information using PyBullet's nearestPoints."""
        for i, (link_a, link_b) in enumerate(self.collision_pairs):
            # Get the closest points between the two links
            # Use -1 for base link
            result = p.getClosestPoints(
                self.robot_id, self.robot_id, 
                distance=1.0,  # Only report distances up to 1.0
                linkIndexA=link_a if link_a >= 0 else -1,
                linkIndexB=link_b if link_b >= 0 else -1
            )
            
            if result:
                # Take the closest point
                closest = min(result, key=lambda x: x[8])  # x[8] is the distance
                self.nearest_distances[0, i] = closest[8]
                self.nearest_points[0, i, 0] = torch.tensor(closest[5], device=self.device)  # positionOnA
                self.nearest_points[0, i, 1] = torch.tensor(closest[6], device=self.device)  # positionOnB
            else:
                # No collision detected within the distance threshold
                self.nearest_distances[0, i] = 1.0
                # Keep previous points or set to a default value