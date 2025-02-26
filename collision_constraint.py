from abc import abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pybullet as p
import torch

from state import EnvState


class Control:
    """Simple control class to store control commands and history."""
    def __init__(self, num_envs: int, control_dim: int, buffer_len: int, device: str):
        self.torque = torch.zeros((num_envs, control_dim), device=device)
        self.buffer = torch.zeros((num_envs, buffer_len, control_dim), device=device)
        
    def push(self, action: torch.Tensor):
        """Push action to buffer."""
        self.buffer = torch.cat((action[:, None, :], self.buffer[:, :-1]), dim=1)
    
    @property
    def prev_action(self):
        return self.buffer[:, 1]
    
    @property
    def action(self):
        return self.buffer[:, 0]
    
    @property
    def ctrl_dim(self) -> int:
        return self.buffer.shape[-1]


class Constraint:
    """
    Base class for constraints in a PyBullet environment.
    Constraints can detect violations, compute penalties, and modify control signals.
    """
    def __init__(
        self,
        robot_id: int, 
        device: str,
        violation_weight: float,
        penalty_weight: float,
        terminate_on_violation: bool,
        skip_stats: bool = True,
    ):
        self.robot_id = robot_id
        self.device = device
        self.violation_weight = violation_weight
        self.penalty_weight = penalty_weight
        self.terminate_on_violation = terminate_on_violation
        self.skip_stats = skip_stats
    
    def reward(self, state: EnvState, control: Control) -> Dict[str, torch.Tensor]:
        """Compute rewards from constraint violation and penalties."""
        retval = {}
        if self.violation_weight != 0:
            retval["hard_violation"] = (
                self.check_violation(state=state, control=control)
                * self.violation_weight
            )
        if self.penalty_weight != 0:
            retval["soft_penalty"] = (
                self.compute_penalty(state=state, control=control) * self.penalty_weight
            )
        return retval
    
    @abstractmethod
    def check_violation(self, state: EnvState, control: Control) -> torch.Tensor:
        """
        Check if the constraint is violated.
        Returns a boolean tensor indicating if the constraint is violated.
        """
        raise NotImplementedError()
    
    @abstractmethod
    def compute_penalty(self, state: EnvState, control: Control) -> torch.Tensor:
        """
        Compute penalty for soft constraints.
        Returns a tensor of penalties, higher values mean more violation.
        """
        raise NotImplementedError()
    
    def check_termination(self, state: EnvState, control: Control) -> torch.Tensor:
        """Check if the constraint violation should terminate the episode."""
        if self.terminate_on_violation:
            return self.check_violation(state=state, control=control)
        return torch.zeros(state.dof_pos.shape[0], device=self.device, dtype=torch.bool)
    
    def step(self, state: EnvState, control: Control) -> Dict[str, torch.Tensor]:
        """Collect statistics for this constraint."""
        if self.skip_stats:
            return {}
        violation = self.check_violation(state=state, control=control)
        return {"violation": violation}


class CollisionConstraint(Constraint):
    """
    Constraint that detects and penalizes collisions.
    Can handle both self-collisions and environment collisions.
    """
    def __init__(
        self,
        robot_id: int, 
        device: str,
        violation_weight: float,
        penalty_weight: float,
        terminate_on_violation: bool,
        # constraint specific parameters
        violation_distance: float,
        penalty_distance: float,
        collision_pairs: Optional[List[Tuple[int, int]]] = None,
        link_names: Optional[List[str]] = None,
        self_collision_only: bool = True,
    ):
        super().__init__(
            robot_id=robot_id,
            device=device,
            violation_weight=violation_weight,
            penalty_weight=penalty_weight,
            terminate_on_violation=terminate_on_violation,
        )
        self.violation_distance = violation_distance
        self.penalty_distance = penalty_distance
        self.self_collision_only = self_collision_only
        
        # Setup collision pairs if provided
        self.link_names = link_names
        self.link_indices = None
        
        if link_names is not None:
            # Get link indices from names
            self.link_indices = []
            for link_name in link_names:
                for i in range(p.getNumJoints(robot_id)):
                    joint_info = p.getJointInfo(robot_id, i)
                    if joint_info[12].decode('utf-8') == link_name:
                        self.link_indices.append(i)
                        break
            assert len(self.link_indices) == len(link_names), "Some links were not found"
        
        # Set collision pairs
        self.collision_pairs = collision_pairs
        if self.collision_pairs is None and self.link_indices is not None:
            # Create pairs for all combinations of link indices
            self.collision_pairs = []
            for i in range(len(self.link_indices)):
                for j in range(i+1, len(self.link_indices)):
                    self.collision_pairs.append((self.link_indices[i], self.link_indices[j]))
    
    def check_violation(self, state: EnvState, control: Control) -> torch.Tensor:
        """Check if any collision distance is below the violation threshold."""
        return (state.nearest_distances < self.violation_distance).any(dim=1)
    
    def compute_penalty(self, state: EnvState, control: Control) -> torch.Tensor:
        """Compute penalty based on how close objects are."""
        # Penalize when distance is below penalty_distance
        # The closer objects are, the higher the penalty
        distances = state.nearest_distances
        penalties = torch.relu(self.penalty_distance - distances)
        return penalties.sum(dim=1)
    
    def step(self, state: EnvState, control: Control) -> Dict[str, torch.Tensor]:
        """Collect statistics for collision constraint."""
        if self.skip_stats:
            return {}
        
        violation = self.check_violation(state=state, control=control)
        
        stats = {
            "violation": violation,
            "min_distance": state.nearest_distances.min(dim=1)[0],
            "avg_distance": state.nearest_distances.mean(dim=1),
        }
        
        # Add per-pair statistics if we're tracking specific links
        if self.link_names is not None:
            for i, (link_a, link_b) in enumerate(self.collision_pairs):
                link_a_name = p.getJointInfo(self.robot_id, link_a)[12].decode('utf-8') if link_a >= 0 else "base"
                link_b_name = p.getJointInfo(self.robot_id, link_b)[12].decode('utf-8') if link_b >= 0 else "base"
                pair_name = f"{link_a_name}_{link_b_name}"
                stats[f"distance/{pair_name}"] = state.nearest_distances[:, i]
        
        return stats