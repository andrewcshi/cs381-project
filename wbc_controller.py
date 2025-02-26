from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pybullet as p
import torch
from scipy.spatial.transform import Rotation

from collision_constraint import Control
from collision_utils import compute_collision_jacobian, modify_control_for_collision_avoidance
from state import EnvState


class PDController:
    """Simple PD controller for joint position control."""
    def __init__(
        self,
        control_dim: int,
        device: str,
        torque_limit: torch.Tensor,
        kp: torch.Tensor,
        kd: torch.Tensor,
        num_envs: int,
        scale: Optional[torch.Tensor] = None,
        offset: Optional[torch.Tensor] = None,
    ):
        self.scale = (
            torch.ones((1, control_dim), device=device)
            if scale is None
            else scale.to(device)
        )
        self.offset = (
            torch.zeros((1, control_dim), device=device)
            if offset is None
            else offset.to(device)
        )
        self.torque_limit = torque_limit.to(device)
        self.device = device
        self.control_dim = control_dim
        self.kp = kp.to(self.device)[None, :].float()
        self.kd = kd.to(self.device)[None, :].float()
        self.num_envs = num_envs
        self.prev_normalized_target = torch.zeros((1, control_dim), device=self.device)

    def __call__(
        self,
        action: torch.Tensor,
        state: EnvState,
    ):
        """
        action: (num_envs, control_dim) from the network
        """
        return self.compute_torque(
            normalized_action=action * self.scale
            + self.offset.repeat(action.shape[0], 1),
            state=state,
        )

    def compute_torque(self, normalized_action: torch.Tensor, state: EnvState):
        """
        normalized_action: (control_dim, ) after __call__
        """
        self.prev_normalized_target = normalized_action
        return torch.clip(
            normalized_action, min=-self.torque_limit, max=self.torque_limit
        )


class PositionController(PDController):
    """PD controller for joint position control."""
    def compute_torque(
        self,
        normalized_action: torch.Tensor,
        state: EnvState,
    ):
        curr_pos = state.dof_pos.clone()
        curr_vel = state.dof_vel.clone()
        assert normalized_action.shape == curr_pos.shape
        assert curr_vel.shape == curr_pos.shape
        if normalized_action.shape[0] != self.kp.shape[0]:
            self.kp = self.kp.repeat(normalized_action.shape[0], 1)
            self.kd = self.kd.repeat(normalized_action.shape[0], 1)
        torques = self.kp * (normalized_action - curr_pos) - self.kd * curr_vel
        return super().compute_torque(torques, state)


class CollisionAwareController(PositionController):
    """Position controller with collision avoidance capabilities."""
    def __init__(
        self,
        control_dim: int,
        device: str,
        torque_limit: torch.Tensor,
        kp: torch.Tensor,
        kd: torch.Tensor,
        num_envs: int,
        robot_id: int,
        joint_indices: List[int],
        collision_margin: float = 0.05,
        collision_avoidance_gain: float = 10.0,
        scale: Optional[torch.Tensor] = None,
        offset: Optional[torch.Tensor] = None,
    ):
        super().__init__(
            control_dim=control_dim,
            device=device,
            torque_limit=torque_limit,
            kp=kp,
            kd=kd,
            num_envs=num_envs,
            scale=scale,
            offset=offset,
        )
        self.robot_id = robot_id
        self.joint_indices = joint_indices
        self.collision_margin = collision_margin
        self.collision_avoidance_gain = collision_avoidance_gain

    def compute_torque(
        self,
        normalized_action: torch.Tensor,
        state: EnvState,
    ):
        # Modify the desired position to avoid collisions
        collision_aware_action = modify_control_for_collision_avoidance(
            state=state,
            control=normalized_action,
            collision_margin=self.collision_margin,
            joint_indices=self.joint_indices,
            gain=self.collision_avoidance_gain,
        )
        
        # Use the base controller to compute torques
        return super().compute_torque(collision_aware_action, state)


class WholeBodyController:
    """
    Whole Body Controller with collision avoidance.
    Manages multiple tasks with priorities and constraints.
    """
    def __init__(
        self,
        robot_id: int,
        joint_indices: List[int],
        link_indices: List[int],
        device: str,
        dt: float,
        collision_margin: float = 0.05,
    ):
        self.robot_id = robot_id
        self.joint_indices = joint_indices
        self.link_indices = link_indices
        self.device = device
        self.dt = dt
        self.collision_margin = collision_margin
        
        # Initialize task list
        self.tasks = []
        self.task_priorities = []
        
        # Initialize constraints
        self.constraints = []
        
        # Get DOF limits
        self.joint_limits_lower = []
        self.joint_limits_upper = []
        
        for joint_idx in joint_indices:
            joint_info = p.getJointInfo(robot_id, joint_idx)
            self.joint_limits_lower.append(joint_info[8])  # lower limit
            self.joint_limits_upper.append(joint_info[9])  # upper limit
        
        self.joint_limits_lower = torch.tensor(self.joint_limits_lower, device=device)
        self.joint_limits_upper = torch.tensor(self.joint_limits_upper, device=device)
        
        # Cache for Jacobians and other matrices
        self.cached_jacobians = {}
    
    def add_task(self, task_name: str, priority: int):
        """Add a task with a specified priority (higher number = higher priority)."""
        self.tasks.append(task_name)
        self.task_priorities.append(priority)
    
    def add_constraint(self, constraint_name: str):
        """Add a constraint to the controller."""
        self.constraints.append(constraint_name)
    
    def compute_control(self, state: EnvState, target_positions: Dict[str, np.ndarray]) -> torch.Tensor:
        """
        Compute control action to achieve tasks while respecting constraints.
        """
        num_dof = len(self.joint_indices)
        num_envs = state.dof_pos.shape[0]
        
        # Initialize joint position commands with current positions
        joint_commands = state.dof_pos.clone()
        
        # Sort tasks by priority (highest first)
        sorted_indices = np.argsort(-np.array(self.task_priorities))
        sorted_tasks = [self.tasks[i] for i in sorted_indices]
        
        # Initialize null space projector as identity (all DOFs available)
        null_space = torch.eye(num_dof, device=self.device, dtype=torch.float32).unsqueeze(0).repeat(num_envs, 1, 1)
        
        # Process tasks in priority order
        for task_name in sorted_tasks:
            if task_name not in target_positions:
                continue
                
            target = target_positions[task_name]
            target_tensor = torch.tensor(target, device=self.device, dtype=torch.float32)
            
            # Get task Jacobian and current task state
            J, current = self._get_task_jacobian_and_state(state, task_name)
            
            # Ensure consistent tensor types
            J = J.to(dtype=torch.float32)
            current = current.to(dtype=torch.float32)
            
            # Compute the task error
            error = target_tensor - current
            
            # Project into the null space of higher priority tasks
            if null_space is not None:
                J_projected = torch.bmm(J, null_space)
            else:
                J_projected = J
            
            # Compute pseudoinverse
            J_pinv = self._compute_pinv(J_projected)
            
            # Compute joint commands for this task
            delta_q = torch.bmm(J_pinv, error.unsqueeze(-1)).squeeze(-1)
            
            # Update joint commands
            joint_commands += delta_q
            
            # Update null space projector for lower priority tasks
            I = torch.eye(J.shape[2], device=self.device).unsqueeze(0).repeat(num_envs, 1, 1)
            null_space_proj = I - torch.bmm(J_pinv, J)
            null_space = torch.bmm(null_space_proj, null_space)
        
        # Apply collision avoidance
        joint_commands = self._apply_collision_avoidance(state, joint_commands)
        
        # Apply joint limits
        joint_commands = torch.max(torch.min(joint_commands, self.joint_limits_upper), self.joint_limits_lower)
        
        return joint_commands
    
    def _get_task_jacobian_and_state(self, state: EnvState, task_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the Jacobian and current state for a specific task."""
        num_envs = state.dof_pos.shape[0]
        
        if task_name.startswith("ee_pos_"):
            # End-effector position task
            link_name = task_name[7:]  # Extract link name from task name
            link_idx = self._get_link_idx_from_name(link_name)
            
            # Get Jacobian for end-effector position
            J_np = np.zeros((num_envs, 3, len(self.joint_indices)))
            current_np = np.zeros((num_envs, 3))
            
            for env_idx in range(num_envs):
                # Get current end-effector position
                link_state = p.getLinkState(self.robot_id, link_idx)
                current_np[env_idx] = link_state[0]  # world position
                
                # Compute Jacobian
                zero_vec = [0.0] * len(self.joint_indices)
                jac_t, jac_r = p.calculateJacobian(
                    self.robot_id, link_idx, 
                    localPosition=[0, 0, 0], 
                    objPositions=state.dof_pos[env_idx].cpu().numpy().tolist(),
                    objVelocities=zero_vec, 
                    objAccelerations=zero_vec
                )
                J_np[env_idx] = np.array(jac_t)  # translational Jacobian
            
            # Convert to torch tensors with explicit dtype
            J = torch.tensor(J_np, device=self.device, dtype=torch.float32)
            current = torch.tensor(current_np, device=self.device, dtype=torch.float32)
            
            return J, current
            
        elif task_name.startswith("ee_orn_"):
            # End-effector orientation task
            link_name = task_name[7:]  # Extract link name from task name
            link_idx = self._get_link_idx_from_name(link_name)
            
            # Get Jacobian for end-effector orientation
            J_np = np.zeros((num_envs, 3, len(self.joint_indices)))
            current_np = np.zeros((num_envs, 3))
            
            for env_idx in range(num_envs):
                # Get current end-effector orientation
                link_state = p.getLinkState(self.robot_id, link_idx)
                quat = link_state[1]  # world orientation (quaternion)
                
                # Convert quaternion to axis-angle representation
                rot = Rotation.from_quat([quat[0], quat[1], quat[2], quat[3]])
                current_np[env_idx] = rot.as_rotvec()  # axis-angle
                
                # Compute Jacobian
                zero_vec = [0.0] * len(self.joint_indices)
                jac_t, jac_r = p.calculateJacobian(
                    self.robot_id, link_idx, 
                    localPosition=[0, 0, 0], 
                    objPositions=state.dof_pos[env_idx].cpu().numpy().tolist(),
                    objVelocities=zero_vec, 
                    objAccelerations=zero_vec
                )
                J_np[env_idx] = np.array(jac_r)  # rotational Jacobian
            
            # Convert to torch tensors with explicit dtype
            J = torch.tensor(J_np, device=self.device, dtype=torch.float32)
            current = torch.tensor(current_np, device=self.device, dtype=torch.float32)
            
            return J, current
            
        elif task_name == "joint_pos":
            # Joint position task
            num_dof = len(self.joint_indices)
            J = torch.eye(num_dof, device=self.device, dtype=torch.float32).unsqueeze(0).repeat(num_envs, 1, 1)
            current = state.dof_pos.to(dtype=torch.float32)
            
            return J, current
        
        else:
            raise ValueError(f"Unknown task: {task_name}")
    
    def _get_link_idx_from_name(self, link_name: str) -> int:
        """Get link index from name."""
        for i in range(p.getNumJoints(self.robot_id)):
            joint_info = p.getJointInfo(self.robot_id, i)
            if joint_info[12].decode('utf-8') == link_name:
                return i
        raise ValueError(f"Link not found: {link_name}")
    
    def _compute_pinv(self, J: torch.Tensor, damping: float = 0.0001) -> torch.Tensor:
        """Compute damped pseudoinverse of Jacobian."""
        # J has shape (num_envs, task_dim, dof_dim)
        batch_size, m, n = J.shape
        
        # Ensure consistent tensor types
        J = J.to(dtype=torch.float32)
        
        # Damped pseudoinverse: J^T * (J * J^T + λI)^-1
        JJT = torch.bmm(J, J.transpose(1, 2))
        
        # Add damping
        damping_matrix = damping * torch.eye(m, device=self.device, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1)
        JJT_damped = JJT + damping_matrix
        
        # Compute inverse
        JJT_inv = torch.inverse(JJT_damped)
        
        # Compute pseudoinverse
        J_pinv = torch.bmm(J.transpose(1, 2), JJT_inv)
        
        return J_pinv
    
    def _apply_collision_avoidance(self, state: EnvState, joint_commands: torch.Tensor) -> torch.Tensor:
        """Apply collision avoidance to joint commands."""
        # Use the collision utility function
        return modify_control_for_collision_avoidance(
            state=state,
            control=joint_commands,
            collision_margin=self.collision_margin,
            joint_indices=self.joint_indices,
            gain=1.0  # Lower gain for stable behavior
        )