import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pybullet as p
import torch
from scipy.spatial.transform import Rotation

from state import EnvState


def create_collision_shape_array(
    robot_id: int, 
    link_indices: Optional[List[int]] = None,
    simplify: bool = True,
    visualization: bool = False
) -> Dict[int, int]:
    """
    Create collision shapes for the robot links.
    
    Args:
        robot_id: PyBullet body ID for the robot
        link_indices: Indices of links to create collision shapes for (None for all)
        simplify: Whether to simplify collision shapes for better performance
        visualization: Whether to visualize the collision shapes
        
    Returns:
        Dictionary mapping link indices to collision shape IDs
    """
    collision_shapes = {}
    
    # If no link indices provided, use all links
    if link_indices is None:
        link_indices = list(range(p.getNumJoints(robot_id)))
    
    for link_idx in link_indices:
        # Get collision shape data
        collision_shapes[link_idx] = _create_link_collision_shape(
            robot_id, link_idx, simplify, visualization
        )
    
    return collision_shapes


def _create_link_collision_shape(
    robot_id: int, 
    link_idx: int, 
    simplify: bool = True,
    visualization: bool = False
) -> int:
    """Create a collision shape for a specific link."""
    # Get link dynamics info
    dynamics_info = p.getDynamicsInfo(robot_id, link_idx)
    collision_shape_type = dynamics_info[2]  # collision shape type
    
    # If no collision shape, return -1
    if collision_shape_type == p.GEOM_MESH and dynamics_info[3] == -1:
        return -1
    
    # Get visual shape data to extract geometry
    visual_data = p.getVisualShapeData(robot_id, link_idx)
    if not visual_data:
        return -1
    
    # Create a simplified collision shape based on type
    if simplify:
        collision_id = _create_simplified_collision_shape(
            robot_id, link_idx, visual_data, visualization
        )
    else:
        # Use the original collision shape
        collision_id = dynamics_info[3]  # original collision shape ID
    
    return collision_id


def _create_simplified_collision_shape(
    robot_id: int, 
    link_idx: int, 
    visual_data: List,
    visualization: bool = False
) -> int:
    """Create a simplified collision shape based on the visual shape."""
    # Extract data from visual shape
    shape_type = visual_data[0][2]  # shape type
    dimensions = visual_data[0][3]  # dimensions
    local_pos = visual_data[0][5]  # local position
    local_orn = visual_data[0][6]  # local orientation
    
    # Create different shapes based on the visual shape type
    collision_id = -1
    
    if shape_type == p.GEOM_BOX:
        # Create box collision shape
        collision_id = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=dimensions,
        )
    elif shape_type == p.GEOM_SPHERE:
        # Create sphere collision shape
        radius = dimensions[0]
        collision_id = p.createCollisionShape(
            p.GEOM_SPHERE,
            radius=radius,
        )
    elif shape_type == p.GEOM_CYLINDER:
        # Create cylinder collision shape
        radius = dimensions[0]
        height = dimensions[1]
        collision_id = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=radius,
            height=height,
        )
    elif shape_type == p.GEOM_CAPSULE:
        # Create capsule collision shape
        radius = dimensions[0]
        height = dimensions[1]
        collision_id = p.createCollisionShape(
            p.GEOM_CAPSULE,
            radius=radius,
            height=height,
        )
    else:
        # For mesh or other types, create a bounding box
        aabb_min, aabb_max = p.getAABB(robot_id, link_idx)
        half_extents = [(aabb_max[i] - aabb_min[i]) / 2 for i in range(3)]
        center = [(aabb_max[i] + aabb_min[i]) / 2 for i in range(3)]
        
        # Create box collision shape
        collision_id = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
        )
    
    # Visualize collision shape if requested
    if visualization and collision_id != -1:
        # Create a visual body at the link position with the collision shape
        link_state = p.getLinkState(robot_id, link_idx)
        link_pos = link_state[0]
        link_orn = link_state[1]
        
        # Combine with local position and orientation
        final_pos, final_orn = p.multiplyTransforms(
            link_pos, link_orn,
            local_pos, local_orn
        )
        
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=-1,
            basePosition=final_pos,
            baseOrientation=final_orn,
            linkMasses=[0],
            linkCollisionShapeIndices=[collision_id],
            linkVisualShapeIndices=[-1],
            linkPositions=[[0, 0, 0]],
            linkOrientations=[[0, 0, 0, 1]],
            linkInertialFramePositions=[[0, 0, 0]],
            linkInertialFrameOrientations=[[0, 0, 0, 1]],
            linkParentIndices=[0],
            linkJointTypes=[p.JOINT_FIXED],
            linkJointAxis=[[0, 0, 0]],
        )
    
    return collision_id


def compute_collision_distances(
    robot_id: int,
    collision_pairs: List[Tuple[int, int]],
    max_distance: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute distances and nearest points for collision pairs.
    
    Args:
        robot_id: PyBullet body ID for the robot
        collision_pairs: List of (link_a, link_b) tuples to check
        max_distance: Maximum distance to check
        
    Returns:
        Tuple of (distances, nearest_points_a, nearest_points_b)
    """
    distances = []
    nearest_points_a = []
    nearest_points_b = []
    
    for link_a, link_b in collision_pairs:
        # Get closest points between links
        closest_points = p.getClosestPoints(
            robot_id, robot_id,
            distance=max_distance,
            linkIndexA=link_a if link_a >= 0 else -1,
            linkIndexB=link_b if link_b >= 0 else -1
        )
        
        if closest_points:
            # Sort by distance
            closest_points.sort(key=lambda x: x[8])
            
            # Get the closest point
            closest = closest_points[0]
            distances.append(closest[8])
            nearest_points_a.append(closest[5])
            nearest_points_b.append(closest[6])
        else:
            # No collision detected within the threshold
            distances.append(max_distance)
            # Use dummy values for points
            nearest_points_a.append([0, 0, 0])
            nearest_points_b.append([0, 0, 0])
    
    return np.array(distances), np.array(nearest_points_a), np.array(nearest_points_b)


def get_collision_info(
    state: EnvState,
    max_distance: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get collision information for the current state.
    
    Args:
        state: Current environment state
        max_distance: Maximum distance to check for collisions
        
    Returns:
        Tuple of (distances, nearest_points)
    """
    # Compute distances for each collision pair
    distances = []
    nearest_points = []
    
    for i, (link_a, link_b) in enumerate(state.collision_pairs):
        # Get closest points between the two links
        result = p.getClosestPoints(
            state.robot_id, state.robot_id,
            distance=max_distance,
            linkIndexA=link_a if link_a >= 0 else -1,
            linkIndexB=link_b if link_b >= 0 else -1
        )
        
        if result:
            # Take the closest point
            closest = min(result, key=lambda x: x[8])  # x[8] is the distance
            distances.append(closest[8])
            nearest_points.append([closest[5], closest[6]])  # positionOnA, positionOnB
        else:
            # No collision detected within the distance threshold
            distances.append(max_distance)
            # Use dummy values for points
            nearest_points.append([[0, 0, 0], [0, 0, 0]])
    
    # Convert to torch tensors
    distances_tensor = torch.tensor(distances, device=state.device).unsqueeze(0)
    nearest_points_tensor = torch.tensor(nearest_points, device=state.device).unsqueeze(0)
    
    return distances_tensor, nearest_points_tensor


def compute_collision_jacobian(
    robot_id: int,
    link_idx: int,
    joint_indices: List[int],
    collision_point: List[float]
) -> np.ndarray:
    """
    Compute the collision Jacobian for a specific link and point.
    
    Args:
        robot_id: PyBullet body ID
        link_idx: Index of the link with the collision point
        joint_indices: Indices of joints to include in the Jacobian
        collision_point: 3D point where collision occurs (world frame)
        
    Returns:
        Jacobian matrix (3 x num_joints)
    """
    # Initialize Jacobian matrix
    num_joints = len(joint_indices)
    jacobian = np.zeros((3, num_joints))
    
    # Get the world position of the collision point
    collision_point = np.array(collision_point)
    
    # For each joint, compute the Jacobian column
    for i, joint_idx in enumerate(joint_indices):
        # Get the joint axis in world frame
        # Skip if the joint doesn't affect this link
        if not does_joint_affect_link(robot_id, joint_idx, link_idx):
            continue
        
        # Get joint info
        joint_info = p.getJointInfo(robot_id, joint_idx)
        joint_type = joint_info[2]
        
        if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
            # Get joint position and orientation in world frame
            joint_state = p.getLinkState(robot_id, joint_idx)
            joint_pos = np.array(joint_state[0])
            joint_orn = np.array(joint_state[1])
            
            if joint_type == p.JOINT_REVOLUTE:
                # For revolute joints, J = axis × (point - joint_pos)
                joint_axis = get_joint_axis_world(robot_id, joint_idx)
                jacobian[:, i] = np.cross(joint_axis, collision_point - joint_pos)
            else:  # JOINT_PRISMATIC
                # For prismatic joints, J = axis
                joint_axis = get_joint_axis_world(robot_id, joint_idx)
                jacobian[:, i] = joint_axis
    
    return jacobian


def does_joint_affect_link(robot_id: int, joint_idx: int, link_idx: int) -> bool:
    """Check if a joint affects a link's motion."""
    # Get the joint's child link
    joint_info = p.getJointInfo(robot_id, joint_idx)
    joint_child = joint_info[16]
    
    # If the joint's child is the link we're checking, it affects it
    if joint_child == link_idx:
        return True
    
    # If the joint is in the chain from the base to the link, it affects it
    link_path = []
    current_link = link_idx
    
    while current_link != -1:
        link_path.append(current_link)
        # Get parent joint
        if current_link == 0:
            # Base link has no parent joint
            current_link = -1
        else:
            joint_info = p.getJointInfo(robot_id, current_link - 1)
            current_link = joint_info[16]
    
    return joint_idx in link_path or joint_idx < link_idx


def get_joint_axis_world(robot_id: int, joint_idx: int) -> np.ndarray:
    """Get joint axis in world frame."""
    # Get joint info
    joint_info = p.getJointInfo(robot_id, joint_idx)
    joint_axis_local = np.array(joint_info[13])
    
    # Get joint state
    joint_state = p.getLinkState(robot_id, joint_idx)
    joint_orn = np.array(joint_state[1])
    
    # Convert quaternion to rotation matrix
    rot = Rotation.from_quat([joint_orn[0], joint_orn[1], joint_orn[2], joint_orn[3]])
    
    # Rotate joint axis to world frame
    joint_axis_world = rot.apply(joint_axis_local)
    
    return joint_axis_world


def modify_control_for_collision_avoidance(
    state: EnvState,
    control: torch.Tensor,
    collision_margin: float,
    joint_indices: List[int],
    gain: float = 10.0
) -> torch.Tensor:
    """
    Modify control signals to avoid collisions.
    
    Args:
        state: Current environment state
        control: Original control signal (num_envs, num_joints)
        collision_margin: Distance margin for collision avoidance
        joint_indices: Indices of controllable joints
        gain: Gain for the repulsive force
        
    Returns:
        Modified control signal
    """
    modified_control = control.clone()
    
    # Check each collision pair
    for env_idx in range(state.dof_pos.shape[0]):
        for pair_idx, (link_a, link_b) in enumerate(state.collision_pairs):
            # Skip if distance is safe
            distance = state.nearest_distances[env_idx, pair_idx].item()
            if distance >= collision_margin:
                continue
            
            # Get collision points
            point_a = state.nearest_points[env_idx, pair_idx, 0].cpu().numpy()
            point_b = state.nearest_points[env_idx, pair_idx, 1].cpu().numpy()
            
            # Compute collision normal (points away from collision)
            collision_normal = point_b - point_a
            if np.linalg.norm(collision_normal) > 0:
                collision_normal = collision_normal / np.linalg.norm(collision_normal)
            else:
                # Skip if normal is zero (shouldn't happen)
                continue
            
            # Compute repulsive magnitude based on proximity
            repulsive_magnitude = gain * (collision_margin - distance) / collision_margin
            
            # Compute Jacobians for both links
            jacobian_a = compute_collision_jacobian(
                state.robot_id, link_a, joint_indices, point_a
            )
            jacobian_b = compute_collision_jacobian(
                state.robot_id, link_b, joint_indices, point_b
            )
            
            # Compute control modification to move away from collision
            # For link_a, we want to move in the direction of the normal
            # For link_b, we want to move in the opposite direction
            control_mod_a = np.dot(jacobian_a.T, collision_normal) * repulsive_magnitude
            control_mod_b = -np.dot(jacobian_b.T, -collision_normal) * repulsive_magnitude
            
            # Apply the modifications
            for i, joint_idx in enumerate(joint_indices):
                joint_mod = control_mod_a[i] + control_mod_b[i]
                modified_control[env_idx, i] += torch.tensor(joint_mod, device=control.device)
    
    return modified_control