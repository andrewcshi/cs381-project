import argparse
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pybullet as p
import torch

from collision_aware_env import CollisionAwareEnv


def run_example(robot_urdf: str, task: str = "circular", duration: int = 1000):
    """
    Run an example of collision-aware control.
    
    Args:
        robot_urdf: Path to robot URDF file
        task: Type of task to execute ('circular' or 'reaching')
        duration: Number of simulation steps
    """
    print(f"Running collision-aware control example with {robot_urdf}")
    print(f"Task: {task}, Duration: {duration} steps")
    
    # Create environment with GUI disabled
    env = CollisionAwareEnv(
        robot_urdf=robot_urdf,
        use_gui=False,  # Disable GUI to avoid X server issues
        dt=1/240.0,
        substeps=1,
        device="cpu",
        collision_margin=0.05
    )
    
    # Find end-effector link
    ee_link = None
    for i, name in enumerate(env.link_names):
        if "hand" in name.lower() or "gripper" in name.lower() or "tool" in name.lower() or "ee" in name.lower():
            ee_link = name
            break
    
    if ee_link is None:
        # If no end-effector found, use the last link
        ee_link = env.link_names[-1]
    
    print(f"Using {ee_link} as end-effector")
    
    # Add WBC tasks
    env.add_wbc_task(f"ee_pos_{ee_link}", priority=2)  # End-effector position
    env.add_wbc_task("joint_pos", priority=1)  # Joint positions
    
    # Reset environment
    state = env.reset()
    
    # Get initial end-effector position
    for i in range(len(env.link_indices)):
        if env.link_names[i] == ee_link:
            ee_pos = state.rigid_body_pos[0, i].cpu().numpy()
            break
    
    print(f"Initial end-effector position: {ee_pos}")
    
    # Generate trajectories based on task
    positions = []
    
    if task == "circular":
        # Generate circular trajectory
        center = ee_pos.copy()
        radius = 0.2
        
        for i in range(duration):
            angle = i * 2 * np.pi / duration
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            z = center[2]
            positions.append([x, y, z])
    
    elif task == "reaching":
        # Generate random reaching points
        num_points = 5
        reaches = []
        
        # Generate random points around initial position
        for _ in range(num_points):
            offset = np.random.uniform(-0.3, 0.3, size=3)
            target = ee_pos + offset
            reaches.append(target)
        
        # Create trajectory with smooth transitions
        steps_per_reach = duration // num_points
        
        for i in range(num_points):
            start = ee_pos if i == 0 else reaches[i-1]
            end = reaches[i]
            
            for t in range(steps_per_reach):
                alpha = t / steps_per_reach
                # Smooth interpolation
                blend = 3 * alpha**2 - 2 * alpha**3  # Smoothstep function
                pos = start * (1 - blend) + end * blend
                positions.append(pos)
    
    else:
        raise ValueError(f"Unknown task: {task}")
    
    # Ensure we have enough positions
    while len(positions) < duration:
        positions.append(positions[-1])
    
    # Track performance metrics
    min_distances = []
    position_errors = []
    violations_count = 0
    
    # Run simulation
    for i in range(duration):
        # Get target position for this step
        target_pos = positions[i]
        
        # Create targets dictionary
        targets = {
            f"ee_pos_{ee_link}": np.array([target_pos]),
            "joint_pos": state.dof_pos.cpu().numpy()  # Current joint positions as targets
        }
        
        # Get joint commands from WBC
        joint_cmds = env.set_wbc_targets(targets)
        
        # Step environment
        state, info = env.step(joint_cmds)
        
        # Print detailed info more frequently
        if i % 50 == 0:
            # Extract collision info
            min_dist = info.get("collision/min_distance", torch.tensor([1.0]))[0].item()
            avg_dist = info.get("collision/avg_distance", torch.tensor([1.0]))[0].item()
            
            # Extract current end-effector position
            for j in range(len(env.link_indices)):
                if env.link_names[j] == ee_link:
                    current_ee_pos = state.rigid_body_pos[0, j].cpu().numpy()
                    break
            
            # Calculate position error
            pos_error = np.linalg.norm(current_ee_pos - target_pos)
            
            # Extract constraint violations if any
            violation = info.get("collision/violation", torch.tensor([False]))[0].item()
            
            # Print comprehensive status
            print(f"Step {i}/{duration}:")
            print(f"  Target position: {target_pos}")
            print(f"  Current position: {current_ee_pos}")
            print(f"  Position error: {pos_error:.4f}")
            print(f"  Collision info:")
            print(f"    Min distance: {min_dist:.4f}")
            print(f"    Avg distance: {avg_dist:.4f}")
            print(f"    Violation: {violation}")
            print("------------------------------")
            
        # Collect metrics for summary
        min_dist = info.get("collision/min_distance", torch.tensor([1.0]))[0].item()
        min_distances.append(min_dist)
        
        # Calculate position error
        for j in range(len(env.link_indices)):
            if env.link_names[j] == ee_link:
                current_ee_pos = state.rigid_body_pos[0, j].cpu().numpy()
                break
        pos_error = np.linalg.norm(current_ee_pos - target_pos)
        position_errors.append(pos_error)
        
        # Track violations
        violation = info.get("collision/violation", torch.tensor([False]))[0].item()
        if violation:
            violations_count += 1
    
    # Print summary statistics
    print("\n===== SIMULATION SUMMARY =====")
    print(f"Total steps: {duration}")
    print(f"Position tracking:")
    print(f"  Average error: {np.mean(position_errors):.4f}")
    print(f"  Max error: {np.max(position_errors):.4f}")
    print(f"  Min error: {np.min(position_errors):.4f}")
    print(f"Collision avoidance:")
    print(f"  Minimum distance ever: {np.min(min_distances):.4f}")
    print(f"  Average minimum distance: {np.mean(min_distances):.4f}")
    print(f"  Violations count: {violations_count} ({violations_count/duration*100:.2f}%)")
    print("=============================")
    
    print("Example completed.")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run collision-aware control example")
    parser.add_argument("--robot", type=str, default="kuka_iiwa/model.urdf", 
                        help="Path to robot URDF file")
    parser.add_argument("--task", type=str, default="circular", choices=["circular", "reaching"],
                        help="Type of task to execute")
    parser.add_argument("--duration", type=int, default=1000, 
                        help="Number of simulation steps")
    
    args = parser.parse_args()
    
    # If the robot is a known robot, use the full path
    known_robots = {
        "panda": "franka_panda/panda.urdf",
        "kuka": "kuka_iiwa/model.urdf",
        "ur5": "ur5/urdf/ur5.urdf"
    }
    
    if args.robot in known_robots:
        # Use the known robot from PyBullet data
        robot_urdf = known_robots[args.robot]
    else:
        # Use the provided path
        robot_urdf = args.robot
    
    run_example(robot_urdf, args.task, args.duration)