# test_moving_obstacles.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle
import argparse
from env import ObstacleEnv
from agent import Robot, Obstacle
from config import ENVIRONMENT

def visualize_obstacle_movement(env, episodes=1, steps_per_episode=100):
    """Visualize obstacle movement without a robot controller."""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    
    # Store obstacle trajectories
    trajectories = []
    obstacle_patches = []
    
    # Create a container for obstacle positions over time
    obstacle_positions = []
    
    # Initialize environment
    obs = env.reset(
        obstacle_num=ENVIRONMENT['obstacle_num'],
        layout='circle',
        moving_obstacle_ratio=0.8,  # Most obstacles move
        obstacle_velocity_scale=0.4  # Medium speed
    )
    
    # Initialize trajectories for each obstacle
    for obstacle in env.obstacle_list:
        trajectories.append([])
        # Create initial circles for each obstacle
        circle = Circle(
            (obstacle.px, obstacle.py), 
            obstacle.radius, 
            fill=True, 
            alpha=0.7,
            color='blue' if obstacle.v_pref > 0 else 'gray'
        )
        obstacle_patches.append(ax.add_patch(circle))
    
    # Dummy robot action (straight toward goal)
    robot_action = [0, 0]
    
    # Run simulation and collect obstacle positions
    for _ in range(steps_per_episode):
        # Record positions for each obstacle
        current_positions = []
        for i, obstacle in enumerate(env.obstacle_list):
            current_positions.append((obstacle.px, obstacle.py, obstacle.radius))
            trajectories[i].append((obstacle.px, obstacle.py))
        obstacle_positions.append(current_positions)
        
        # Take a step in the environment
        next_obs, reward, done, info = env.step(robot_action)
        
        if done:
            break
    
    # Draw robot and goal
    robot_patch = Circle(
        env.robot.get_position(),
        env.robot.radius,
        fill=True,
        color='red',
        alpha=0.7
    )
    ax.add_patch(robot_patch)
    
    goal_patch = Circle(
        env.robot.get_goal_position(),
        0.2,
        fill=True,
        color='green',
        alpha=0.7
    )
    ax.add_patch(goal_patch)
    
    # Add legend
    static_circle = Circle((0, 0), 0.3, fill=True, color='gray', alpha=0.7)
    moving_circle = Circle((0, 0), 0.3, fill=True, color='blue', alpha=0.7)
    robot_circle = Circle((0, 0), 0.3, fill=True, color='red', alpha=0.7)
    goal_circle = Circle((0, 0), 0.3, fill=True, color='green', alpha=0.7)
    
    ax.legend(
        [static_circle, moving_circle, robot_circle, goal_circle],
        ['Static Obstacle', 'Moving Obstacle', 'Robot', 'Goal'],
        loc='upper right'
    )
    
    # Helper function to draw obstacle trajectories
    def draw_trajectories():
        for i, traj in enumerate(trajectories):
            # Only draw trajectories for moving obstacles
            if env.obstacle_list[i].v_pref > 0:
                x_vals = [p[0] for p in traj]
                y_vals = [p[1] for p in traj]
                ax.plot(x_vals, y_vals, 'b-', alpha=0.3)
    
    # Animation update function
    def update(frame):
        # Draw trajectories
        draw_trajectories()
        
        # Update obstacle positions
        if frame < len(obstacle_positions):
            positions = obstacle_positions[frame]
            for i, (px, py, radius) in enumerate(positions):
                obstacle_patches[i].center = (px, py)
                
        # Add frame counter
        ax.set_title(f'Frame: {frame}/{len(obstacle_positions)-1}')
        return obstacle_patches
    
    # Create animation
    ani = animation.FuncAnimation(
        fig, 
        update, 
        frames=len(obstacle_positions),
        interval=100,  # milliseconds between frames
        blit=False,
        repeat=True
    )
    
    # Display the animation
    ax.set_title('Obstacle Movement Simulation')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Print obstacle movement types
    print("\nObstacle Movement Types:")
    for i, obstacle in enumerate(env.obstacle_list):
        if obstacle.v_pref > 0:
            print(f"Obstacle {i+1}: {obstacle.movement_type.capitalize()}, Speed: {obstacle.v_pref:.2f}")
        else:
            print(f"Obstacle {i+1}: Static")
    
    plt.tight_layout()
    plt.show()

def test_moving_obstacle_consistency(steps=100, trials=10):
    """Test if obstacle movements are consistent and working properly."""
    env = ObstacleEnv()
    
    # Test with different environments
    for trial in range(trials):
        print(f"\nTrial {trial+1}/{trials}")
        
        # Reset environment
        obs = env.reset(
            obstacle_num=5,
            layout='circle',
            moving_obstacle_ratio=0.8,
            obstacle_velocity_scale=0.4
        )
        
        # Print initial obstacle info
        print("Obstacle Info:")
        for i, obstacle in enumerate(env.obstacle_list):
            print(f"  Obstacle {i+1}: {'Moving' if obstacle.v_pref > 0 else 'Static'}, " + 
                  f"Type: {obstacle.movement_type if obstacle.v_pref > 0 else 'N/A'}, " +
                  f"Speed: {obstacle.v_pref:.2f}")
        
        # Run simulation for several steps
        initial_positions = [(obs.px, obs.py) for obs in env.obstacle_list]
        
        # Placeholder robot action
        robot_action = [0, 0]
        
        # Track if obstacles actually move
        position_changes = [False] * len(env.obstacle_list)
        collision_occurred = False
        
        for step in range(steps):
            next_obs, reward, done, info = env.step(robot_action)
            
            # Check if positions changed
            for i, obstacle in enumerate(env.obstacle_list):
                if (obstacle.px, obstacle.py) != initial_positions[i]:
                    position_changes[i] = True
            
            if done:
                if info == "collision":
                    collision_occurred = True
                    print(f"  Collision detected at step {step+1}")
                break
        
        # Report results
        moving_count = sum(1 for obs in env.obstacle_list if obs.v_pref > 0)
        actually_moved = sum(position_changes)
        
        print(f"  Expected moving obstacles: {moving_count}")
        print(f"  Obstacles that actually moved: {actually_moved}")
        
        if moving_count > 0 and actually_moved == 0:
            print("  WARNING: Moving obstacles didn't move!")
        elif moving_count != actually_moved and not collision_occurred:
            print("  WARNING: Mismatch between expected and actual moving obstacles")
        else:
            print("  SUCCESS: Movement behavior is as expected")
            
def main():
    parser = argparse.ArgumentParser(description='Test moving obstacles')
    parser.add_argument('--visual', action='store_true', help='Run visualization')
    parser.add_argument('--test', action='store_true', help='Run consistency test')
    parser.add_argument('--obstacle-num', type=int, default=5, help='Number of obstacles')
    parser.add_argument('--steps', type=int, default=100, help='Number of steps to simulate')
    args = parser.parse_args()
    
    env = ObstacleEnv()
    
    if args.visual:
        print("Running visualization of moving obstacles...")
        visualize_obstacle_movement(env, steps_per_episode=args.steps)
    
    if args.test:
        print("Running consistency test for moving obstacles...")
        test_moving_obstacle_consistency(steps=args.steps)
        
    if not args.visual and not args.test:
        # Default behavior if no flags are specified
        print("Running both visualization and consistency test...")
        test_moving_obstacle_consistency(steps=args.steps)
        visualize_obstacle_movement(env, steps_per_episode=args.steps)

if __name__ == "__main__":
    main()