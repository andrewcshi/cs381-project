import os
import torch
import numpy as np
from env import ObstacleEnv
from models import create_model
from config import MODEL, ENVIRONMENT, PATHS

def debug_single_episode(model_type, dimension):
    """Run a single episode with detailed debugging."""
    print(f"\n===== Debugging {model_type} model in {dimension} environment =====")
    
    # Create environment with the specified dimension
    env = ObstacleEnv(dimension=dimension)
    
    # Create model
    model = create_model(model_type, dimension)
    
    # Try to load a trained model
    model_path = f"{PATHS['base_model_path']}/{model_type}/"
    model_loaded = False
    
    # Check multiple possible model files
    possible_model_files = [
        os.path.join(model_path, f"{model_type}_{dimension}_final.pth"),
        os.path.join(model_path, f"{model_type}_final.pth"),
        os.path.join(model_path, f"{model_type}_{dimension}_ep_499.pth"),
        os.path.join(model_path, f"{model_type}_ep_499.pth")
    ]
    
    for model_file in possible_model_files:
        if os.path.exists(model_file):
            try:
                model.load_state_dict(torch.load(model_file))
                print(f"Loaded model from {model_file}")
                model_loaded = True
                break
            except Exception as e:
                print(f"Error loading model from {model_file}: {e}")
                continue
    
    if not model_loaded:
        print(f"No model file found. Using untrained model.")
    
    model.eval()  # Set to evaluation mode
    
    # Reset environment
    print("Resetting environment...")
    obs = env.reset(
        obstacle_num=5,
        layout='circle',
        test_phase=True,
        moving_obstacle_ratio=0.8,
        obstacle_velocity_scale=0.3,
        dimension=dimension
    )
    
    print(f"Number of obstacles: {len(env.obstacle_list)}")
    print(f"Moving obstacles: {sum(1 for obs in env.obstacle_list if obs.v_pref > 0)}")
    
    # Get initial state
    state = env.convert_coord(obs)
    print(f"State shape: {state.shape}")
    
    # Print robot and goal positions
    print(f"Robot position: {env.robot.get_position()}")
    print(f"Goal position: {env.robot.get_goal_position()}")
    
    # Run a single episode
    done = False
    steps = 0
    total_reward = 0
    
    print("\nStarting episode simulation:")
    while not done and steps < 100:  # Limit to 100 steps to avoid infinite loops
        # Get action from model
        with torch.no_grad():
            q_values = model(state)
            action_idx = torch.argmax(q_values).item()
        
        # Get the actual velocity action
        vel_action = env.vel_samples[action_idx]
        
        # Log action
        print(f"Step {steps+1}: Action {action_idx}, Velocity: {vel_action}")
        
        # Take the action
        next_obs, reward, done, info = env.step(vel_action)
        
        # Calculate distance to goal
        if dimension == '3D':
            distance = np.linalg.norm(np.array([env.robot.px, env.robot.py, env.robot.pz]) - 
                                    np.array([env.robot.gx, env.robot.gy, env.robot.gz]))
        else:
            distance = np.linalg.norm(np.array([env.robot.px, env.robot.py]) - 
                                    np.array([env.robot.gx, env.robot.gy]))
        
        # Log results
        print(f"  Position: {env.robot.get_position()}")
        print(f"  Distance to goal: {distance:.4f}")
        print(f"  Reward: {reward:.4f}, Done: {done}, Info: {info}")
        
        # Update state
        state = env.convert_coord(next_obs)
        total_reward += reward
        steps += 1
        
        if done:
            print(f"\nEpisode terminated after {steps} steps")
            print(f"Final reward: {total_reward:.4f}")
            print(f"Outcome: {info}")
            
            if "Goal reached" in info:
                print("SUCCESS: Agent reached the goal!")
            elif info == "collision":
                print("FAILURE: Agent collided with an obstacle")
            elif info == "timeout":
                print("FAILURE: Episode timed out")
            else:
                print(f"UNKNOWN OUTCOME: {info}")
    
    # If the episode didn't terminate
    if not done:
        print(f"\nEpisode did not terminate after {steps} steps")
        print(f"Final reward: {total_reward:.4f}")
        print(f"Final distance to goal: {distance:.4f}")
    
    print("===== Debugging complete =====\n")

def main():
    """Main debugging function."""
    print("===== Evaluation Debugging Tool =====")
    
    # Test all model and dimension combinations
    for model_type in ['lstm', 'transformer']:
        for dimension in ['2D', '3D']:
            try:
                debug_single_episode(model_type, dimension)
            except Exception as e:
                print(f"Error debugging {model_type} in {dimension}: {e}")
    
    print("Debugging complete.")

if __name__ == "__main__":
    main()