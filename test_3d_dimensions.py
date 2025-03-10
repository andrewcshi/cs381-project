#!/usr/bin/env python
# Debug script to test 3D dimensions

import os
import sys
import torch
import numpy as np
from env import ObstacleEnv
from models import create_model
from config import MODEL, ENVIRONMENT

# Enable debug dimension logging in environment
# This needs to be added to the ObstacleEnv class
def enable_debug(env):
    env.debug_dims = True
    return env

def main():
    print("Testing 3D model dimensions")
    print("--------------------------")
    
    # Create 3D environment
    env = ObstacleEnv(dimension='3D')
    env = enable_debug(env)
    
    # Initialize environment with obstacles
    obs = env.reset(
        obstacle_num=5, 
        layout='circle', 
        test_phase=False, 
        counter=None,
        moving_obstacle_ratio=0.2,
        obstacle_velocity_scale=0.2
    )
    
    print(f"Environment dimension: {env.dimension}")
    print(f"Number of observations: {len(obs)}")
    print(f"Robot state shape: {len(obs[0])}")
    
    # Convert observations to tensor format
    state = env.convert_coord(obs)
    print(f"Converted state shape: {state.shape}")
    
    if state.shape[2] != 19:  # 5 (robot) + 14 (obstacle)
        print(f"ERROR: Expected state dimension 19 for 3D, got {state.shape[2]}")
        if state.shape[2] == 13:  # This is the 2D dimension (5+7+1)
            print("This suggests the convert_coord method is using 2D mode even though dimension is set to 3D")
    else:
        print("State dimensions for 3D are correct")
    
    # Test LSTM model
    print("\nTesting LSTM model:")
    lstm_model = create_model('lstm', '3D')
    print(f"LSTM input dimension: {lstm_model.obstacle_dim}")
    print(f"LSTM expected feature count: {lstm_model.robot_dim + lstm_model.obstacle_dim}")
    
    try:
        # Try to forward pass
        q_value = lstm_model(state)
        print(f"LSTM forward pass successful, output shape: {q_value.shape}")
        print(f"Expected action dimensions: {MODEL['num_actions_3d']}")
    except Exception as e:
        print(f"LSTM Error: {str(e)}")
    
    # Test Transformer model
    print("\nTesting Transformer model:")
    transformer_model = create_model('transformer', '3D')
    print(f"Transformer input dimension: {transformer_model.obstacle_dim}")
    print(f"Transformer expected feature count: {transformer_model.robot_dim + transformer_model.obstacle_dim}")
    
    try:
        # Try to forward pass
        q_value = transformer_model(state)
        print(f"Transformer forward pass successful, output shape: {q_value.shape}")
        print(f"Expected action dimensions: {MODEL['num_actions_3d']}")
    except Exception as e:
        print(f"Transformer Error: {str(e)}")
        
    print("\nDimension test complete")

if __name__ == "__main__":
    main()