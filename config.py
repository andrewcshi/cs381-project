# config.py

# Training hyperparams
TRAINING = {
    'lr': 0.0001,                    # Learning rate
    'batch_size': 64,                # Batch size for training
    'gamma': 0.9,                    # Discount factor
    'eps_decay': 50000,              # Decay rate for epsilon
    'eps_start': 0.5,                # Starting value of epsilon
    'eps_end': 0.1,                  # Minimum value of epsilon
    'initial_memory': 10000,         # Initial replay memory size before training starts
    'memory_size': 100000,           # Maximum replay memory size
    'num_episodes': 500,             # Default number of episodes to train for
    'update_frequency': 4,           # How often to update the network
    'save_frequency': 50,            # How often to save the model (in episodes)
    'log_frequency': 1,              # How often to log training info (in episodes)
    'plot_frequency': 10,            # How often to update plots (in episodes)
    'detailed_log_frequency': 10,    # How often to print detailed stats (in episodes)
    'terminal_output_frequency': 5,  # How often to print directly to terminal (in episodes)
    'progress_bar_length': 30,       # Length of the progress bar in characters
}

# Model hyperparams
MODEL = {
    'robot_dim': 6,                  # Robot state dimension (2D)
    'robot_dim_3d': 9,               # Robot state dimension (3D)
    'obstacle_dim': 7,               # Obstacle state dimension (2D)
    'obstacle_dim_3d': 10,           # Obstacle state dimension (3D)
    'hidden_dim': 48,                # Hidden dimension for both LSTM and Transformer
    'num_actions': 80,               # Number of discrete actions (2D)
    'num_actions_3d': 160,           # Number of discrete actions (3D)
    
    # LSTM-specific parameters
    'lstm': {
        'hidden_dim': 48,            # LSTM hidden state dimension
    },
    
    # Transformer-specific parameters
    'transformer': {
        'hidden_dim': 48,            # Transformer hidden dimension
        'nhead': 4,                  # Number of attention heads
        'num_layers': 2,             # Number of transformer layers
        'dropout': 0.1,              # Dropout rate
        'max_seq_len': 20,           # Maximum sequence length for positional encoding
    }
}

# Environment hyperparameters
ENVIRONMENT = {
    'obstacle_num': 5,               # Default number of obstacles
    'layout': 'circle',              # Default obstacle layout
    'time_out_duration': 25.0,       # Maximum episode duration in seconds
    'moving_obstacle_ratio': 0.8,    # Ratio of obstacles that move
    'obstacle_velocity_scale': 0.3,  # Velocity scale factor for obstacles
    'curriculum_learning': True,     # Whether to use curriculum learning
    'curriculum_phases': [           # Phases for curriculum learning
        {'moving_ratio': 0.2, 'velocity_scale': 0.2, 'episodes': 100},  # Phase 1: Few slow-moving obstacles
        {'moving_ratio': 0.5, 'velocity_scale': 0.3, 'episodes': 200},  # Phase 2: Half obstacles moving at medium speed
        {'moving_ratio': 0.8, 'velocity_scale': 0.4, 'episodes': 200},  # Phase 3: Most obstacles moving at higher speed
    ],
    'dimension': '2D',               # Dimension of the environment: '2D' or '3D'
    '3d': {
        'height_limit': 5.0,         # Limit for z-coordinate
        'enable_flying': True,       # Whether obstacles can move in z-direction
        'z_velocity_scale': 0.2,     # Scale for z-direction velocity (typically lower than x,y)
    },
}

# Paths for saving models and figures
PATHS = {
    'base_model_path': 'weights',    # Base directory for model weights
    'figures_path': 'figures',       # Directory for saving figures
    'eval_path': 'evaluation',       # Directory for evaluation results
    'log_path': 'logs',              # Directory for log files
}