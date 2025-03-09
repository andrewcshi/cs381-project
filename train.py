# train.py

import numpy as np
from collections import deque
import random
import torch
from torch import nn
import os
import time
import datetime
import logging
from env import ObstacleEnv
from models import LSTMNetwork, TransformerNetwork
from matplotlib import pyplot as plt
from IPython.display import clear_output
import argparse
from config import TRAINING, MODEL, ENVIRONMENT, PATHS

def setup_logger(model_type):
    """Set up a logger for the training process."""
    # Create log directory if it doesn't exist
    os.makedirs(PATHS['log_path'], exist_ok=True)
    
    # Set up logging format
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{PATHS['log_path']}/{model_type}_{timestamp}.log"
    
    # Configure logger
    logger = logging.getLogger(model_type)
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers (to prevent duplicates)
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s',
                                 datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Print banner to make terminal output more visible
    print("\n" + "="*80)
    print(f" TRAINING: {model_type.upper()} MODEL")
    print("="*80 + "\n")
    
    logger.info(f"Starting training for {model_type.upper()} model")
    logger.info(f"Log file: {log_file}")
    
    # Log configuration
    logger.info("Training configuration:")
    for key, value in TRAINING.items():
        logger.info(f"  {key}: {value}")
    
    logger.info(f"{model_type.upper()} model configuration:")
    if model_type == "lstm":
        for key, value in MODEL['lstm'].items():
            logger.info(f"  {key}: {value}")
    else:
        for key, value in MODEL['transformer'].items():
            logger.info(f"  {key}: {value}")
    
    logger.info("Environment configuration:")
    for key, value in ENVIRONMENT.items():
        logger.info(f"  {key}: {value}")
    
    return logger

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=TRAINING['progress_bar_length'], fill='█'):
    """
    Call in a loop to create a progress bar in the terminal.
    
    Args:
        iteration: Current iteration (Int)
        total: Total iterations (Int)
        prefix: Prefix string (Str)
        suffix: Suffix string (Str)
        decimals: Number of decimal places for percentage (Int)
        length: Character length of bar (Int)
        fill: Bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='', flush=True)
    # Print new line on complete
    if iteration == total:
        print()

def plot_stats(frame_idx, rewards, losses, step, model_type):
    """
    Plot training statistics with improved styling and smoothing.
    
    Args:
        frame_idx: Current frame index
        rewards: List of episode rewards
        losses: List of training losses
        step: Current training step
        model_type: Type of model being trained (lstm or transformer)
    """
    clear_output(True)
    
    # Apply smoothing with moving averages to reduce noise
    def smooth(data, window_size=10):
        if len(data) < window_size:
            return data
        smoothed = []
        for i in range(len(data)):
            start = max(0, i - window_size + 1)
            smoothed.append(sum(data[start:i+1]) / (i - start + 1))
        return smoothed
    
    # Create smoothed versions for plotting
    smoothed_rewards = smooth(rewards)
    smoothed_losses = smooth(losses, window_size=100) if len(losses) > 100 else smooth(losses)
    
    # Calculate statistics
    recent_rewards = rewards[-10:] if rewards else []
    avg_recent_reward = np.mean(recent_rewards) if recent_rewards else 0
    max_reward = max(rewards) if rewards else 0
    
    # Set up figure with custom styling
    plt.figure(figsize=(18, 8))
    plt.style.use('ggplot')
    
    # Reward plot
    plt.subplot(1, 2, 1)
    plt.plot(rewards, alpha=0.3, color='#1f77b4', label='Raw')
    plt.plot(smoothed_rewards, linewidth=2, color='#ff7f0e', label='Smoothed')
    
    # Add horizontal line for recent average
    if rewards:
        plt.axhline(y=avg_recent_reward, color='#2ca02c', linestyle='--', 
                   label=f'Recent Avg: {avg_recent_reward:.2f}')
    
    plt.title(f'{model_type.upper()} Training Rewards', fontsize=14, fontweight='bold')
    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Episode Reward', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add stats as text
    info_text = (
        f"Total Frames: {frame_idx}\n"
        f"Episodes: {len(rewards)}\n"
        f"Recent Avg Reward: {avg_recent_reward:.2f}\n"
        f"Max Reward: {max_reward:.2f}"
    )
    plt.annotate(info_text, xy=(0.02, 0.02), xycoords='axes fraction', 
                 bbox=dict(boxstyle="round,pad=0.5", fc='white', alpha=0.8),
                 fontsize=10)
    
    plt.legend(loc='upper left')
    
    # Loss plot
    plt.subplot(1, 2, 2)
    if losses:
        plt.plot(losses, alpha=0.2, color='#d62728', label='Raw')
        plt.plot(smoothed_losses, linewidth=2, color='#9467bd', label='Smoothed')
        
        plt.title(f'{model_type.upper()} Training Loss', fontsize=14, fontweight='bold')
        plt.xlabel('Updates', fontsize=12)
        plt.ylabel('Loss Value', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Only show legend if we have data
        plt.legend(loc='upper right')
        
        # Use log scale if the loss range is large
        if max(losses) / (min(losses) + 1e-10) > 100:
            plt.yscale('log')
    else:
        plt.title('No Loss Data Yet', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{PATHS['figures_path']}/{model_type}_training_fig_{step}.png", dpi=150)
    plt.close()

class DQN:
    def __init__(self, model_path, env, model_type, logger=None):
        self.env = env
        self.model_path = model_path
        self.model_type = model_type
        self.logger = logger or setup_logger(model_type)
        
        # Load hyperparameters from config
        self.lr = TRAINING['lr']
        self.gamma = TRAINING['gamma']
        self.eps_decay = TRAINING['eps_decay']
        self.eps_start = TRAINING['eps_start']
        self.eps_end = TRAINING['eps_end']
        self.initial_memory = TRAINING['initial_memory']
        self.batch_size = TRAINING['batch_size']
        
        # Main replay buffer and collision experience buffer
        memory_size = TRAINING['memory_size']
        self.replay_buffer = deque(maxlen=memory_size)
        self.replay_buffer_b = deque(maxlen=memory_size)  # Buffer for collision experiences
        
        # Model dimensions from config
        self.num_actions = MODEL['num_actions']
        self.robot_dim = MODEL['robot_dim']
        self.obstacle_dim = MODEL['obstacle_dim']
        self.hidden_dim = MODEL['hidden_dim']
        
        # Create model based on specified type
        self.model = self.make_model()
        self.first = False  # Flag to indicate when training begins
        
        # Create output directories if they don't exist
        os.makedirs(model_path, exist_ok=True)
        os.makedirs(PATHS['figures_path'], exist_ok=True)
        
        self.logger.info(f"Initialized {model_type.upper()} model with {self.get_param_count():,} parameters")

    def get_param_count(self):
        """Count the number of parameters in the model."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
    def make_model(self):
        """Create the appropriate model based on model_type."""
        if self.model_type == "lstm":
            model = LSTMNetwork(
                self.robot_dim, 
                self.obstacle_dim, 
                MODEL['lstm']['hidden_dim'], 
                self.num_actions
            )
        elif self.model_type == "transformer":
            model = TransformerNetwork(
                self.robot_dim, 
                self.obstacle_dim, 
                MODEL['transformer']['hidden_dim'], 
                self.num_actions,
                nhead=MODEL['transformer']['nhead'],
                num_layers=MODEL['transformer']['num_layers'],
                dropout=MODEL['transformer']['dropout']
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
        return model

    def agent_policy(self, state, epsilon):
        """Epsilon-greedy policy for action selection."""
        if np.random.rand() < epsilon:
            # Random action
            action = random.randrange(self.num_actions)
        else:
            # Greedy action
            q_value = self.model(state)
            action = np.argmax(q_value.detach().numpy())

        return action

    def add_to_replay_buffer(self, state, action, reward, next_state, terminal):
        """Add experience to replay buffer."""
        self.replay_buffer.append((state, action, reward, next_state, terminal))

    def sample_from_reply_buffer(self):
        """Sample experiences from both replay buffers."""
        # Sample from main buffer
        random_sample = random.sample(self.replay_buffer, self.batch_size)
        
        # Sample from collision buffer if it has enough experiences
        if len(self.replay_buffer_b) >= int(self.batch_size/2):
            random_sample_b = random.sample(self.replay_buffer_b, int(self.batch_size/2))
            # Replace some samples from main buffer with collision samples
            random_sample = random_sample[:int(self.batch_size/2)] + random_sample_b
            
        return random_sample

    def get_memory(self, random_sample):
        """Extract batches of states, actions, rewards, etc. from sampled experiences."""
        states = torch.cat([i[0] for i in random_sample], dim=0)
        actions = torch.tensor([i[1] for i in random_sample])
        rewards = torch.tensor([i[2] for i in random_sample])
        next_states = torch.cat([i[3] for i in random_sample], dim=0)
        terminals = torch.tensor([i[4] for i in random_sample]) * 1

        return states, actions, rewards, next_states, terminals

    def train_with_relay_buffer(self):
        """Train the model using experiences from the replay buffer."""
        # Ensure buffer has enough samples
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample experiences and extract batches
        sample = self.sample_from_reply_buffer()
        states, actions, rewards, next_states, terminals = self.get_memory(sample)
        
        # Compute target Q-values
        next_q_mat = self.model(next_states)
        next_q_vec = np.max(next_q_mat.detach().numpy(), axis=1).squeeze()
        target_vec = rewards + self.gamma * next_q_vec * (1 - terminals.detach().numpy())
        
        # Compute current Q-values
        q_mat = self.model(states)
        q_vec = q_mat.gather(dim=1, index=actions.unsqueeze(1)).type(torch.FloatTensor)
        target_vec = target_vec.unsqueeze(1).type(torch.FloatTensor)
        
        # Compute loss and optimize
        loss = self.loss_func(q_vec, target_vec)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss

    def train(self, num_episodes=None):
        """Main training loop with curriculum learning for moving obstacles."""
        if num_episodes is None:
            num_episodes = TRAINING['num_episodes']
            
        self.model.train()
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Initialize training metrics
        steps_done = 0
        losses = []
        rewards_list = []
        collision_count = 0
        success_count = 0
        timeout_count = 0
        episode_steps_list = []
        
        # Set up curriculum learning if enabled
        use_curriculum = ENVIRONMENT.get('curriculum_learning', False)
        curriculum_phases = ENVIRONMENT.get('curriculum_phases', [])
        current_phase = 0
        phase_episode_count = 0
        moving_ratio = ENVIRONMENT.get('moving_obstacle_ratio', 0.8)
        velocity_scale = ENVIRONMENT.get('obstacle_velocity_scale', 0.3)
        
        # Start time for tracking training duration
        start_time = time.time()
        self.logger.info(f"Starting training for {num_episodes} episodes")
        
        if use_curriculum and curriculum_phases:
            self.logger.info("Using curriculum learning for moving obstacles:")
            for i, phase in enumerate(curriculum_phases):
                self.logger.info(f"  Phase {i+1}: Moving ratio {phase['moving_ratio']}, " +
                            f"Velocity scale {phase['velocity_scale']}, " +
                            f"Episodes {phase['episodes']}")
            
            # Initialize with first phase settings
            moving_ratio = curriculum_phases[0]['moving_ratio']
            velocity_scale = curriculum_phases[0]['velocity_scale']
            self.logger.info(f"Starting with Phase 1: Moving ratio {moving_ratio}, Velocity scale {velocity_scale}")
        
        # Print initial progress bar
        print_progress_bar(0, num_episodes, prefix=f'{self.model_type.upper()} Progress:', 
                        suffix='Starting...', length=TRAINING['progress_bar_length'])
        
        for episode in range(num_episodes):
            episode_start_time = time.time()
            
            # Check if we need to advance to the next curriculum phase
            if use_curriculum and curriculum_phases and current_phase < len(curriculum_phases):
                phase_episode_count += 1
                phase_episodes = curriculum_phases[current_phase]['episodes']
                
                if phase_episode_count >= phase_episodes and current_phase < len(curriculum_phases) - 1:
                    current_phase += 1
                    phase_episode_count = 0
                    moving_ratio = curriculum_phases[current_phase]['moving_ratio']
                    velocity_scale = curriculum_phases[current_phase]['velocity_scale']
                    self.logger.info(f"Advancing to Phase {current_phase+1}: " +
                                f"Moving ratio {moving_ratio}, Velocity scale {velocity_scale}")
            
            # Reset environment with obstacles and current curriculum parameters
            obs = self.env.reset(
                obstacle_num=ENVIRONMENT['obstacle_num'], 
                layout=ENVIRONMENT['layout'], 
                test_phase=False, 
                counter=None,
                moving_obstacle_ratio=moving_ratio,
                obstacle_velocity_scale=velocity_scale
            )
            state = self.env.convert_coord(obs)
            reward_for_episode = 0
            num_step_per_eps = 0
            ep_buffer = []  # Buffer to store episode experiences
            
            while True:
                # Compute epsilon for exploration
                epsilon = self.eps_end + (self.eps_start - self.eps_end) * np.exp(-steps_done / self.eps_decay)
                
                # Select action
                received_action = self.agent_policy(state, epsilon)
                vel_action = self.env.vel_samples[received_action]
                
                # Take step in environment
                steps_done += 1
                num_step_per_eps += 1
                next_obs, reward, terminal, info = self.env.step(vel_action)
                next_state = self.env.convert_coord(next_obs)
                
                # Store experience
                ep_buffer.append((state, received_action, reward, next_state, terminal))
                
                # Update total episode reward
                reward_for_episode += reward
                state = next_state

                # Train if memory is sufficient and it's time to update
                if max(len(self.replay_buffer), len(self.replay_buffer_b)) >= self.initial_memory and steps_done % TRAINING['update_frequency'] == 0:
                    loss = self.train_with_relay_buffer()
                    if loss is not None:
                        losses.append(loss.item())

                # Log when training begins
                if max(len(self.replay_buffer), len(self.replay_buffer_b)) >= self.initial_memory and not self.first:
                    self.logger.info(f"Starting learning from buffer after {steps_done} steps")
                    self.logger.info(f"Main buffer size: {len(self.replay_buffer)}")
                    self.logger.info(f"Collision buffer size: {len(self.replay_buffer_b)}")
                    if use_curriculum:
                        self.logger.info(f"Current curriculum phase: {current_phase+1}/{len(curriculum_phases)}")
                        self.logger.info(f"  Moving ratio: {moving_ratio}, Velocity scale: {velocity_scale}")
                    self.first = True
                
                # Episode termination
                if terminal:
                    # Update counters based on outcome
                    if info == "collision":
                        collision_count += 1
                        self.replay_buffer_b += ep_buffer
                    elif "Goal reached" in info:
                        success_count += 1
                        self.replay_buffer += ep_buffer
                    elif info == "timeout":
                        timeout_count += 1
                        self.replay_buffer += ep_buffer
                    else:
                        self.replay_buffer += ep_buffer
                    
                    # Log episode results
                    rewards_list.append(reward_for_episode)
                    episode_steps_list.append(num_step_per_eps)
                    episode_time = time.time() - episode_start_time
                    
                    # Update progress bar
                    avg_reward = np.mean(rewards_list[-10:]) if len(rewards_list) >= 10 else np.mean(rewards_list)
                    progress_suffix = f'Ep: {episode+1}/{num_episodes} | Reward: {reward_for_episode:.2f} | Avg: {avg_reward:.2f}'
                    print_progress_bar(episode+1, num_episodes, prefix=f'{self.model_type.upper()} Progress:',
                                    suffix=progress_suffix, length=TRAINING['progress_bar_length'])
                    
                    # Decide log level based on frequency
                    if episode % TRAINING['log_frequency'] == 0:
                        curriculum_info = ""
                        if use_curriculum:
                            curriculum_info = f" | Phase: {current_phase+1}/{len(curriculum_phases)} " + \
                                            f"({moving_ratio:.1f} moving, {velocity_scale:.1f} vel)"
                        
                        self.logger.info(
                            f"Episode: {episode}/{num_episodes} | "
                            f"Reward: {reward_for_episode:.2f} | "
                            f"Status: {info} | "
                            f"Steps: {num_step_per_eps}"
                            f"{curriculum_info} | "
                            f"Time: {episode_time:.2f}s"
                        )
                    
                    # Detailed terminal output at specified intervals
                    if episode % TRAINING['terminal_output_frequency'] == 0:
                        # Calculate statistics
                        success_rate = success_count / (episode + 1)
                        collision_rate = collision_count / (episode + 1)
                        avg_steps = np.mean(episode_steps_list[-10:]) if len(episode_steps_list) >= 10 else np.mean(episode_steps_list)
                        
                        # Clear the current line (progress bar will be redrawn)
                        print("\r", end="")
                        
                        # Print stats to terminal
                        print(f"\n--- {self.model_type.upper()} Episode {episode+1}/{num_episodes} Stats ---")
                        print(f"  Last reward: {reward_for_episode:.2f}, Avg reward: {avg_reward:.2f}")
                        print(f"  Success: {success_rate:.1%}, Collision: {collision_rate:.1%}")
                        print(f"  Epsilon: {epsilon:.4f}, Steps: {num_step_per_eps} (avg: {avg_steps:.1f})")
                        if use_curriculum:
                            print(f"  Curriculum: Phase {current_phase+1}/{len(curriculum_phases)}, " +
                                f"{moving_ratio:.1f} moving ratio, {velocity_scale:.1f} velocity scale")
                        print(f"  Memory: {len(self.replay_buffer)} main, {len(self.replay_buffer_b)} collision")
                        print()
                        
                        # Redraw progress bar
                        print_progress_bar(episode+1, num_episodes, prefix=f'{self.model_type.upper()} Progress:',
                                        suffix=progress_suffix, length=TRAINING['progress_bar_length'])
                    
                    # Detailed logging at specified intervals
                    if episode > 0 and episode % TRAINING['detailed_log_frequency'] == 0:
                        elapsed_time = time.time() - start_time
                        avg_recent_reward = np.mean(rewards_list[-10:]) if len(rewards_list) >= 10 else np.mean(rewards_list)
                        
                        # Calculate success and collision rates
                        success_rate = success_count / (episode + 1)
                        collision_rate = collision_count / (episode + 1)
                        timeout_rate = timeout_count / (episode + 1)
                        
                        curriculum_info = ""
                        if use_curriculum:
                            curriculum_info = f" | Phase: {current_phase+1}/{len(curriculum_phases)} " + \
                                            f"({moving_ratio:.1f} moving, {velocity_scale:.1f} vel)"
                        
                        self.logger.info(
                            f"Training Stats | "
                            f"Episodes: {episode+1}/{num_episodes} | "
                            f"Steps: {steps_done} | "
                            f"Elapsed: {elapsed_time:.2f}s | "
                            f"Avg Reward: {avg_recent_reward:.2f} | "
                            f"Success: {success_rate:.2%} | "
                            f"Collision: {collision_rate:.2%} | "
                            f"Epsilon: {epsilon:.4f}"
                            f"{curriculum_info}"
                        )
                    
                    # Update visualization at specified intervals
                    if episode % TRAINING['plot_frequency'] == 0:
                        plot_stats(steps_done, rewards_list, losses, episode, self.model_type)
                    
                    # Save model at specified intervals
                    if (episode + 1) % TRAINING['save_frequency'] == 0 or episode == num_episodes - 1:
                        path = os.path.join(self.model_path, f"{self.model_type}_ep_{episode}.pth")
                        torch.save(self.model.state_dict(), path)
                        self.logger.info(f"Model saved to {path}")
                        # Print to terminal for visibility
                        print(f"\n[CHECKPOINT] Model saved at episode {episode+1}: {path}\n")
                        # Redraw progress bar
                        print_progress_bar(episode+1, num_episodes, prefix=f'{self.model_type.upper()} Progress:',
                                        suffix=progress_suffix, length=TRAINING['progress_bar_length'])
                    
                    break
        
        # Training complete
        total_time = time.time() - start_time
        success_rate = success_count / num_episodes
        collision_rate = collision_count / num_episodes
        timeout_rate = timeout_count / num_episodes
        
        # Clear the progress bar line
        print("\n")
        
        # Print training summary to terminal
        print("\n" + "="*80)
        print(f" {self.model_type.upper()} TRAINING COMPLETE")
        print("="*80)
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Success rate: {success_rate:.2%}")
        print(f"Collision rate: {collision_rate:.2%}")
        print(f"Timeout rate: {timeout_rate:.2%}")
        print(f"Average reward: {np.mean(rewards_list):.2f}")
        print(f"Total steps: {steps_done}")
        if use_curriculum:
            print(f"Final curriculum phase: {current_phase+1}/{len(curriculum_phases)}")
            print(f"  Moving obstacle ratio: {moving_ratio:.2f}")
            print(f"  Obstacle velocity scale: {velocity_scale:.2f}")
        print("="*80 + "\n")
        
        self.logger.info(f"Training completed in {total_time:.2f} seconds")
        self.logger.info(f"Final Success Rate: {success_rate:.2%}")
        self.logger.info(f"Final Collision Rate: {collision_rate:.2%}")
        self.logger.info(f"Final Timeout Rate: {timeout_rate:.2%}")
        self.logger.info(f"Average Reward: {np.mean(rewards_list):.2f}")
        if use_curriculum:
            self.logger.info(f"Final curriculum settings: {moving_ratio:.2f} moving ratio, {velocity_scale:.2f} velocity scale")
        
        # Save final model
        final_path = os.path.join(self.model_path, f"{self.model_type}_final.pth")
        torch.save(self.model.state_dict(), final_path)
        self.logger.info(f"Final model saved to {final_path}")
        
        # Create final plot
        plot_stats(steps_done, rewards_list, losses, num_episodes, self.model_type)
        self.logger.info(f"Final plot saved to {PATHS['figures_path']}/{self.model_type}_training_fig_{num_episodes}.png")
        
        return {
            'success_rate': success_rate,
            'collision_rate': collision_rate,
            'timeout_rate': timeout_rate,
            'avg_reward': np.mean(rewards_list),
            'final_reward': rewards_list[-1] if rewards_list else 0,
            'episodes': num_episodes,
            'steps': steps_done,
            'training_time': total_time,
        }

def main():
    parser = argparse.ArgumentParser(description='Train obstacle avoidance with DQN')
    parser.add_argument('--model', type=str, choices=['lstm', 'transformer'], required=True,
                      help='Model architecture to use (lstm or transformer)')
    parser.add_argument('--episodes', type=int, default=TRAINING['num_episodes'],
                      help='Number of episodes to train')
    args = parser.parse_args()
    
    # Set up logger
    logger = setup_logger(args.model)
    
    # Initialize environment
    env = ObstacleEnv()

    # Set model path based on model type
    model_path = f"{PATHS['base_model_path']}/{args.model}/"
    
    logger.info(f'Starting training with {args.model} model on obstacle environment for {args.episodes} episodes')
    
    # Measure training time
    start_time = time.time()
    
    # Create and train the model
    dqn = DQN(model_path, env, args.model, logger)
    results = dqn.train(args.episodes)
    
    # Log training summary
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info(f"Training completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    logger.info(f"Final success rate: {results['success_rate']:.2%}")
    logger.info(f"Final collision rate: {results['collision_rate']:.2%}")
    logger.info(f"Final timeout rate: {results['timeout_rate']:.2%}")
    logger.info(f"Average reward: {results['avg_reward']:.2f}")
    
    return results

if __name__ == "__main__":
    main()