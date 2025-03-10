# evaluate.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from env import ObstacleEnv
from models import create_model
from config import MODEL, ENVIRONMENT, PATHS

class Evaluator:
    def __init__(self, model_path, model_type, episode, dimension='2D'):
        self.model_path = model_path
        self.model_type = model_type
        self.episode = episode
        self.dimension = dimension
        self.env = ObstacleEnv(dimension=dimension)
        
        # Load the model
        self.model = self.load_model()
        
        # Output directory
        os.makedirs(PATHS['eval_path'], exist_ok=True)
    
    def load_model(self):
        """Load the trained model with improved file checking."""
        # Create model using factory function
        model = create_model(self.model_type, self.dimension)
        
        # Check multiple possible model file patterns
        possible_model_files = [
            # Try with episode and dimension
            os.path.join(self.model_path, f"{self.model_type}_{self.dimension}_ep_{self.episode}.pth"),
            # Try final model with dimension
            os.path.join(self.model_path, f"{self.model_type}_{self.dimension}_final.pth"),
            # Try with episode but without dimension
            os.path.join(self.model_path, f"{self.model_type}_ep_{self.episode}.pth"),
            # Try final model without dimension
            os.path.join(self.model_path, f"{self.model_type}_final.pth"),
            # Try interrupted model
            os.path.join(self.model_path, f"{self.model_type}_{self.dimension}_interrupted.pth")
        ]
        
        # Try to load each possible file
        model_loaded = False
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
            print(f"No model file found for {self.model_type} {self.dimension}. Using untrained model.")
        
        model.eval()  # Set to evaluation mode
        return model
    
    def run_evaluation(self, num_episodes=100, render_every=25, 
                 obstacle_num=ENVIRONMENT['obstacle_num'], 
                 layout=ENVIRONMENT['layout'],
                 moving_obstacle_ratio=ENVIRONMENT.get('moving_obstacle_ratio', 0.8),
                 obstacle_velocity_scale=ENVIRONMENT.get('obstacle_velocity_scale', 0.3)):
        """
        Run evaluation episodes.
        
        Args:
            num_episodes: Number of evaluation episodes
            render_every: Render environment every N episodes (0 to disable)
            obstacle_num: Number of obstacles
            layout: Obstacle layout
            moving_obstacle_ratio: Ratio of obstacles that move
            obstacle_velocity_scale: Velocity scale factor for obstacles
            
        Returns:
            Dictionary of evaluation metrics
        """
        success_rate = 0
        collision_rate = 0
        timeout_rate = 0
        rewards = []
        path_lengths = []
        
        # For tracking performance with different obstacle mobility settings
        static_results = {'success': 0, 'collision': 0, 'timeout': 0, 'episodes': 0}
        moving_results = {'success': 0, 'collision': 0, 'timeout': 0, 'episodes': 0}
        
        print(f"Running evaluation in {self.dimension} environment with {self.model_type} model...")
        
        try:
            for episode in range(num_episodes):
                # Reset environment
                obs = self.env.reset(
                    obstacle_num=obstacle_num, 
                    layout=layout, 
                    test_phase=True, 
                    counter=episode,
                    moving_obstacle_ratio=moving_obstacle_ratio,
                    obstacle_velocity_scale=obstacle_velocity_scale,
                    dimension=self.dimension
                )
                state = self.env.convert_coord(obs)
                
                # Count the number of moving obstacles in this episode
                num_moving_obstacles = sum(1 for obs in self.env.obstacle_list if obs.v_pref > 0)
                
                # Track whether this episode has moving obstacles
                has_moving_obstacles = num_moving_obstacles > 0
                if has_moving_obstacles:
                    moving_results['episodes'] += 1
                else:
                    static_results['episodes'] += 1
                    
                done = False
                episode_reward = 0
                steps = 0
                
                # Flag to render this episode
                should_render = render_every > 0 and episode % render_every == 0
                
                while not done:
                    # Get action from model
                    with torch.no_grad():
                        q_values = self.model(state)
                        action_idx = torch.argmax(q_values).item()
                    
                    # Execute action
                    vel_action = self.env.vel_samples[action_idx]
                    next_obs, reward, done, info = self.env.step(vel_action)
                    next_state = self.env.convert_coord(next_obs)
                    
                    # Update variables
                    state = next_state
                    episode_reward += reward
                    steps += 1
                    
                    # Render if needed
                    if should_render and episode == render_every - 1:
                        self.env.render()
                
                # Update metrics based on episode outcome
                if info == "collision":
                    collision_rate += 1
                    if has_moving_obstacles:
                        moving_results['collision'] += 1
                    else:
                        static_results['collision'] += 1
                elif info == "timeout":
                    timeout_rate += 1
                    if has_moving_obstacles:
                        moving_results['timeout'] += 1
                    else:
                        static_results['timeout'] += 1
                elif "Goal reached" in info or "goal reached" in info.lower():  # Check for success in a more flexible way
                    success_rate += 1
                    path_lengths.append(steps)
                    if has_moving_obstacles:
                        moving_results['success'] += 1
                    else:
                        static_results['success'] += 1
                else:
                    # If we get here, it's an unrecognized outcome - log it
                    print(f"Unrecognized episode outcome: '{info}'")
                    # Default to timeout for unrecognized outcomes
                    timeout_rate += 1
                    if has_moving_obstacles:
                        moving_results['timeout'] += 1
                    else:
                        static_results['timeout'] += 1
                
                rewards.append(episode_reward)
                
                if episode % 10 == 0:
                    print(f"Completed episode {episode}/{num_episodes}, reward: {episode_reward:.2f}, " +
                        f"outcome: {info}, moving obstacles: {num_moving_obstacles}/{obstacle_num}")
        
        except KeyboardInterrupt:
            print("\nEvaluation interrupted by user.")
            if not rewards:  # If we haven't completed any episodes
                raise KeyboardInterrupt("No evaluation data collected")
            
            # Adjust the actual number of episodes completed
            num_episodes = len(rewards)
            print(f"Using data from {num_episodes} completed episodes for evaluation.")
        
        # Normalize rates based on actual episodes completed
        success_rate /= num_episodes
        collision_rate /= num_episodes
        timeout_rate /= num_episodes
        
        # Calculate rates for static and moving obstacle scenarios
        static_success_rate = static_results['success'] / max(static_results['episodes'], 1)
        static_collision_rate = static_results['collision'] / max(static_results['episodes'], 1)
        
        moving_success_rate = moving_results['success'] / max(moving_results['episodes'], 1)
        moving_collision_rate = moving_results['collision'] / max(moving_results['episodes'], 1)
        
        # Compute average path length for successful episodes
        avg_path_length = np.mean(path_lengths) if path_lengths else 0
        
        # Create results dictionary
        results = {
            "model_type": self.model_type,
            "dimension": self.dimension,
            "success_rate": success_rate,
            "collision_rate": collision_rate,
            "timeout_rate": timeout_rate,
            "avg_reward": np.mean(rewards),
            "avg_path_length": avg_path_length,
            "static_success_rate": static_success_rate,
            "static_collision_rate": static_collision_rate,
            "moving_success_rate": moving_success_rate,
            "moving_collision_rate": moving_collision_rate,
            "static_episodes": static_results['episodes'],
            "moving_episodes": moving_results['episodes'],
            "episodes_completed": num_episodes  # Add the actual number of episodes completed
        }
        
        return results
    
    def plot_results(self, results):
        """Plot evaluation results with improved styling."""
        plt.style.use('ggplot')
        plt.figure(figsize=(14, 10))
        
        # Success, collision, timeout rates
        plt.subplot(221)
        rates = [results["success_rate"], results["collision_rate"], results["timeout_rate"]]
        labels = ["Success", "Collision", "Timeout"]
        colors = ['#2ca02c', '#d62728', '#7f7f7f']  # Green, Red, Gray
        
        bars = plt.bar(labels, rates, color=colors, alpha=0.8)
        plt.ylim(0, 1)
        plt.title(f"{self.model_type.upper()} {self.dimension} Overall Performance", fontsize=14, fontweight='bold')
        plt.ylabel("Rate", fontsize=12)
        
        # Add percentage labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # Static vs Moving obstacles comparison
        plt.subplot(222)
        
        # Calculate static and moving rates
        static_rates = [results["static_success_rate"], results["static_collision_rate"]]
        moving_rates = [results["moving_success_rate"], results["moving_collision_rate"]]
        
        x = np.arange(2)
        width = 0.35
        
        ax = plt.gca()
        rects1 = ax.bar(x - width/2, static_rates, width, label='Static Obstacles', color='#1f77b4')
        rects2 = ax.bar(x + width/2, moving_rates, width, label='Moving Obstacles', color='#ff7f0e')
        
        ax.set_ylim(0, 1)
        ax.set_ylabel('Rate')
        ax.set_title('Static vs. Moving Obstacle Performance', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['Success', 'Collision'])
        ax.legend()
        
        # Add percentage labels
        def add_percentage_labels(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.1%}',
                          xy=(rect.get_x() + rect.get_width() / 2, height),
                          xytext=(0, 3),  # 3 points vertical offset
                          textcoords="offset points",
                          ha='center', va='bottom', fontweight='bold')
        
        add_percentage_labels(rects1)
        add_percentage_labels(rects2)
        
        # Add text with numerical results
        plt.subplot(223)
        plt.axis('off')
        result_text = "\n".join([
            f"Model: {self.model_type.upper()} ({self.dimension})",
            f"Success Rate: {results['success_rate']:.2%}",
            f"Collision Rate: {results['collision_rate']:.2%}",
            f"Timeout Rate: {results['timeout_rate']:.2%}",
            f"Average Reward: {results['avg_reward']:.2f}",
            f"Average Path Length: {results['avg_path_length']:.2f} steps",
            f"\nStatic Obstacles ({results['static_episodes']} episodes):",
            f"  Success Rate: {results['static_success_rate']:.2%}",
            f"  Collision Rate: {results['static_collision_rate']:.2%}",
            f"\nMoving Obstacles ({results['moving_episodes']} episodes):",
            f"  Success Rate: {results['moving_success_rate']:.2%}",
            f"  Collision Rate: {results['moving_collision_rate']:.2%}"
        ])
        plt.text(0.1, 0.5, result_text, fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.5", fc='white', alpha=0.8))
        
        # Add a radar chart for visual comparison of metrics
        plt.subplot(224, polar=True)
        
        # Create normalized metrics for radar chart
        metrics = ['Overall\nSuccess', 'Static\nSuccess', 'Moving\nSuccess', 'Safety\n(1-Collision)', 'Efficiency\n(Reward)']
        
        # Normalize metrics to 0-1 scale (where 1 is always better)
        values = [
            results['success_rate'],
            results['static_success_rate'],
            results['moving_success_rate'],
            1 - results['collision_rate'],  # Invert collision rate so higher is better
            # Normalize reward to 0-1 scale (approximation)
            min(max(results['avg_reward'] / 10, 0), 1) if results['avg_reward'] > 0 else 0
        ]
        
        # Add first value again to close the polygon
        metrics = np.concatenate((metrics, [metrics[0]]))
        values = np.concatenate((values, [values[0]]))
        
        # Compute angle for each metric
        angles = np.linspace(0, 2*np.pi, len(metrics)-1, endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        
        # Draw the radar chart
        ax = plt.subplot(224, polar=True)
        ax.fill(angles, values, color='#1f77b4', alpha=0.25)
        ax.plot(angles, values, 'o-', linewidth=2, color='#1f77b4')
        
        # Set radar chart properties
        ax.set_thetagrids(angles[:-1] * 180/np.pi, metrics[:-1])
        ax.set_ylim(0, 1)
        ax.grid(True)
        ax.set_title(f"{self.model_type.upper()} {self.dimension} Performance Profile", fontsize=14, fontweight='bold')
        
        # Add rings for reference
        ax.set_rticks([0.25, 0.5, 0.75, 1])
        ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'])
        
        # Save figure with dimension in filename
        plt.tight_layout()
        plt.savefig(f"{PATHS['eval_path']}/{self.model_type}_{self.dimension}_eval_results.png", dpi=150)
        plt.close()

def compare_models(results_dict):
    """
    Compare and visualize results from different models and dimensions.
    
    Args:
        results_dict: Dictionary mapping model-dimension pairs to results
    """
    # Setup for comparison plots
    plt.figure(figsize=(15, 10))
    
    # Collect all model-dimension combinations
    model_dims = list(results_dict.keys())
    num_models = len(model_dims)
    
    # Extract metrics for comparison
    metrics = ["success_rate", "collision_rate", "static_success_rate", "moving_success_rate", "avg_reward"]
    labels = ["Overall Success", "Collision Rate", "Static Success", "Moving Success", "Avg Reward"]
    
    # Normalize reward for better visualization
    max_reward = max([results_dict[md]["avg_reward"] for md in model_dims])
    
    # Prepare values for plotting
    values = []
    for model_dim in model_dims:
        results = results_dict[model_dim]
        model_values = [
            results["success_rate"],
            results["collision_rate"],
            results["static_success_rate"],
            results["moving_success_rate"],
            results["avg_reward"] / max_reward if max_reward != 0 else 0
        ]
        values.append(model_values)
    
    # Bar chart comparison
    x = np.arange(len(labels))
    width = 0.8 / num_models  # Adjust bar width based on number of models
    
    # Plot bars for each model-dimension combination
    for i, model_dim in enumerate(model_dims):
        plt.bar(x + (i - num_models/2 + 0.5) * width, values[i], width, label=model_dim)
    
    plt.xlabel('Metrics')
    plt.title('Model Performance Comparison (2D vs 3D)', fontsize=14, fontweight='bold')
    plt.xticks(x, labels)
    plt.legend()
    
    # Add actual values as text
    for i, model_dim in enumerate(model_dims):
        for j, v in enumerate(values[i]):
            if metrics[j] == "avg_reward":
                plt.text(j + (i - num_models/2 + 0.5) * width, v + 0.02, 
                        f"{results_dict[model_dim]['avg_reward']:.2f}", 
                        ha='center', fontsize=8)
            else:
                plt.text(j + (i - num_models/2 + 0.5) * width, v + 0.02, 
                        f"{v:.2f}", ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f"{PATHS['eval_path']}/model_dimension_comparison.png")
    
    # Print comparison results
    print("\nModel-Dimension Comparison:")
    print("-" * 70)
    
    # Table headers
    header = "| {:^20} | {:^10} | {:^10} | {:^10} | {:^10} |".format(
        "Model-Dimension", "Success", "Collision", "Static Succ", "Moving Succ")
    print(header)
    print("-" * 70)
    
    # Table rows
    for model_dim in model_dims:
        results = results_dict[model_dim]
        row = "| {:^20} | {:^10.2%} | {:^10.2%} | {:^10.2%} | {:^10.2%} |".format(
            model_dim,
            results["success_rate"],
            results["collision_rate"],
            results["static_success_rate"],
            results["moving_success_rate"]
        )
        print(row)
    
    print("-" * 70)
    print("\nDetailed Analysis:")
    
    # Compare 2D vs 3D for each model type
    model_types = set([md.split('-')[0] for md in model_dims])
    for model in model_types:
        if f"{model}-2D" in results_dict and f"{model}-3D" in results_dict:
            r2d = results_dict[f"{model}-2D"]
            r3d = results_dict[f"{model}-3D"]
            
            print(f"\n{model.upper()} 2D vs 3D:")
            print(f"  Success: 2D={r2d['success_rate']:.2%}, 3D={r3d['success_rate']:.2%}, " +
                 f"Diff={r3d['success_rate']-r2d['success_rate']:.2%}")
            print(f"  Collision: 2D={r2d['collision_rate']:.2%}, 3D={r3d['collision_rate']:.2%}, " +
                 f"Diff={r3d['collision_rate']-r2d['collision_rate']:.2%}")
            print(f"  Moving Success: 2D={r2d['moving_success_rate']:.2%}, 3D={r3d['moving_success_rate']:.2%}, " +
                 f"Diff={r3d['moving_success_rate']-r2d['moving_success_rate']:.2%}")
    
    # Compare LSTM vs Transformer for each dimension
    for dim in ["2D", "3D"]:
        if f"lstm-{dim}" in results_dict and f"transformer-{dim}" in results_dict:
            rl = results_dict[f"lstm-{dim}"]
            rt = results_dict[f"transformer-{dim}"]
            
            print(f"\nLSTM vs Transformer in {dim}:")
            print(f"  Success: LSTM={rl['success_rate']:.2%}, Transformer={rt['success_rate']:.2%}, " +
                 f"Diff={rt['success_rate']-rl['success_rate']:.2%}")
            print(f"  Collision: LSTM={rl['collision_rate']:.2%}, Transformer={rt['collision_rate']:.2%}, " +
                 f"Diff={rt['collision_rate']-rl['collision_rate']:.2%}")
            print(f"  Moving Success: LSTM={rl['moving_success_rate']:.2%}, Transformer={rt['moving_success_rate']:.2%}, " +
                 f"Diff={rt['moving_success_rate']-rl['moving_success_rate']:.2%}")
    
    print("-" * 70)

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained models')
    parser.add_argument('--lstm-episode', type=int, default=499, help='LSTM model episode to evaluate')
    parser.add_argument('--transformer-episode', type=int, default=499, help='Transformer model episode to evaluate')
    parser.add_argument('--eval-episodes', type=int, default=200, help='Number of evaluation episodes')
    parser.add_argument('--render-every', type=int, default=0, help='Render every N episodes (0 to disable)')
    parser.add_argument('--dimension', type=str, choices=['2D', '3D', 'both'], default='both', 
                      help='Dimension to evaluate (2D, 3D, or both)')
    args = parser.parse_args()
    
    # Paths
    lstm_path = f"{PATHS['base_model_path']}/lstm/"
    transformer_path = f"{PATHS['base_model_path']}/transformer/"
    
    # Dimensions to evaluate
    dimensions = ['2D', '3D'] if args.dimension == 'both' else [args.dimension]
    
    # Store results for comparison
    all_results = {}
    
    # Run evaluations for each model and dimension
    for dimension in dimensions:
        try:
            # Evaluate LSTM model
            print(f"\nEvaluating LSTM model in {dimension} environment...")
            lstm_evaluator = Evaluator(lstm_path, "lstm", args.lstm_episode, dimension)
            lstm_results = lstm_evaluator.run_evaluation(args.eval_episodes, args.render_every)
            lstm_evaluator.plot_results(lstm_results)
            all_results[f"lstm-{dimension}"] = lstm_results
            
            # Evaluate Transformer model
            print(f"\nEvaluating Transformer model in {dimension} environment...")
            transformer_evaluator = Evaluator(transformer_path, "transformer", args.transformer_episode, dimension)
            transformer_results = transformer_evaluator.run_evaluation(args.eval_episodes, args.render_every)
            transformer_evaluator.plot_results(transformer_results)
            all_results[f"transformer-{dimension}"] = transformer_results
        except KeyboardInterrupt:
            print(f"\nEvaluation interrupted for {dimension} environment.")
            # Continue with the next dimension if available
            continue
        except Exception as e:
            print(f"\nError evaluating models in {dimension} environment: {str(e)}")
            continue
    
    # Compare all models and dimensions
    if len(all_results) > 1:
        try:
            compare_models(all_results)
        except Exception as e:
            print(f"Error comparing models: {str(e)}")

if __name__ == "__main__":
    main()