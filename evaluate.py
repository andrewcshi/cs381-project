import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from env import ObstacleEnv
from models import LSTMNetwork, TransformerNetwork
from config import MODEL, ENVIRONMENT, PATHS

class Evaluator:
    def __init__(self, model_path, model_type, episode):
        self.model_path = model_path
        self.model_type = model_type
        self.episode = episode
        self.env = ObstacleEnv()
        
        # Load the model
        self.model = self.load_model()
        
        # Output directory
        os.makedirs(PATHS['eval_path'], exist_ok=True)
    
    def load_model(self):
        """Load the trained model."""
        # Get model parameters from config
        robot_dim = MODEL['robot_dim']
        obstacle_dim = MODEL['obstacle_dim']
        num_actions = MODEL['num_actions']
        
        # Create model architecture
        if self.model_type == "lstm":
            model = LSTMNetwork(
                robot_dim, 
                obstacle_dim, 
                MODEL['lstm']['hidden_dim'], 
                num_actions
            )
        elif self.model_type == "transformer":
            model = TransformerNetwork(
                robot_dim, 
                obstacle_dim, 
                MODEL['transformer']['hidden_dim'], 
                num_actions, 
                nhead=MODEL['transformer']['nhead'], 
                num_layers=MODEL['transformer']['num_layers'], 
                dropout=MODEL['transformer']['dropout']
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Load weights
        model_file = os.path.join(self.model_path, f"{self.model_type}_ep_{self.episode}.pth")
        if os.path.exists(model_file):
            model.load_state_dict(torch.load(model_file))
            print(f"Loaded model from {model_file}")
        else:
            print(f"Model file {model_file} not found. Using untrained model.")
        
        model.eval()  # Set to evaluation mode
        return model
    
    def run_evaluation(self, num_episodes=100, render_every=25, 
                     obstacle_num=ENVIRONMENT['obstacle_num'], 
                     layout=ENVIRONMENT['layout']):
        """
        Run evaluation episodes.
        
        Args:
            num_episodes: Number of evaluation episodes
            render_every: Render environment every N episodes (0 to disable)
            obstacle_num: Number of obstacles
            layout: Obstacle layout
            
        Returns:
            Dictionary of evaluation metrics
        """
        success_rate = 0
        collision_rate = 0
        timeout_rate = 0
        rewards = []
        path_lengths = []
        
        for episode in range(num_episodes):
            # Reset environment
            obs = self.env.reset(obstacle_num=obstacle_num, layout=layout, 
                               test_phase=True, counter=episode)
            state = self.env.convert_coord(obs)
            
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
            elif info == "timeout":
                timeout_rate += 1
            else:  # Goal reached
                success_rate += 1
                path_lengths.append(steps)
            
            rewards.append(episode_reward)
            
            if episode % 10 == 0:
                print(f"Completed episode {episode}/{num_episodes}, reward: {episode_reward:.2f}, outcome: {info}")
        
        # Normalize rates
        success_rate /= num_episodes
        collision_rate /= num_episodes
        timeout_rate /= num_episodes
        
        # Compute average path length for successful episodes
        avg_path_length = np.mean(path_lengths) if path_lengths else 0
        
        # Create results dictionary
        results = {
            "model_type": self.model_type,
            "success_rate": success_rate,
            "collision_rate": collision_rate,
            "timeout_rate": timeout_rate,
            "avg_reward": np.mean(rewards),
            "avg_path_length": avg_path_length
        }
        
        return results
    
    def plot_results(self, results):
        """Plot evaluation results with improved styling."""
        plt.style.use('ggplot')
        plt.figure(figsize=(12, 8))
        
        # Success, collision, timeout rates
        plt.subplot(221)
        rates = [results["success_rate"], results["collision_rate"], results["timeout_rate"]]
        labels = ["Success", "Collision", "Timeout"]
        colors = ['#2ca02c', '#d62728', '#7f7f7f']  # Green, Red, Gray
        
        bars = plt.bar(labels, rates, color=colors, alpha=0.8)
        plt.ylim(0, 1)
        plt.title(f"{self.model_type.upper()} Performance Rates", fontsize=14, fontweight='bold')
        plt.ylabel("Rate", fontsize=12)
        
        # Add percentage labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # Add text with numerical results
        plt.subplot(222)
        plt.axis('off')
        result_text = "\n".join([
            f"Model: {self.model_type.upper()}",
            f"Success Rate: {results['success_rate']:.2%}",
            f"Collision Rate: {results['collision_rate']:.2%}",
            f"Timeout Rate: {results['timeout_rate']:.2%}",
            f"Average Reward: {results['avg_reward']:.2f}",
            f"Average Path Length: {results['avg_path_length']:.2f} steps"
        ])
        plt.text(0.1, 0.5, result_text, fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.5", fc='white', alpha=0.8))
        
        # Add a radar chart for visual comparison of metrics
        plt.subplot(212)
        
        # Create normalized metrics for radar chart
        metrics = ['Success Rate', 'Safety\n(1-Collision)', 'Efficiency\n(Reward)', 'Path\nOptimality']
        
        # Normalize metrics to 0-1 scale (where 1 is always better)
        values = [
            results['success_rate'],
            1 - results['collision_rate'],  # Invert collision rate so higher is better
            # Normalize reward to 0-1 scale (approximation)
            min(max(results['avg_reward'] / 10, 0), 1) if results['avg_reward'] > 0 else 0,
            # Path optimality - shorter paths are better (if we have successful episodes)
            0.8 / max(results['avg_path_length'] / 20, 0.1) if results['avg_path_length'] > 0 else 0
        ]
        
        # Add first value again to close the polygon
        metrics = np.concatenate((metrics, [metrics[0]]))
        values = np.concatenate((values, [values[0]]))
        
        # Compute angle for each metric
        angles = np.linspace(0, 2*np.pi, len(metrics)-1, endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        
        # Draw the radar chart
        ax = plt.subplot(212, polar=True)
        ax.fill(angles, values, color='#1f77b4', alpha=0.25)
        ax.plot(angles, values, 'o-', linewidth=2, color='#1f77b4')
        
        # Set radar chart properties
        ax.set_thetagrids(angles[:-1] * 180/np.pi, metrics[:-1])
        ax.set_ylim(0, 1)
        ax.grid(True)
        ax.set_title(f"{self.model_type.upper()} Performance Profile", fontsize=14, fontweight='bold')
        
        # Add rings for reference
        ax.set_rticks([0.25, 0.5, 0.75, 1])
        ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'])
        
        # Save figure
        plt.tight_layout()
        plt.savefig(f"{PATHS['eval_path']}/{self.model_type}_eval_results.png", dpi=150)
        plt.close()
        plt.close()

def compare_models(lstm_results, transformer_results):
    """Compare and visualize results from both models."""
    # Create bar chart comparing key metrics
    metrics = ["success_rate", "collision_rate", "avg_reward", "avg_path_length"]
    labels = ["Success Rate", "Collision Rate", "Avg Reward", "Avg Path Length"]
    
    plt.figure(figsize=(14, 10))
    
    # Normalize path length and reward for better visualization
    max_reward = max(lstm_results["avg_reward"], transformer_results["avg_reward"])
    max_path = max(lstm_results["avg_path_length"], transformer_results["avg_path_length"])
    
    lstm_values = [
        lstm_results["success_rate"],
        lstm_results["collision_rate"],
        lstm_results["avg_reward"] / max_reward if max_reward != 0 else 0,
        lstm_results["avg_path_length"] / max_path if max_path != 0 else 0
    ]
    
    transformer_values = [
        transformer_results["success_rate"],
        transformer_results["collision_rate"],
        transformer_results["avg_reward"] / max_reward if max_reward != 0 else 0,
        transformer_results["avg_path_length"] / max_path if max_path != 0 else 0
    ]
    
    x = np.arange(len(labels))
    width = 0.35
    
    plt.bar(x - width/2, lstm_values, width, label='LSTM')
    plt.bar(x + width/2, transformer_values, width, label='Transformer')
    
    plt.xlabel('Metrics')
    plt.title('LSTM vs Transformer Performance Comparison')
    plt.xticks(x, labels)
    plt.legend()
    
    # Add actual values as text
    for i, v in enumerate(lstm_values):
        if metrics[i] == "avg_reward":
            plt.text(i - width/2, v + 0.02, f"{lstm_results['avg_reward']:.2f}", ha='center')
        elif metrics[i] == "avg_path_length":
            plt.text(i - width/2, v + 0.02, f"{lstm_results['avg_path_length']:.2f}", ha='center')
        else:
            plt.text(i - width/2, v + 0.02, f"{v:.2f}", ha='center')
    
    for i, v in enumerate(transformer_values):
        if metrics[i] == "avg_reward":
            plt.text(i + width/2, v + 0.02, f"{transformer_results['avg_reward']:.2f}", ha='center')
        elif metrics[i] == "avg_path_length":
            plt.text(i + width/2, v + 0.02, f"{transformer_results['avg_path_length']:.2f}", ha='center')
        else:
            plt.text(i + width/2, v + 0.02, f"{v:.2f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(f"{PATHS['eval_path']}/model_comparison.png")
    
    # Print comparison results
    print("\nModel Comparison:")
    print("-" * 50)
    print(f"LSTM Success Rate: {lstm_results['success_rate']:.2%}")
    print(f"Transformer Success Rate: {transformer_results['success_rate']:.2%}")
    print()
    print(f"LSTM Collision Rate: {lstm_results['collision_rate']:.2%}")
    print(f"Transformer Collision Rate: {transformer_results['collision_rate']:.2%}")
    print()
    print(f"LSTM Average Reward: {lstm_results['avg_reward']:.2f}")
    print(f"Transformer Average Reward: {transformer_results['avg_reward']:.2f}")
    print()
    print(f"LSTM Average Path Length: {lstm_results['avg_path_length']:.2f}")
    print(f"Transformer Average Path Length: {transformer_results['avg_path_length']:.2f}")
    print("-" * 50)
    
    # Determine winner
    lstm_score = lstm_results["success_rate"] - lstm_results["collision_rate"] + lstm_results["avg_reward"] / max_reward
    transformer_score = transformer_results["success_rate"] - transformer_results["collision_rate"] + transformer_results["avg_reward"] / max_reward
    
    if lstm_score > transformer_score:
        diff = (lstm_score - transformer_score) / transformer_score * 100 if transformer_score != 0 else float('inf')
        print(f"LSTM outperforms Transformer by approximately {diff:.2f}%")
    elif transformer_score > lstm_score:
        diff = (transformer_score - lstm_score) / lstm_score * 100 if lstm_score != 0 else float('inf')
        print(f"Transformer outperforms LSTM by approximately {diff:.2f}%")
    else:
        print("Both models perform similarly")

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained models')
    parser.add_argument('--lstm-episode', type=int, default=499, help='LSTM model episode to evaluate')
    parser.add_argument('--transformer-episode', type=int, default=499, help='Transformer model episode to evaluate')
    parser.add_argument('--eval-episodes', type=int, default=100, help='Number of evaluation episodes')
    parser.add_argument('--render-every', type=int, default=0, help='Render every N episodes (0 to disable)')
    args = parser.parse_args()
    
    # Paths
    lstm_path = f"{PATHS['base_model_path']}/lstm/"
    transformer_path = f"{PATHS['base_model_path']}/transformer/"
    
    # Evaluate LSTM model
    print("\nEvaluating LSTM model...")
    lstm_evaluator = Evaluator(lstm_path, "lstm", args.lstm_episode)
    lstm_results = lstm_evaluator.run_evaluation(args.eval_episodes, args.render_every)
    lstm_evaluator.plot_results(lstm_results)
    
    # Evaluate Transformer model
    print("\nEvaluating Transformer model...")
    transformer_evaluator = Evaluator(transformer_path, "transformer", args.transformer_episode)
    transformer_results = transformer_evaluator.run_evaluation(args.eval_episodes, args.render_every)
    transformer_evaluator.plot_results(transformer_results)
    
    # Compare models
    compare_models(lstm_results, transformer_results)

if __name__ == "__main__":
    main()