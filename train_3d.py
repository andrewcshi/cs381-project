# train_3d.py

import os
import argparse
import torch
import numpy as np
from env import ObstacleEnv
from models import create_model
from train import DQN, setup_logger
from config import TRAINING, ENVIRONMENT, PATHS, MODEL

def main():
    parser = argparse.ArgumentParser(description='Train obstacle avoidance in 3D environment')
    parser.add_argument('--model', type=str, choices=['lstm', 'transformer'], required=True,
                      help='Model architecture to use (lstm or transformer)')
    parser.add_argument('--dimension', type=str, choices=['2D', '3D'], required=True,
                      help='Environment dimension (2D or 3D)')
    parser.add_argument('--episodes', type=int, default=TRAINING['num_episodes'],
                      help='Number of episodes to train')
    parser.add_argument('--obstacle-num', type=int, default=ENVIRONMENT['obstacle_num'],
                      help='Number of obstacles')
    parser.add_argument('--moving-ratio', type=float, default=ENVIRONMENT['moving_obstacle_ratio'],
                      help='Ratio of moving obstacles (0.0-1.0)')
    parser.add_argument('--eval', action='store_true',
                      help='Evaluate after training')
    parser.add_argument('--render', action='store_true',
                      help='Render during evaluation')
    parser.add_argument('--eval-episodes', type=int, default=200,
                      help='Number of episodes for evaluation')
    args = parser.parse_args()
    
    # Set up logger with dimension info
    logger = setup_logger(f"{args.model}_{args.dimension}")
    
    # Initialize environment with specified dimension
    env = ObstacleEnv(dimension=args.dimension)

    # Set model path based on model type
    model_path = f"{PATHS['base_model_path']}/{args.model}/"
    os.makedirs(model_path, exist_ok=True)
    
    logger.info(f'Starting training with {args.model} model in {args.dimension} environment '
               f'for {args.episodes} episodes')
    logger.info(f'Obstacles: {args.obstacle_num}, Moving ratio: {args.moving_ratio}')
    
    # Set dimensions based on environment type
    if args.dimension == '3D':
        robot_dim = MODEL['robot_dim_3d']
        obstacle_dim = MODEL['obstacle_dim_3d']
        num_actions = MODEL['num_actions_3d']
    else:
        robot_dim = MODEL['robot_dim']
        obstacle_dim = MODEL['obstacle_dim']
        num_actions = MODEL['num_actions']
        
    # Create the model directly with the correct dimensions - don't use DQN's make_model
    if args.model == "lstm":
        if args.dimension == '3D':
            model = create_model('lstm', '3D')
            hidden_dim = MODEL['lstm']['hidden_dim']
        else:
            model = create_model('lstm', '2D')
            hidden_dim = MODEL['lstm']['hidden_dim']
    else:  # transformer
        if args.dimension == '3D':
            model = create_model('transformer', '3D')
            hidden_dim = MODEL['transformer']['hidden_dim']
        else:
            model = create_model('transformer', '2D')
            hidden_dim = MODEL['transformer']['hidden_dim']
    
    # Create the DQN agent with the pre-built model
    dqn = DQN(model_path, env, args.model, logger, 
            robot_dim=robot_dim, 
            obstacle_dim=obstacle_dim, 
            num_actions=num_actions,
            hidden_dim=hidden_dim,
            model=model)
    
    try:
        # Train the model
        results = dqn.train(args.episodes)
        
        # Save the model with the dimension in the filename
        final_path = os.path.join(model_path, f"{args.model}_{args.dimension}_final.pth")
        torch.save(dqn.model.state_dict(), final_path)
        logger.info(f"Final {args.dimension} model saved to {final_path}")
        
        # Print training summary
        print("\n" + "="*80)
        print(f" {args.model.upper()} {args.dimension} TRAINING SUMMARY")
        print("="*80)
        print(f"Success rate: {results['success_rate']:.2%}")
        print(f"Collision rate: {results['collision_rate']:.2%}")
        print(f"Timeout rate: {results['timeout_rate']:.2%}")
        print(f"Average reward: {results['avg_reward']:.2f}")
        print(f"Training steps: {results['steps']}")
        print(f"Training time: {results['training_time']:.2f} seconds")
        print("="*80)
        
        # Run evaluation if requested
        if args.eval:
            print("\nRunning evaluation...")
            from evaluate import Evaluator
            
            # Create evaluator with the same dimension
            evaluator = Evaluator(model_path, args.model, 'final', args.dimension)
            
            # Run evaluation using command-line argument
            eval_episodes = args.eval_episodes
            render_every = 10 if args.render else 0
            
            logger.info(f"Starting evaluation with {eval_episodes} episodes")
            print(f"Evaluating model over {eval_episodes} episodes...")
            
            try:
                eval_results = evaluator.run_evaluation(
                    num_episodes=eval_episodes,
                    render_every=render_every,
                    obstacle_num=args.obstacle_num,
                    moving_obstacle_ratio=args.moving_ratio
                )
                
                # Plot evaluation results
                evaluator.plot_results(eval_results)
                
                # Print evaluation summary
                print("\n" + "="*80)
                print(f" {args.model.upper()} {args.dimension} EVALUATION RESULTS")
                print("="*80)
                print(f"Success rate: {eval_results['success_rate']:.2%}")
                print(f"Collision rate: {eval_results['collision_rate']:.2%}")
                print(f"Average reward: {eval_results['avg_reward']:.2f}")
                print(f"Average path length: {eval_results['avg_path_length']:.2f} steps")
                print(f"Static obstacle success: {eval_results['static_success_rate']:.2%}")
                print(f"Moving obstacle success: {eval_results['moving_success_rate']:.2%}")
                print("="*80)
                
                print(f"\nEvaluation results saved to {PATHS['eval_path']}/{args.model}_{args.dimension}_eval_results.png")
            except KeyboardInterrupt:
                print("\nEvaluation interrupted by user.")
                logger.warning("Evaluation interrupted by user.")
            except Exception as e:
                print(f"\nError during evaluation: {str(e)}")
                logger.error(f"Error during evaluation: {str(e)}")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        logger.warning("Training interrupted by user.")
        # Still save the model in case of interruption
        interrupted_path = os.path.join(model_path, f"{args.model}_{args.dimension}_interrupted.pth")
        torch.save(dqn.model.state_dict(), interrupted_path)
        logger.info(f"Interrupted model saved to {interrupted_path}")
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        print(f"Error training {args.model} in {args.dimension} environment: {str(e)}")

if __name__ == "__main__":
    main()