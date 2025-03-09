# run_experiments.py

import subprocess
import time
import argparse
import os
import datetime
import logging
from config import TRAINING, PATHS

def setup_experiment_logger():
    """Set up a logger for the experiment runner."""
    # Create log directory if it doesn't exist
    os.makedirs(PATHS['log_path'], exist_ok=True)
    
    # Set up logging format
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{PATHS['log_path']}/experiment_{timestamp}.log"
    
    # Configure logger
    logger = logging.getLogger('experiment')
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
    
    # Print banner
    print("\n" + "*"*80)
    print(" OBSTACLE AVOIDANCE TRAINING EXPERIMENT ".center(80, "*"))
    print("*"*80 + "\n")
    
    return logger

def run_experiment(model_type, episodes, logger):
    """Run a training experiment with the specified model type."""
    logger.info(f"{'='*50}")
    logger.info(f"Starting experiment with {model_type.upper()} model for {episodes} episodes")
    logger.info(f"{'='*50}")
    
    # Print more visible separator to terminal
    print("\n" + "="*80)
    print(f" STARTING {model_type.upper()} MODEL TRAINING ".center(80, "="))
    print("="*80 + "\n")
    
    start_time = time.time()
    
    # Run the training process
    result = subprocess.run(
        ['python', 'train.py', '--model', model_type, '--episodes', str(episodes)],
        capture_output=True,
        text=True
    )
    
    # Process has completed
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Print completion banner
    print("\n" + "="*80)
    print(f" {model_type.upper()} TRAINING COMPLETE ".center(80, "="))
    print(f" Time: {int(hours)}h {int(minutes)}m {seconds:.2f}s ".center(80, "="))
    print("="*80 + "\n")
    
    # Log output
    if result.stdout:
        logger.info(f"Output from {model_type} training:")
        for line in result.stdout.splitlines():
            logger.info(f"  {line}")
    
    # Log errors if any
    if result.stderr:
        logger.error(f"Errors from {model_type} training:")
        for line in result.stderr.splitlines():
            logger.error(f"  {line}")
    
    logger.info(f"{'='*50}")
    logger.info(f"Finished {model_type.upper()} experiment in {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    logger.info(f"{'='*50}")
    
    return {
        'model_type': model_type,
        'elapsed_time': elapsed_time,
        'return_code': result.returncode,
        'success': result.returncode == 0
    }

def main():
    parser = argparse.ArgumentParser(description='Run obstacle avoidance experiments')
    parser.add_argument('--models', nargs='+', choices=['lstm', 'transformer', 'both'], 
                        default=['both'], help='Models to train')
    parser.add_argument('--episodes', type=int, default=TRAINING['num_episodes'], 
                        help='Number of episodes per experiment')
    args = parser.parse_args()
    
    # Set up logger
    logger = setup_experiment_logger()
    logger.info(f"Starting obstacle avoidance experiments with {args.episodes} episodes")
    
    # Determine which models to run
    models_to_run = []
    if 'both' in args.models:
        models_to_run = ['lstm', 'transformer']
    else:
        models_to_run = args.models
    
    logger.info(f"Models to train: {', '.join(model.upper() for model in models_to_run)}")
    
    # Create directories
    for path_type, path in PATHS.items():
        os.makedirs(path, exist_ok=True)
        logger.info(f"Ensured directory exists: {path}")
    
    # Run experiments
    results = []
    overall_start_time = time.time()
    
    for model in models_to_run:
        result = run_experiment(model, args.episodes, logger)
        results.append(result)
    
    # Calculate overall time
    overall_time = time.time() - overall_start_time
    hours, remainder = divmod(overall_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Print summary
    logger.info("\nExperiment Summary:")
    logger.info("-" * 60)
    for result in results:
        model = result['model_type']
        hours, remainder = divmod(result['elapsed_time'], 3600)
        minutes, seconds = divmod(remainder, 60)
        status = "SUCCESS" if result['success'] else "FAILED"
        logger.info(f"{model.upper()} training: {status} in {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    logger.info("-" * 60)
    logger.info(f"Total experiment time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    
    # Print summary to terminal for visibility
    print("\n" + "="*80)
    print(" EXPERIMENT SUMMARY ".center(80, "="))
    print("="*80)
    for result in results:
        model = result['model_type']
        hours, remainder = divmod(result['elapsed_time'], 3600)
        minutes, seconds = divmod(remainder, 60)
        status = "SUCCESS" if result['success'] else "FAILED"
        time_str = f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
        print(f" {model.upper()} Training: {status}".ljust(40) + f"Time: {time_str}".rjust(39))
    print("-"*80)
    hours, remainder = divmod(overall_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f" TOTAL EXPERIMENT TIME: {int(hours)}h {int(minutes)}m {seconds:.2f}s ".center(80, "-"))
    print("="*80 + "\n")
    
    # Compare performance if both models were run successfully
    lstm_result = next((r for r in results if r['model_type'] == 'lstm'), None)
    transformer_result = next((r for r in results if r['model_type'] == 'transformer'), None)
    
    if lstm_result and transformer_result and lstm_result['success'] and transformer_result['success']:
        lstm_time = lstm_result['elapsed_time']
        transformer_time = transformer_result['elapsed_time']
        
        if transformer_time < lstm_time:
            speedup = (lstm_time / transformer_time - 1) * 100
            logger.info(f"Transformer was {speedup:.2f}% faster than LSTM")
        elif lstm_time < transformer_time:
            speedup = (transformer_time / lstm_time - 1) * 100
            logger.info(f"LSTM was {speedup:.2f}% faster than Transformer")
        else:
            logger.info("Both models took approximately the same time")
    
    # Suggest next steps
    logger.info("\nNext steps:")
    logger.info("1. To evaluate the trained models, run: python evaluate.py")
    logger.info("2. View training plots in the 'figures' directory")
    logger.info("3. Trained models are saved in the 'weights' directory")
    
    return results

if __name__ == "__main__":
    main()