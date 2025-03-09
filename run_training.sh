# run_training.sh

#!/bin/bash
# Full training script for obstacle avoidance with both LSTM and Transformer models

# Function to print fancy headers
print_header() {
    echo
    echo "============================================================================="
    echo "$1" | tr '[:lower:]' '[:upper:]' | sed 's/^/ /' | sed 's/$/ /'
    echo "============================================================================="
    echo
}

# Default number of episodes
EPISODES=500

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --episodes)
      EPISODES="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: ./run_training.sh [--episodes NUM] [--model lstm|transformer|both]"
      exit 1
      ;;
  esac
done

# Create necessary directories
mkdir -p weights/lstm weights/transformer figures logs evaluation

# Set timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
print_header "Starting Training Run: ${TIMESTAMP}"

# Run the experiment(s)
if [ -z "$MODEL" ] || [ "$MODEL" = "both" ]; then
  print_header "Training both LSTM and Transformer models (${EPISODES} episodes each)"
  python run_experiments.py --models both --episodes ${EPISODES}
elif [ "$MODEL" = "lstm" ]; then
  print_header "Training LSTM model (${EPISODES} episodes)"
  python run_experiments.py --models lstm --episodes ${EPISODES}
elif [ "$MODEL" = "transformer" ]; then
  print_header "Training Transformer model (${EPISODES} episodes)"
  python run_experiments.py --models transformer --episodes ${EPISODES}
else
  echo "Invalid model: ${MODEL}"
  echo "Must be 'lstm', 'transformer', or 'both'"
  exit 1
fi

# Run evaluation after training
print_header "Running Evaluation On Trained Models"
python evaluate.py