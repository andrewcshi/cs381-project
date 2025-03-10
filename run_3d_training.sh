#!/bin/bash
# Script to train and evaluate obstacle avoidance in 2D and 3D environments

# Function to print fancy headers
print_header() {
    echo
    echo "=============================================================================="
    echo "$1" | tr '[:lower:]' '[:upper:]' | sed 's/^/ /' | sed 's/$/ /'
    echo "=============================================================================="
    echo
}

# Default parameters
EPISODES=500
OBSTACLES=5
MOVING_RATIO=0.8
MODEL="both"
DIMENSION="both"
EVAL=true
RENDER=false
DEBUG=false
EVAL_EPISODES=200  # Add this new parameter for evaluation episodes

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --episodes)
      EPISODES="$2"
      shift 2
      ;;
    --obstacles)
      OBSTACLES="$2"
      shift 2
      ;;
    --moving-ratio)
      MOVING_RATIO="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --dimension)
      DIMENSION="$2"
      shift 2
      ;;
    --no-eval)
      EVAL=false
      shift
      ;;
    --eval)
      EVAL=true
      shift
      ;;
    --eval-episodes)  # Add this new option
      EVAL_EPISODES="$2"
      shift 2
      ;;
    --render)
      RENDER=true
      shift
      ;;
    --debug)
      DEBUG=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: ./run_3d_training.sh [--episodes NUM] [--obstacles NUM] [--moving-ratio RATIO] [--model lstm|transformer|both] [--dimension 2D|3D|both] [--no-eval] [--eval-episodes NUM] [--render] [--debug]"
      exit 1
      ;;
  esac
done

# Create necessary directories
mkdir -p weights/lstm weights/transformer figures logs evaluation

# Set timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
print_header "Starting Training Run: ${TIMESTAMP}"
echo "Episodes: $EPISODES"
echo "Obstacles: $OBSTACLES"
echo "Moving Ratio: $MOVING_RATIO"
echo "Model(s): $MODEL"
echo "Dimension(s): $DIMENSION"
echo "Evaluation: $EVAL"
echo "Evaluation Episodes: $EVAL_EPISODES"  # Output the evaluation episodes
echo "Rendering: $RENDER"
echo "Debug Mode: $DEBUG"

# Determine which models to run
MODELS=()
if [ "$MODEL" = "both" ]; then
  MODELS=("lstm" "transformer")
elif [ "$MODEL" = "lstm" ] || [ "$MODEL" = "transformer" ]; then
  MODELS=("$MODEL")
else
  echo "Invalid model: $MODEL"
  echo "Must be 'lstm', 'transformer', or 'both'"
  exit 1
fi

# Determine which dimensions to run
DIMENSIONS=()
if [ "$DIMENSION" = "both" ]; then
  DIMENSIONS=("2D" "3D")
elif [ "$DIMENSION" = "2D" ] || [ "$DIMENSION" = "3D" ]; then
  DIMENSIONS=("$DIMENSION")
else
  echo "Invalid dimension: $DIMENSION"
  echo "Must be '2D', '3D', or 'both'"
  exit 1
fi

# Build evaluation flags
EVAL_FLAG=""
if [ "$EVAL" = true ]; then
  EVAL_FLAG="--eval"
fi

RENDER_FLAG=""
if [ "$RENDER" = true ]; then
  RENDER_FLAG="--render"
fi

DEBUG_FLAG=""
if [ "$DEBUG" = true ]; then
  DEBUG_FLAG="-v"  # Verbose mode for Python if needed
fi

# Run the training for each combination
for model in "${MODELS[@]}"; do
  for dimension in "${DIMENSIONS[@]}"; do
    print_header "Training ${model} model in ${dimension} environment"
    
    python $DEBUG_FLAG train_3d.py \
      --model "$model" \
      --dimension "$dimension" \
      --episodes "$EPISODES" \
      --obstacle-num "$OBSTACLES" \
      --moving-ratio "$MOVING_RATIO" \
      --eval-episodes "$EVAL_EPISODES" \
      $EVAL_FLAG $RENDER_FLAG
    
    # Check if the training was successful
    if [ $? -ne 0 ]; then
      echo "Error training ${model} in ${dimension} environment"
      echo "Check logs for details"
    fi
  done
done

# If we trained multiple configurations, run comparison
if [ ${#MODELS[@]} -gt 1 ] || [ ${#DIMENSIONS[@]} -gt 1 ]; then
  print_header "Running Comparison Across All Models and Dimensions"
  
  # Only run if evaluation is enabled
  if [ "$EVAL" = true ]; then
    # Pass the number of evaluation episodes to evaluate.py
    python evaluate.py --dimension both --eval-episodes $EVAL_EPISODES
  else
    echo "Skipping comparison (evaluation not enabled)"
    echo "To run comparison, use --eval flag"
  fi
fi

print_header "Training Completed: ${TIMESTAMP}"
echo "Results can be found in the 'evaluation' directory"
echo "Model weights are stored in the 'weights' directory"