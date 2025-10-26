#!/bin/bash
# generate_models.sh

# runs python scripts in order to generate 
# and parse results of svm models for all
# combinations of hyperparameters


PYTHON_SCRIPT_TRAIN="TrainSVC.py"
PYTHON_SCRIPT_RESULTS="parse_results.py"

# output pickle file
RESULTS_PICKLE="results.pkl"

# hyperparameter sets to iterate over
KERNELS=("linear" "rbf" "poly" "sigmoid")
C_VALUES=(0.1 1 10 100)
GAMMA_VALUES=('scale' '0.01' '0.1' '10')

# reproducibility parameters
TEST_SIZE=0.2
SEED=42

# loops every combination of hyperparameters
for KERNEL in "${KERNELS[@]}"; do

  if [[ "$KERNEL" == "linear" ]]; then
    for C in "${C_VALUES[@]}"; do

      echo "Running: kernel=$KERNEL, C=$C, test_size=$TEST_SIZE, seed=$SEED"

      python "$PYTHON_SCRIPT_TRAIN" \
        --kernel "$KERNEL" \
        --C "$C" \
        --test_size "$TEST_SIZE" \
        --seed "$SEED" \
        --results_pickle "$RESULTS_PICKLE"
    done


  else 
    for C in "${C_VALUES[@]}"; do
      for GAMMA in "${GAMMA_VALUES[@]}"; do
        echo "Running: kernel=$KERNEL, C=$C, gamma=$GAMMA, test_size=$TEST_SIZE, seed=$SEED"
        
        # Call Python script with arguments
        python "$PYTHON_SCRIPT_TRAIN" \
          --kernel "$KERNEL" \
          --C "$C" \
          --gamma "$GAMMA" \
          --test_size "$TEST_SIZE" \
          --seed "$SEED" \
          --results_pickle "$RESULTS_PICKLE"
      done
    done
  fi
done

echo "Parsing Training Results"
python "$PYTHON_SCRIPT_RESULTS"