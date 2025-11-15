#!/bin/bash
# generate_models.sh

# runs python scripts in order to generate 
# and parse results of svm models for all
# combinations of hyperparameters


PYTHON_SCRIPT_TRAIN="NN_train.py"
PYTHON_SCRIPT_TEST="NN_test.py"


# hyperparameter sets to iterate over
BATCHSIZES=(16 32 64 128)
LEARNINGRATES=(0.001 0.0005 0.0001)
DROPOUTS=(0.0 0.1 0.2 0.3)
MAXEPOCHS=(100 200 300)
# reproducibility parameters

# loops every combination of hyperparameters
for BATCHSIZE in "${BATCHSIZES[@]}"; do
  for LEARNINGRATE in "${LEARNINGRATES[@]}"; do
    for DROPOUT in "${DROPOUTS[@]}"; do
      for MAXEPOCH in "${MAXEPOCHS[@]}"; do

        echo "Running: batch size=$BATCHSIZE, learning rate=$LEARNINGRATES, dropout=$TEST_SIZE, max epochs=$SEEDMAXEPOCHS"

        python "$PYTHON_SCRIPT_TRAIN" \
          --bs "$BATCHSIZE" \
          --lr "$LEARNINGRATE" \
          --max_epoch "$MAXEPOCH" \
          --dropout "$DROPOUT" \
            
      done
    done
  done
done

python "$PYTHON_SCRIPT_TEST"
