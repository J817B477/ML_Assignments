The files work this way:

1. NN_dataset.py:
  - defines the class for the test and train Dataset objects used
1. NN_MLP.py:
  - defines the MLP architecture as a class
1. NN_train.py
  - used to fit MLPs and tune hyper parameters

1. NN_test.py 
  - used to test results

1. tune_hyperParams.sh:
  - used to iteratively train MLPs on different hyperparameter combinations
  - calls NN_train.py and NN_test.py 

1. analyze_results.py
  - used to find the best model and get visuals for report

### to recreate first run tune_hyperParams.sh then run analyze_results.py

folders:
- models: sotres all of the MLPs fitted
- plots: stores the auc and confusion matrix heatmap used in the report

