# Assignment 4: Support Vector Machines Code Explanation
One of the available options presented in class for this assignment was to experiment with combinations of hyperparameters by calling a training script in the terminal with flags that are passed to the script by way of the `argparse` python library. To navigate this in accordance with the expectations of the assignment for comparing models, three scripts were produced:
- `generate_models.sh`: a shell script designed to call a python script that trains a model for each of the combinations of hyperparameters we were asked to train with as well as call another python script that parses the accumulated results
- `TrainSVC.py`: a python script that trains the model and stores the relevant analysis results, based on the scope of the assignment, to a `results` pickle file
- `parse_results.py`: a python script that processes the `results` pickle file so that an aggregate perspective could be established for evaluating the effects of these parameters and the best combination of parameters could be determined

The system created between these scripts ensures that all of the combinations are assessed in a collective context, storing the results in their totality while establishing a modular approach where individual scripts can be modified without having to correct downstream implications within a single file.

#### To recreate the results run the `generate_models.sh` script in a bash terminal with the project folder `Assignment4` as the current working directory. 
