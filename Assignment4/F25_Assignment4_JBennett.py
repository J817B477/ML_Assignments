'''
John Bennett
Assignment 4: Support Vector Machines
Fall 2025

approach: This script is designed to rely on argparse to input arguments when calling the script with flags from the terminal. In accordance with this, the results of each script are stored in a pickle that will be called from another script designed to parse the results of those terminal calls. The calls will be handled by a bash script and all of the steps that are not simple calls are written into functions that are called when __name__ is __Main__. A sklearn Pipeline object handles the fitting of the svm model for each terminal call of this script. 
'''
import pandas as pd
import os
import pickle
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import time
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix
from sklearn import datasets

# generates overview of dataframe
def dataframe_overview(df: pd.DataFrame)-> dict:

  '''
    Parameters: 'df' pandas dataframe that overview is created for
    Output: dictionary containing information for dataset
      - 'nrows': number of rows in the dataframe
      - 'ncolumns': number of columns in the dataframe
      - 'head': contains 1st 5 rows of dataframe
      - 'summary_table': containing pandas dataframe providing:
          - attribute: name of columns of the passed dataframe
          - mean: mean of each attribute
          - std: standard deviation of each attribute
          - min: minimum value of each variable
          - 25%: value of first quartile of each variable
          - 50%: value of second quartile of each variable (median)
          - 75%: value of third quartile of each variable
          - max: maximum value of each variable
          - data_type: the data type of each attribute
          - number_null: the number of null (type) values in each attribute
          - number_unique: the number of unique values in each attribute
  '''

  # generates initial summary df with attributes as records
  df_info = pd.DataFrame({
    'attribute': df.columns,
    'data_type': df.dtypes,
    'num_null': df.isnull().sum(),
    'num_unique': df.nunique()
  })

  # creates df of stats for numeric data
  stats = df.describe()
  df_stats = stats.T.reset_index(drop = True)
  df_stats["attribute"] = stats.columns
  df_stats.head()

  # reorders columns to support full merger layout
  df_stats = df_stats[[df_stats.columns[-1]] + list(df_stats.columns[:-1])]

  # merger joins two summary dfs into one
  df_summary = df_stats.merge(df_info,
                              on = "attribute",
                              how = "right").drop('count', axis =1)
  
  # returns summary df with df's number of rows, columns and 1st 5 rows
  return {'nrows': df.shape[0],
          'ncolumns': df.shape[1],
          'head': df.head(),
          'summary_table': df_summary}

# separates the feature vectors from their targets 
def get_train_target(df: pd.DataFrame, target: str)-> tuple[pd.Series,pd.DataFrame]:
    '''
      Takes the dataframe and the name of its attribute containing
      the class values and returns separated target attribute and 
      agnostic dataset

      Parameters:
        - 'df' of source data for model
        - 'target' attribute containing classes
      
      Output:
        - tuple containing:
          - Series containing the target values
          - Data frame with omitted target values
    '''
    X = df.drop(target, axis=1, inplace=False)
    y = df[target]

    return X,y


    '''
      Takes the dataframe and the name of its attribute containing
      the class values and returns separated target attribute and 
      agnostic dataset

      Parameters:
        - 'df' of source data for model
        - 'target' attribute containing classes
      
      Output:
        - tuple containing:
          - Series containing the target values
          - Data frame with omitted target values
    '''
    X = df.drop(target, axis=1, inplace=False)
    y = df[target]

    return X,y

# store results of model training and testing to pickle
def store_results(file_name: str, new_dict: dict, ) -> None:
  '''
  Function that handles storage of collected outputs from Support Vector Machine model training
  '''


  # brings in any preexisting results to append new results to
  if os.path.isfile(file_name):
    with open(file_name, "rb") as f:
      results_dict = pickle.load(f)
    
  else:
    results_dict = {}

  # appends to results dictionary
  results_dict.update(new_dict)

  with open(file_name, "wb") as f:
        pickle.dump(results_dict, f)

  print(f"{file_name} updated with {list(new_dict.keys())[0]} model results.")


# Trains model: with terminal args
def train_svm(source_data: pd.DataFrame, target_name: str, args: argparse.ArgumentParser):
  X,y = get_train_target(source_data, target_name)
  X_train, X_test, y_train, y_test = train_test_split(
    X,y,
    test_size=args.test_size,
    random_state=args.seed,
    stratify = y
  )

  # handles linear kernel with default/passed arg.gamma
  if args.kernel == "linear":

    print("Warning: for linear model, gamma argument ignored.")
    args.gamma = None

    pipe = Pipeline(
      steps=[
          ("scaler", StandardScaler()),
          ("svm", SVC(kernel=args.kernel, C=args.C))
      ]
    )

    start = time.time()
    pipe.fit(X_train,y_train)
    end = time.time()

    # dict key string
    model_key = f"{args.kernel}_{args.C}"

  else: 
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel=args.kernel, C=args.C, gamma=args.gamma))
        ]
    )

    start = time.time()
    pipe.fit(X_train,y_train)
    end = time.time()
    
    # dict key string
    model_key = f"{args.kernel}_{args.C}_{args.gamma}"

  # pipe auto standardizes X_test (annotation for future self)
  y_pred = pipe.predict(X_test)

  # results dict: locals allows dynamic variable naming
  results_dict = {
     'model': pipe,
     'c_scalar': args.C,
     'gamma': args.gamma,
     'number_support_vectors': pipe.named_steps["svm"].n_support_,
     'training_time': end-start,
     'accuracy': accuracy_score(y_test, y_pred),
     'classification_report': classification_report(y_test,y_pred),
     'confusion_matrix': confusion_matrix(y_test, y_pred),
     'training_time': end-start
  }

  # call store_results() to add new dict to pickle
  store_results(args.results_pickle, {model_key: results_dict})


# handles terminal arguments
def get_arg():
  #creates parser object
  parser = argparse.ArgumentParser (description= "Handles arguments passed for defining hyperparameters for training SVM models on randomly generated fixed arbitrary dataset.")

  # where results are stored
  parser.add_argument("--results_pickle", type = str, default= "results.pkl")

  # hyperparameters
  parser.add_argument("--kernel", type=str, default="linear", choices=["linear", "rbf", "poly", "sigmoid"])
  parser.add_argument("--C", type=float, default=1.0)
  parser.add_argument("--gamma", type=float, default=0.1)
  parser.add_argument("--test_size", type=float, default=0.2)
  parser.add_argument("--seed", type=int, default=42)
  return parser.parse_args()


############# Driver Code ############

if __name__ == "__main__":
  ## choose one of the dataset options
  X,y = datasets.make_classification(
    n_samples=40000,
    n_features=10,
    n_informative=10,
    n_redundant=0,
    n_classes=2,
    n_clusters_per_class=3,
    flip_y=0.03,
    weights=[0.35,0.65],
    random_state=42,
    hypercube=True
  )

  # for the sake of emulating real dataset
  feature_names = [f'feature_{i}' for i in range(X.shape[1])]

  X = pd.DataFrame(X,columns=feature_names)
  y = pd.Series(y,name="target")

  df = pd.concat([X,y],axis=1)

  args = get_arg()

  ## add metadata to storage using store_results
  overview = dataframe_overview(df)
  store_results(args.results_pickle,{'meta':overview})

  ## train model with train_svm() call
  train_svm(df,"target", args)

# import pickle
# with open("results.pkl", "rb") as f:
#   results = pickle.load(f)

# print(results)