import pandas as pd
import os
import numpy as np
from sklearn.impute import KNNImputer 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report,\
                            accuracy_score,\
                            confusion_matrix,\
                            precision_score,\
                            recall_score,\
                            f1_score


# Function: imports a dataframe from csv
def get_csv(csv_name: str) -> pd.DataFrame:

  '''
    Parameters: 'csv_name' string <name>.csv of csv file in data folder
    Provides current directory for error handling
    Output: pandas dataframe object
  '''
  # base directory relative to current file
  base_dir = os.getcwd()

  print(f"~~~~\n{base_dir}\n~~~~")

  # generate path
  data_path = os.path.join(base_dir, "data", csv_name)

  try:
    data = pd.read_csv(data_path)
    return data
  except:
    print("put data in \"data\" folder in top layer of project")
    print(data_path)

  

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
  df_info = pd.DataFrame({
    'attribute': df.columns,
    'data_type': df.dtypes,
    'num_null': df.isnull().sum(),
    'num_unique': df.nunique()
  })

  stats = df.describe()
  df_stats = stats.T.reset_index(drop = True)
  df_stats["attribute"] = stats.columns
  df_stats.head()
  df_stats = df_stats[[df_stats.columns[-1]] + list(df_stats.columns[:-1])]

  df_summary = df_stats.merge(df_info,
                              on = "attribute",
                              how = "right").drop('count', axis =1)
  
  return {'nrows': df.shape[0],
          'ncolumns': df.shape[1],
          'head': df.head(),
          'summary_table': df_summary}


def hot_deck(df: pd.DataFrame, attr: str, n_neighbors: int = 3) -> pd.Series:
    """
    Hot-deck KNN imputation for a single column using all other columns.
    
    Parameters:
        df : pd.DataFrame
            DataFrame containing numeric and/or categorical columns.
        attr : str
            The column to impute.
        n_neighbors : int
            Number of neighbors for KNN.
    
    Returns:
        pd.Series
            Imputed column with the same index as df.
    """
    if attr not in df.columns:
        raise ValueError(f"Column '{attr}' not found in DataFrame.")

    # Identify categorical columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    # One-hot encode if there are categorical columns
    if len(cat_cols) > 0:
        df_cat_encoded = pd.get_dummies(df[cat_cols])
        df_num = df.drop(cat_cols, axis=1)
        imputer_input = pd.concat([df_num, df_cat_encoded], axis=1)
    else:
        # All numeric
        imputer_input = df.copy()

    # KNN imputer
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_array = imputer.fit_transform(imputer_input)

    # Return the imputed column
    return pd.Series(imputed_array[:, imputer_input.columns.get_loc(attr)], index=df.index)



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

def train_knn(k_candidates: list, 
              X_train, y_train, 
              X_test, y_test)->dict:
  '''
    Performs training of k-nearest neighbors model over set of k candidate
    hyperparameters; provides performance measures & plots

    params:
      - list of k-candidates
      - training (feature tuple set and associated targets) and testing sets

    returns: dict including
      - classification report for each k (performance measures/class)
      - overall performance for each k (aggregate performance measures)
      - confusion matrix for each k
      - model for each k
      - accuracy vs k-candidate for all parameters

  '''
  
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.fit_transform(X_test)

  model_dict = {}

  for k in k_candidates:
    #place holder dict to provide dict as value for each model_dict key
    eval_dict = {} 
    # fit model with current k 
    knn = KNeighborsClassifier(n_neighbors=k)

    model = knn.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)


    # create current conf matrix 
    cm = confusion_matrix(y_test, y_pred)

    # store eval (acc, precision, recall, F1)
    agg_measures = {
      'accuracy': accuracy_score(y_test,y_pred),
      'precision': precision_score(y_test, y_pred, average='macro'),
      'recall': recall_score(y_test, y_pred, average='macro'), 
      'f1': f1_score(y_test, y_pred, average='macro')
    }
    # get current classification_report for class level evaluation
    cr = classification_report(y_test,y_pred)

    # add model, conf matrix, agg_measures, and classification_report in eval_dict
    eval_dict['model'] = model
    eval_dict['confusion_matrix'] = cm
    eval_dict['agg_measures'] = agg_measures
    eval_dict['classification_report'] = cr
    # assign current eval_dict as value to current k as key in model_dict
    model_dict[k] = eval_dict


  # create accuracy vs k candidate plot and add to model_dict
  accuracy = [model_dict[k]['agg_measures']['accuracy'] for k in k_candidates]

  fig, ax = plt.subplots()
  ax.plot(k_candidates,\
          accuracy,\
          color = "orange",
          marker = "."
          )
  
  ax.set_title("Accuracy vs k-Candidate")
  ax.set_xlabel("k(Number of Neighbors)")
  ax.set_ylabel("Model Accuracy")

  model_dict['accuracy_plot'] = fig
  plt.close(fig)
  
  # create aggregate performance report table and add to model_dict
  aggs = [(model_dict[k]['agg_measures']) for k in k_candidates]
  
  aggs_df = pd.DataFrame(aggs)
  aggs_df.insert(0, 'k_candidates', k_candidates)
  model_dict['aggregate_performance_table'] = aggs_df

  return model_dict


if __name__ == "__main__":
  
  ##### Initial evaluation of dataset #####
  df = get_csv("mobile_price.csv")

  # overview of attributes
  df_overview = dataframe_overview(df) 

  # prints exploratory information about the data set
  print("\n\nTask 1: Load and Inspect the Data\n")
  print(f"number of rows: {df_overview['nrows']}")
  print(f"number of columns: {df_overview['ncolumns']}")
  print(f"\n\ndf head:\n {df_overview['head']}")
  print(f"\n\ndf summary:\n {df_overview['summary_table']}")
  # prints the overview with latex syntax for reporting
  print(f"\n\nlatex table:\n {df_overview['summary_table'].to_latex(index = False)}")

  # assesses 0 = NaN scenarios 
  df[["fc", "px_height", "sc_w"]] = df[["fc", "px_height", "sc_w"]].replace(0,np.nan)

  # clarifies hidden nulls
  df_overview2 = dataframe_overview(df)
  print(f"\n\ndf summary2:\n {df_overview2['summary_table']}")


  # imputations
  df['fc'] = hot_deck(df, 'fc')
  df['sc_w'] = hot_deck(df, 'sc_w')
  df['px_height'] = hot_deck(df, 'px_height')

  # final overview with exposed nulls imputed with hot_deck()
  df_overview3 = dataframe_overview(df)
  print(f"\n\nfinal df summary:\n {df_overview3['summary_table']}\n")

  # separates independent(training) and dependent(classification) attributes
  X,y = get_train_target(df, 'price_range')

  print(f"Training attributes:\n {X.columns}")

  print(f"Target attribute: {y.name}\n")

  ##### Model fitting and parameter tuning #####

  # generates the training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # creates k-value candidates
  k_candidates = list(range(1,17,2))
  
  # gets models
  models = train_knn(k_candidates, X_train, y_train, X_test, y_test)



  # display each classification report with Title identifying k using comprehension
  cr_gen = (models[k]['classification_report'] for k in k_candidates)

  for k,cr in zip(k_candidates,cr_gen):
    print(f"{k} Neighbors Classification Report:\n {cr}\n")

  # display each confusion matrix with Title identifying k using comprehension

  cm_gen = (models[k]['confusion_matrix'] for k in k_candidates)

  for k,cm in zip(k_candidates,cm_gen):
    print(f"Saving {k} Neighbors Confusion Matrix\n")

    fig_cm, ax_cm = plt.subplots(figsize=(5,4))
  
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", cbar=False, ax = ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("True")
    ax_cm.set_title(f"{k} Neighbors Confusion Matrix")
    fig_cm.tight_layout()
    fig_cm.savefig(f"{k}_Neighbors confusion_matrix.png", dpi=300)
    fig_cm.show()
    plt.pause(3)
    plt.close(fig_cm)

  # display agg_table
  print(f"Aggregate Performance Table:\n {models['aggregate_performance_table']}")

  # display accuracy plot
  acc_plot = models['accuracy_plot']
  acc_plot.savefig("accuracy_vs_k.png", dpi=300)

  print(f"Saving Accuracy Plot\n")
  plt.figure(acc_plot)   # make it the active figure
  plt.show(block = False) 
  plt.pause(3)
  plt.close()  

  print("All figures are saved to project folder!")