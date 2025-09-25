import pandas as pd
import os
import numpy as np
from sklearn.impute import KNNImputer 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import math
import multiprocessing
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
import pickle
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns


def get_csv(csv_name: str) -> pd.DataFrame:
  # base directory relative to current file
  base_dir = os.getcwd()

  print(f"~~~~\n{base_dir}\n~~~~")

  # generate path
  data_path = os.path.join(base_dir, "data", csv_name)

  try:
    data = pd.read_csv(data_path)
    return data
  except:
    print("put data in data folder of top layer of project")
    print(data_path)

  

def dataframe_overview(df):

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
   
   X = df.drop(target, axis=1, inplace=False)
   y = df[target]

   return X,y

def train_rf(X_train, y_train, random_state = 42, param_grid=None, cv=5, scoring='roc_auc_ovr'):
    length_train = X_train.shape[0]

    crf = RandomForestClassifier(random_state=random_state)

    # gets number of trees
    n_trees = [50, 100, 200, 300, 500]

    # gets hyperparameter candidate ranges relative to the size of the training data 
    # hyperparameter conventions drive me crazy (because I dont really appreciate why the are what they are but they matter)
    # I am looking for ways to define them systematically

    # max_depth value to build the range around
    base_depth = math.log2(length_train)

    # heuristic length of max_depths tests
    depth_candidate_list = max(8, int(base_depth))

    # 
    depth_range = math.log10(length_train)

    # this can be better but I wanted to not get too carried away right now
    exponent_values = np.linspace(1 - 0.5*depth_range, 1 + 0.5*depth_range, depth_candidate_list)

    # makes depths list
    max_depth_candidates = [max(2, int(base_depth*exp)) for exp in exponent_values]

    # handles redundancies in depth_candidates
    unique_depths = np.unique(max_depth_candidates)
    if len(unique_depths) < depth_candidate_list:
        max_depth_candidates = np.linspace(min(unique_depths), max(unique_depths), depth_candidate_list, dtype=int)

    # handles the minimum number of records in a node to split on 
    frac_train_set = [0.01, 0.02, 0.05, 0,1, 0.15]
    min_split_candidates = [max(2, int(length_train * f)) for f in frac_train_set]


    # handles redundancies in depth candidates
    unique_splits = np.unique(min_split_candidates)
    if len(unique_splits) < depth_candidate_list:
        unique_splits = np.linspace(min(unique_splits), max(unique_splits), depth_candidate_list, dtype=int)

    # dict is a arg for GridSearchCV
    param_grid = {
        'n_estimators': n_trees,
        'max_depth': max_depth_candidates,
        'min_samples_split': min_split_candidates,
    }

    # allocates cores for efficiency
    total_cores = multiprocessing.cpu_count()
    n_jobs = max(1, total_cores - 2)

    grid_search = GridSearchCV(
        estimator=crf,
        param_grid=param_grid,
        cv=cv,
        scoring= scoring,
        return_train_score=True,
        n_jobs=n_jobs,
        verbose=True
    )

    grid_search.fit(X_train,y_train)
    
    #gets results as a df
    results_df = pd.DataFrame(grid_search.cv_results_)
   
    # edits the results_df
    display_columns = ['params', 'mean_test_score', 'std_test_score', 'mean_train_score', 'std_train_score']

    results_df = results_df[display_columns].sort_values(by='mean_test_score', ascending=False).reset_index(drop=True)

    results_df['test_train_gap'] = results_df['mean_train_score'] - results_df['mean_test_score']

    # makes model with best hyperparams
    best_model = grid_search.best_estimator_
    
    return {
       'best_hyperparams': best_model.get_params(),
       'validation_results': results_df,
       'best_model': best_model
    }

# # >>>>>>>>>>>>>>>>>>>>>>>> driver code <<<<<<<<<<<<<<<<<<<<<<<
if __name__ == "__main__":

    ### Initial evaluation of dataset
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

    # generates the training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"training set size: {y_train.shape[0]}")
    print(f"testing set size: {y_test.shape[0]}\n")

    # generate model includes hyperparameter validation under hood
    model_dict = train_rf(X_train, y_train)

    print(model_dict['best_hyperparams'])

    # prints the top 5 results table of the model fitting with h-param tuning
    validation_table = model_dict['validation_results'][['params','mean_test_score','test_train_gap']].head()

    print(model_dict['validation_results'][['params','mean_test_score','test_train_gap']].head())

#     # save the top five to csv to convert to table in the report
#     validation_table.to_csv('data/best_model.csv', index = False)

#     #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#     # saves train_rf dict so I can stop rerunning it ;)
#     with open('model_dict.pkl', 'wb') as f:
#         pickle.dump(model_dict, f)
#     #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#     # gets the best model from storage
#     with open('model_dict.pkl', 'rb') as f:
#        model_dict = pickle.load(f)
#     #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    best_model = model_dict['best_model']

    # Predicting from the best model
    y_pred = best_model.predict(X_test)

    # gets performance results 
    performance_report = classification_report(y_test,y_pred)
    print(performance_report)

    #binarization of the y_test values
    classes = np.unique(y_test)
    y_test_binaries = label_binarize(y_test, classes=classes)

    # creates auc scores

    ## wraps best model in one vs rest classifier
    best_model_ovr = OneVsRestClassifier(best_model)

    ## fits the wrapped model
    best_model_ovr.fit(X_train, y_train)
    y_score = best_model_ovr.predict_proba(X_test) 

    ## get roc and auc values per class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i, cls in enumerate(classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binaries[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])


    plt.figure(figsize=(8, 6))

    for i, cls in enumerate(classes):
        plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {cls} (AUC = {roc_auc[i]:.2f})')
        plt.fill_between(fpr[i], tpr[i], alpha=0.1) 

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Mobile Phone Price Ranges')
    plt.legend(loc='lower right')
    plt.show()  


    # creates heat-mapped confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Multi-Class Random Forest Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.close() 
