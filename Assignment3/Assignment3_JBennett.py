import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


# Function: imports a dataframe from csv
def get_csv(csv_name: str) -> pd.DataFrame:

  '''
    Parameters: 'csv_name' string <name>.csv of csv file in data folder
    Provides current directory for error handling
    Output: pandas dataframe object
  '''
  # base directory relative to current file
  base_dir = os.getcwd()

  # print(f"~~~~\n{base_dir}\n~~~~")

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

# function: linear model performance
def linear_performance(y_pred: pd.Series, y_test: pd.Series)-> dict:
  errors = y_test - y_pred
  sq_errors = errors**2
  sse = np.sum(sq_errors)
  mse = sse/len(y_pred)
  actual_mean = np.mean(y_test)
  sst = np.sum((y_test-actual_mean)**2)
  r_sq = 1-(sse/sst)

  return {'residuals': errors,
          'sse': sse,
          'mse': mse,
          'R_squared': r_sq}

def make_linear_model(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series)-> dict:

  Xtrain = X_train.copy()
  Xtest = X_test.copy()
  # preps training and testing X to handle categorical features
  categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns

  # handles independent data transformation if any categorical
  if len(categorical_cols) > 0:
    ct = ColumnTransformer(
    [('encoder', OneHotEncoder(drop='first'), categorical_cols)],
    remainder='passthrough'
    )

    ct.fit(Xtrain)

    Xtrain = ct.transform(Xtrain)
    Xtest = ct.transform(Xtest)

  lm = LinearRegression()
  lm.fit(Xtrain,y_train)
  y_pred = lm.predict(Xtest)

  model_dict = linear_performance(y_pred, y_test)

  model_dict['model'] = lm
  model_dict['predictions'] = y_pred

  return model_dict


if __name__ == "__main__":

  # imports dataframe
  df = get_csv("house_price_regression_dataset.csv")
  
  # generates summary report of df
  overview = dataframe_overview(df)
  
  # prints summary outputs
  print("\n~~~~~~~~~~~~~~~~~~~\nData Frame Overview\n~~~~~~~~~~~~~~~~~~~\n")
  
  print(f"dataframe rows: {overview['nrows']}")
  print(f"dataframe columns: {overview['ncolumns']}\n")
  print(f"dataframe top 5 rows:\n {overview['head']}\n")
  print(f"dataframe aggregates:\n {overview['summary_table']}")

  # print(f"overview latex_table:\n {overview['summary_table'].to_latex(index=False, float_format='%.2f'.__mod__)}")

  # Changes the data type of 'Neighborhood_Quality' to categorical
  print("\n~~~~~~~~~~~~~~~~~~~\nCleaning Changes\n~~~~~~~~~~~~~~~~~~~")
  print(f"Data Type of 'Neighborhood_Quality': {df['Neighborhood_Quality'].dtype}")

  df['Neighborhood_Quality'] = pd.Categorical(df['Neighborhood_Quality'])
 
  #confirms datatype change
  print(f"New Data Type of 'Neighborhood_Quality': {df['Neighborhood_Quality'].dtype}")

  # Fits the multiple linear regression model

  print("\n~~~~~~~~~~~~~~~~~~~\nBasic Model Fitting\n~~~~~~~~~~~~~~~~~~~")
  X,y = get_train_target(df, 'House_Price')
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  lin_model = make_linear_model(X_train, X_test, y_train, y_test)  

  print(f"R^2: {lin_model['R_squared']}")
  print(f"MSE: {lin_model['mse']}")
  print(f"Model Coefficients: \n{lin_model['model'].coef_}")
  print(f"Intercept: {lin_model['model'].intercept_}")

  # Creates scatter plot y_test v y_pred
  plt.scatter(x = y_test, y = lin_model['predictions'], alpha= .6)
  plt.plot(y_test, y_test, color='black', linestyle='--')
  plt.grid()
  plt.title("Actual vs Predicted: Linear Regression", fontsize = 14)
  plt.xlabel("Actual Home Price", fontsize = 11)
  plt.ylabel("Predicted Price", fontsize = 11)
  plt.savefig("actualVpredicted.png", dpi=300)
  plt.show()
  plt.close()

  # Creates scatter plot of y_test v Residuals
  plt.scatter(x = y_test, y = lin_model['residuals'], alpha= .6)
  plt.axhline(y=np.mean(lin_model['residuals']), color='black', linestyle='--', linewidth=1) 
  plt.grid()
  plt.show()
  plt.close()

  # Creates scatter plot for individual features
  col_names = X_train.columns

  for i in range(len(col_names)):
      name = col_names[i]
    
      if X_train[name].dtype =="object" or X_train[name].dtype =="category":
        continue

      model = make_linear_model(pd.DataFrame(X_train[name]), pd.DataFrame(X_test[name]),y_train,y_test)

      
      coef = model['model'].coef_
      intercept = model['model'].intercept_

      x_values = X_train[name]
      y_values = (coef * x_values.astype(float)) + intercept
      plt.scatter(x_values, y_train, alpha= 0.6, label = "Train Data")
      plt.plot(x_values,y_values, color = "red", label = "Fitted Line")
      plt.xlabel(name)
      plt.ylabel('House_Price')
      plt.title(f"Fitted Line against {name}/House_Price Bivariate Points")
      plt.legend()
      plt.show()
      
