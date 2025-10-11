import pandas as pd
import os
import numpy as np
from pandas.api.types import CategoricalDtype
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.inspection import permutation_importance
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
  '''
    provides statistical measures associated with a linear model

    Parameters:
    - 'y_pred': generated predicted values from trained model
    - 'y_test': ground truth values associated with testing feature vectors

    Output:
    - dict containing:
      -'residuals': the distances between the predicted and ground truth target values
      -'sse': sum of squared differences between predicted and actual values (errors)
      -'mse': mean of sum of squared errors
      -'R_squared': measure of variance in data explained by model
  '''
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

def make_linear_model(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, degree = 0, ridge = False, alpha = 1.0)-> dict:

  '''
    makes linear models with or without polynomial features and with or without regularization

    Parameters:
      - X_train: Dataframe of training feature vectors
      - X_test: Dataframe of testing features passed to the trained model
      - y_train: Series of true target values for training
      - y_test: Series of true target values compared to model predictions
      - degree: (default = 0) degree of polynomialized features
      - ridge: boolean that clarifies if modeli regularized
      - alpha: (default = 1.0) factor of penalty term in adjusted sse of regularized model
    
    Output:
      - dict including
        - model: fitted model object containing coefficients and intercept
        - residuals: errors between predicted and test outputs
        - sse: sum of squared errors
        - mse: mean sum of squared errors
        - R_squared: performance measure indicating variance in data explained my model
        - predictions: target values produced by model using X_test
        - feature performance: measures of what features are most impactful in model
        - feature names: names of features in model reflecting any categorical of polynomial transformations
  '''

  #### preps X sets to handle transformations ####

  # ensures that the passed data remains unaffected
  Xtrain = X_train.copy()
  Xtest = X_test.copy()
  
  # important for feature transformation branches
  categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
  numeric_cols = X_train.select_dtypes(exclude=['object', 'category']).columns


  #### handles polynomial transformations ####
  if degree > 0 and len(numeric_cols) > 0:
        
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        Xtrain_poly = pd.DataFrame(
            poly.fit_transform(Xtrain[numeric_cols]),
            columns=poly.get_feature_names_out(numeric_cols),
            index=Xtrain.index
        )
        Xtest_poly = pd.DataFrame(
            poly.transform(Xtest[numeric_cols]),
            columns=poly.get_feature_names_out(numeric_cols),
            index=Xtest.index
        )

        # replace numeric features with expanded polys
        Xtrain = pd.concat([Xtrain.drop(columns=numeric_cols), Xtrain_poly], axis=1)
        Xtest = pd.concat([Xtest.drop(columns=numeric_cols), Xtest_poly], axis=1)

  #### handles categorical feature transformations ####
  ordinal_cols = []
  nominal_cols = []

  if len(categorical_cols) > 0:
    
    ordinal_cols = [col for col in categorical_cols 
                    if pd.api.types.is_categorical_dtype(Xtrain[col]) 
                    and Xtrain[col].cat.ordered]

    nominal_cols = [col for col in categorical_cols if col not in ordinal_cols]

    # handle nominal with OneHotEncoder
    if len(nominal_cols) > 0:
      ct = ColumnTransformer(
        [('encoder', OneHotEncoder(drop="first", sparse_output=False), nominal_cols)],
        remainder='passthrough'
      )
      ct.fit(Xtrain)
      Xtrain = ct.transform(Xtrain)
      Xtest = ct.transform(Xtest)
      feature_names = ct.get_feature_names_out()
      feature_names = [n.split("__", 1)[-1] for n in feature_names]
    else:
      ct = None

  # convert ordinal categorical to numeric codes
  for col in ordinal_cols:
      Xtrain[col] = Xtrain[col].cat.codes
      Xtest[col] = Xtest[col].cat.codes

  # Rebuild feature names in the original column order
  feature_names = []
  for col in X_train.columns:
      if col in nominal_cols and ct is not None:
          # add one-hot encoded feature names for this column
          encoded_names = [n.split("__", 1)[-1] for n in ct.get_feature_names_out([col])]
          feature_names.extend(encoded_names)
      elif col in ordinal_cols or col in numeric_cols:
          feature_names.append(col)

  if len(categorical_cols) == 0:
    feature_names = Xtest.columns

  # fits model
  if ridge:
      lm = Ridge(alpha=alpha)
  else:
      lm = LinearRegression()

  lm.fit(Xtrain,y_train)
  y_pred = lm.predict(Xtest)

  # occupies model dictionary
  model_dict = linear_performance(y_pred, y_test)
  importance = permutation_importance(lm, Xtest, y_test, n_repeats = 30, random_state=42)

  model_dict['model'] = lm
  model_dict['predictions'] = y_pred
  model_dict['feature_performance'] = importance
  model_dict['feature_names'] = feature_names

  return model_dict

####################  driver code ####################
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

  # for report

  # print(f"overview latex_table:\n {overview['summary_table'].to_latex(index=False, float_format='%.2f'.__mod__)}")


  #### generates test/train for across models ####
  X,y = get_train_target(df, 'House_Price')
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


  # for creating table of model attributes
  models = []

  #### Fits the basic multiple linear regression model ####
  print("\n~~~~~~~~~~~~~~~~~~~\nBasic Model Fitting\n~~~~~~~~~~~~~~~~~~~")

  multi_model = make_linear_model(X_train, X_test, y_train, y_test) 

  # tracks model stats
  models.append({
     'model': "linear",
     'mse': multi_model['mse'],
     r'R^2': multi_model['R_squared'],
     'intercept': multi_model['model'].intercept_
  })

  

  print(f"R^2: {multi_model['R_squared']}")
  print(f"MSE: {multi_model['mse']}")
  print(f"\nModel Coefficients: \n{multi_model['model'].coef_}")
  print(f"\nIntercept: {multi_model['model'].intercept_}")

  # Creates scatter plot y_test v y_pred
  plt.scatter(x = y_test, y = multi_model['predictions'], alpha= .6)
  plt.plot(y_test, y_test, color='black', linestyle='--')
  plt.grid()
  plt.suptitle("Actual vs Predicted: Linear Regression", fontsize=14)
  plt.title("(with linear features)", fontsize = 12)
  
  plt.xlabel("Actual Home Price", fontsize = 11)
  plt.ylabel("Predicted Price", fontsize = 11)
  plt.savefig("basicLR_actualVpredicted.png", dpi=300)
  plt.show()
  plt.close()

  # Creates scatter plot of y_test v Residuals
  plt.scatter(x = y_test, y = multi_model['residuals'], alpha= .6)
  plt.axhline(y=np.mean(multi_model['residuals']), color='black', linestyle='--', linewidth=1) 
  plt.suptitle("Multiple Linear Regression Residuals", fontsize = 14)
  plt.title("(with linear features)", fontsize=12)
  plt.xlabel("True Prices")
  plt.ylabel("Price Estimation Errors")
  plt.grid()
  plt.savefig("basicLR_residualsPlot.png")
  plt.show()
  plt.close()

  # Creates scatter plot for individual features
  col_names = X_train.columns

  for i in range(len(col_names)):
    name = col_names[i]
  
    if X_train[name].dtype =="object" or X_train[name].dtype =="category":
      continue

    simple_model = make_linear_model(pd.DataFrame(X_train[name]), pd.DataFrame(X_test[name]),y_train,y_test)

    
    coef = simple_model['model'].coef_
    intercept = simple_model['model'].intercept_

    x_values = X_train[name]
    y_values = (coef * x_values.astype(float)) + intercept
    plt.scatter(x_values, y_train, alpha= 0.6, label = "Train Data")
    plt.plot(x_values,y_values, color = "red", label = "Fitted Line")
    plt.suptitle(f"Fitted Line against {name}/House_Price Bivariate Points", fontsize = 14)
    plt.title("(with linear features)", fontsize=12)
    plt.xlabel(name)
    plt.ylabel('House_Price')
    plt.legend()
    plt.savefig(f"{name}_simpleLinearRegression_basic.png")
    plt.show()

      
      
  plt.close()

  # Creates permuted feature performance 
  plt.barh(multi_model['feature_names'],
          multi_model['feature_performance'].importances_mean,
          xerr = multi_model['feature_performance'].importances_std,
          alpha=0.7, 
          color='skyblue', 
          ecolor='black', 
          capsize=4
  )

  plt.title("Linear Regression Model Feature Importance", fontsize = 14)
  plt.xlabel(r"Feature Importance: Mean Decrease in $R^2$", fontsize = 12)
  plt.ylabel("Independent Features", fontsize = 12)
  plt.tight_layout()
  plt.savefig("basicLR_featureImportance.png")
  plt.show()





  #### Fits polynomial featured linear model ####
  print("\n~~~~~~~~~~~~~~~~~~~~~~~~\nPolynomial Model Fitting\n~~~~~~~~~~~~~~~~~~~~~~~~")
  print("deg = 5")

  poly_model = make_linear_model(X_train, X_test, y_train, y_test, degree = 5)  

  #tracks model stats
  models.append({
     'model': 'poly_deg5',
     'mse': poly_model['mse'],
     r'R^2': poly_model['R_squared'],
     'intercept': poly_model['model'].intercept_
  })

  print(f"R^2: {poly_model['R_squared']}")
  print(f"MSE: {poly_model['mse']}")
  print(f"\nModel Coefficients: \n{poly_model['model'].coef_}")
  print(f"\nIntercept: {poly_model['model'].intercept_}")

  # Creates scatter plot y_test v y_pred
  plt.scatter(x = y_test, y = poly_model['predictions'], alpha= .6)
  plt.plot(y_test, y_test, color='black', linestyle='--')
  plt.grid()
  plt.suptitle("Actual vs Predicted: Linear Regression", fontsize = 14)
  plt.title("(Degree 5 polynomial features)", fontsize=12)
  plt.xlabel("Actual Home Price", fontsize = 11)
  plt.ylabel("Predicted Price", fontsize = 11)
  plt.savefig("deg5polyLR_actualVpredicted.png", dpi=300)
  plt.show()
  plt.close()

  # Creates scatter plot of y_test v Residuals
  plt.scatter(x = y_test, y = poly_model['residuals'], alpha= .6)
  plt.axhline(y=np.mean(poly_model['residuals']), color='black', linestyle='--', linewidth=1) 
  plt.suptitle("Multiple Linear Regression Residuals", fontsize = 14)
  plt.suptitle("(Degree 5 polynomial features)", fontsize=12)
  plt.xlabel("True Prices")
  plt.ylabel("Price Estimation Errors")
  plt.grid()
  plt.tight_layout()
  plt.savefig("deg5polyLR_residualsPlot.png")
  plt.show()
  plt.close()

  for name in col_names:
    if X_train[name].dtype in ["object", "category"]:
        continue

    # Fit simple model with same degree as full model
    simple_model = make_linear_model(
        pd.DataFrame(X_train[name]), 
        pd.DataFrame(X_test[name]),
        y_train,
        y_test,
        degree=5
    )

    coef = simple_model['model'].coef_
    intercept = simple_model['model'].intercept_

    # Original column, sorted for smooth curve
    x_values = X_train[name]
    x_sorted = np.sort(x_values.values).reshape(-1, 1)

    # Create polynomial features for sorted values
    poly = PolynomialFeatures(degree=5, include_bias=False)
    x_poly = poly.fit_transform(x_sorted)

    # Compute y-values
    y_values = x_poly @ coef + intercept

    plt.scatter(x_values, y_train, alpha=0.6, label="Train Data")
    plt.plot(x_sorted, y_values, color="red", label="Fitted Polynomial Curve")
    plt.suptitle(f"Fitted Curve against {name}/House_Price", fontsize = 14)
    plt.title(f"(polynomial degree = 5)", fontsize=12)
    plt.xlabel(name)
    plt.ylabel("House_Price")
    plt.legend()
    plt.savefig(f"{name}_simpleLinearRegression_5poly.png")
    plt.show()

  #### Fits polynomial featured linear model ####
  print("\n~~~~~~~~~~~~~~~~~~~~~~~~\nPolynomial Model Fitting\n~~~~~~~~~~~~~~~~~~~~~~~~")
  print("deg = 2")

  poly_model = make_linear_model(X_train, X_test, y_train, y_test, degree = 2)  

  #tracks model stats
  models.append({
    'model': 'poly_deg2',
    'mse': poly_model['mse'],
    r'R^2': poly_model['R_squared'],
    'intercept': poly_model['model'].intercept_
  })

  print(f"R^2: {poly_model['R_squared']}")
  print(f"MSE: {poly_model['mse']}")
  print(f"\nModel Coefficients: \n{poly_model['model'].coef_}")
  print(f"\nIntercept: {poly_model['model'].intercept_}")

  # Creates scatter plot y_test v y_pred
  plt.scatter(x = y_test, y = poly_model['predictions'], alpha= .6)
  plt.plot(y_test, y_test, color='black', linestyle='--')
  plt.grid()
  plt.suptitle("Actual vs Predicted: Linear Regression", fontsize = 14)
  plt.title("(Degree 2 polynomial features)", fontsize=12)
  plt.xlabel("Actual Home Price", fontsize = 11)
  plt.ylabel("Predicted Price", fontsize = 11)
  plt.savefig("deg2polyLR_actualVpredicted.png", dpi=300)
  plt.show()
  plt.close()

  # Creates scatter plot of y_test v Residuals
  plt.scatter(x = y_test, y = poly_model['residuals'], alpha= .6)
  plt.axhline(y=np.mean(poly_model['residuals']), color='black', linestyle='--', linewidth=1) 
  plt.suptitle("Multiple Linear Regression Residuals", fontsize = 14)
  plt.suptitle("(Degree 2 polynomial features)", fontsize=12)
  plt.xlabel("True Prices")
  plt.ylabel("Price Estimation Errors")
  plt.grid()
  plt.tight_layout()
  plt.savefig("deg2polyLR_residualsPlot.png")
  plt.show()
  plt.close()

  for name in col_names:
    if X_train[name].dtype in ["object", "category"]:
        continue

    # Fit simple model with same degree as full model
    simple_model = make_linear_model(
        pd.DataFrame(X_train[name]), 
        pd.DataFrame(X_test[name]),
        y_train,
        y_test,
        degree=5
    )

    coef = simple_model['model'].coef_
    intercept = simple_model['model'].intercept_

    # Original column, sorted for smooth curve
    x_values = X_train[name]
    x_sorted = np.sort(x_values.values).reshape(-1, 1)

    # Create polynomial features for sorted values
    poly = PolynomialFeatures(degree=5, include_bias=False)
    x_poly = poly.fit_transform(x_sorted)

    # Compute y-values
    y_values = x_poly @ coef + intercept

    plt.scatter(x_values, y_train, alpha=0.6, label="Train Data")
    plt.plot(x_sorted, y_values, color="red", label="Fitted Polynomial Curve")
    plt.suptitle(f"Fitted Curve against {name}/House_Price", fontsize = 14)
    plt.title(f"(polynomial degree = 2)", fontsize=12)
    plt.xlabel(name)
    plt.ylabel("House_Price")
    plt.legend()
    plt.savefig(f"{name}_simpleLinearRegression_2poly.png")
    plt.show()

  #### Fits polynomial featured ridge regression model ####
  print("\n~~~~~~~~~~~~~~~~~~~~~~~~\nPolynomial Ridge Model Fitting\n~~~~~~~~~~~~~~~~~~~~~~~~")
  print("deg = 5")

  ridge_poly_model = make_linear_model(
      X_train, X_test, y_train, y_test, degree = 5, ridge=True, alpha=1)

  #tracks model stats
  models.append({
    'model': 'ridgePoly_deg5',
    'mse': ridge_poly_model['mse'],
    r'R^2': ridge_poly_model['R_squared'],
    'intercept': ridge_poly_model['model'].intercept_
  })

  print(f"R^2: {ridge_poly_model['R_squared']}")
  print(f"MSE: {ridge_poly_model['mse']}")
  print(f"\nModel Coefficients: \n{ridge_poly_model['model'].coef_}")
  print(f"\nIntercept: {ridge_poly_model['model'].intercept_}")

  # Scatter plot y_test vs y_pred
  plt.scatter(x=y_test, y=ridge_poly_model['predictions'], alpha=0.6)
  plt.plot(y_test, y_test, color='black', linestyle='--')
  plt.suptitle("Actual vs Predicted: Ridge Regression", fontsize=14)
  plt.title("(degree 5 polynomial features)", fontsize=12)
  plt.xlabel("Actual Home Price", fontsize=11)
  plt.ylabel("Predicted Price", fontsize=11)
  plt.grid()
  plt.savefig("deg5polyRidge_actualVpredicted.png", dpi=300)
  plt.show()
  plt.close()

  # Scatter plot y_test vs residuals
  plt.scatter(x=y_test, y=ridge_poly_model['residuals'], alpha=0.6)
  plt.axhline(y=np.mean(ridge_poly_model['residuals']), color='black', linestyle='--', linewidth=1)
  plt.suptitle("Ridge Regression Residuals", fontsize=14)
  plt.title("(degree 5 polynomial features)", fontsize=12)
  plt.xlabel("True Prices")
  plt.ylabel("Price Estimation Errors")
  plt.grid()
  plt.savefig("deg5polyRidge_residualsPlot.png")
  plt.show()
  plt.close()

  # Individual feature curves
  for name in col_names:
      if X_train[name].dtype in ["object", "category"]:
          continue

      simple_model = make_linear_model(
          pd.DataFrame(X_train[name]), 
          pd.DataFrame(X_test[name]),
          y_train,
          y_test,
          degree=5,
          ridge=True,
          alpha=1.0
      )

      coef = simple_model['model'].coef_
      intercept = simple_model['model'].intercept_

      x_values = X_train[name]
      x_sorted = np.sort(x_values.values).reshape(-1, 1)

      poly = PolynomialFeatures(degree=5, include_bias=False)
      x_poly = poly.fit_transform(x_sorted)

      y_values = x_poly @ coef + intercept

      plt.scatter(x_values, y_train, alpha=0.6, label="Train Data")
      plt.plot(x_sorted, y_values, color="red", label="Fitted Polynomial Curve")
      plt.suptitle(f"Fitted Curve against {name}/House_Price", fontsize=14)
      plt.title(f"(polynomial degree = 5, ridge regression)", fontsize=12)
      plt.xlabel(name)
      plt.ylabel("House_Price")
      plt.legend()
      plt.savefig(f"{name}_simpleRidgeRegression_5rigdepoly.png")
      plt.show()
      plt.close()

  print("\n~~~~~~~~~~~~~~~~~~~~~~~~\nPolynomial Ridge Model Fitting\n~~~~~~~~~~~~~~~~~~~~~~~~")
  print("deg = 2")

  ridge_poly_model = make_linear_model(
      X_train, X_test, y_train, y_test, degree = 2, ridge=True, alpha=1)
  
  #tracks model stats
  models.append({
    'model': 'ridgePoly_deg2',
    'mse': ridge_poly_model['mse'],
    r'R^2': ridge_poly_model['R_squared'],
    'intercept': ridge_poly_model['model'].intercept_
  })


  print(f"R^2: {ridge_poly_model['R_squared']}")
  print(f"MSE: {ridge_poly_model['mse']}")
  print(f"\nModel Coefficients: \n{ridge_poly_model['model'].coef_}")
  print(f"\nIntercept: {ridge_poly_model['model'].intercept_}")

  # Scatter plot y_test vs y_pred
  plt.scatter(x=y_test, y=ridge_poly_model['predictions'], alpha=0.6)
  plt.plot(y_test, y_test, color='black', linestyle='--')
  plt.suptitle("Actual vs Predicted: Ridge Regression", fontsize=14)
  plt.title("(degree 2 polynomial features)", fontsize=12)
  plt.xlabel("Actual Home Price", fontsize=11)
  plt.ylabel("Predicted Price", fontsize=11)
  plt.grid()
  plt.savefig("deg2polyRidge_actualVpredicted.png", dpi=300)
  plt.show()
  plt.close()

  # Scatter plot y_test vs residuals
  plt.scatter(x=y_test, y=ridge_poly_model['residuals'], alpha=0.6)
  plt.axhline(y=np.mean(ridge_poly_model['residuals']), color='black', linestyle='--', linewidth=1)
  plt.suptitle("Ridge Regression Residuals", fontsize=14)
  plt.title("(degree 2 polynomial features)", fontsize=12)
  plt.xlabel("True Prices")
  plt.ylabel("Price Estimation Errors")
  plt.grid()
  plt.tight_layout()
  plt.savefig("deg2polyRidge_residualsPlot.png")
  plt.show()
  plt.close()

  # Individual feature curves
  for name in col_names:
      if X_train[name].dtype in ["object", "category"]:
          continue

      simple_model = make_linear_model(
          pd.DataFrame(X_train[name]), 
          pd.DataFrame(X_test[name]),
          y_train,
          y_test,
          degree=2,
          ridge=True,
          alpha=1.0
      )

      coef = simple_model['model'].coef_
      intercept = simple_model['model'].intercept_

      x_values = X_train[name]
      x_sorted = np.sort(x_values.values).reshape(-1, 1)

      poly = PolynomialFeatures(degree=2, include_bias=False)
      x_poly = poly.fit_transform(x_sorted)

      y_values = x_poly @ coef + intercept

      plt.scatter(x_values, y_train, alpha=0.6, label="Train Data")
      plt.plot(x_sorted, y_values, color="red", label="Fitted Polynomial Curve")
      plt.suptitle(f"Fitted Curve against {name}/House_Price", fontsize=14)
      plt.title(f"(polynomial degree = 2, ridge regression)", fontsize=12)
      plt.xlabel(name)
      plt.ylabel("House_Price")
      plt.legend()
      plt.savefig(f"{name}_simpleRidgeRegression_2rigdepoly.png")
      plt.show()
      plt.close()


models_table = pd.DataFrame(models)

print(f"\n{models_table}")
print(f"\n{models_table.to_latex(index=False)}")