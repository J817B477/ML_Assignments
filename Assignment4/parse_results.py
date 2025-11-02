'''
John Bennett
Assignment 4: Support Vector Machines
Fall 2025
Script: parse_results.py

Approach: This script is used handle the results of the training across combinations of parameters stored in the "results.pkl" file. This file reflects all of the program calls to "TrainSCV.py" in the "generate_models.sh" bash shell script. The data produced for each model is processed here to look for the best model and to analyse trends in the results, which are accompanied by visuals. 
'''
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings
from sklearn.decomposition import PCA
import numpy as np
from sklearn.svm import SVC
warnings.filterwarnings("ignore") # nuisance 

# retrieves results: separates metadata & model data
def get_results(results_file: str):
  '''
  function gets the results of trainings for all of the models stored in the pickle file by the TrainSVC.py script

  parameter: string to identify project level path of pickle file
  
  return: two dictionaries
  - metadata: information about the source training data
  - model_dict: all of the results data collected for each trained model
  '''
  # loads results dict
  with open(results_file, "rb") as f:
    results = pickle.load(f)

  # isolates metadata
  metadata = results['meta']

  # isolates model training results
  model_keys = list(results.keys())[1:]
  models_dict = {key: results[key] for key in model_keys}

  return metadata, models_dict

## Summary Table of Models
def create_performance_table(models_results: dict):
  '''
  formats the results values from the model trainings into a table that performance trends can be extracted from

  parameter: model training results dict
  return: pandas df
  '''
  # gets the keys for each models results dictionary
  model_keys = list(next(iter(models_results.values())).keys())

  # transforms dict to dataframe
  table = pd.DataFrame(models_results).T

  # drops values that are not displayed in the table
  remove_features = model_keys[6:]
  table.drop(remove_features,axis=1,inplace=True)
  #renames columns of table
  table.columns = ['Kernel Type', 'C','Gamma', 'Test Accuracy', '# Support Vectors', 'Training Time']

  # organizes model records by C_scalar and Gamma
  table = table.sort_values(['C','Gamma'])
  return table
  
def create_comparative_accuracy(models_results: dict)-> dict:
  '''
  function generates tables intended specifically for creating kernel specific tables to facilitate visualizations specific to the hyperparameters that accompany each kernel type

  parameter: model training results dict
  return: dictionary containing individualized tables per kernel
  '''
  # generates table from models_results dictionary
  df = pd.DataFrame(models_results).T[["kernel_type","c_scalar", "gamma","accuracy"]]
  df.columns = ['Kernel', 'C', 'Gamma', 'Accuracy']
  kernels = df['Kernel'].unique()

  # generates dfs for vis sources based on kernel type
  comparative_dict = {}
  for k in kernels:
    if k == 'linear':
      comparative_dict[k] = df[df['Kernel'] == k]
    else:
      df_temp = df[df['Kernel'] == k]
      df_temp['Accuracy'] = df_temp['Accuracy'].astype(float)
      comparative_dict[k] = df_temp.pivot(
        index = 'C',
        columns= 'Gamma',
        values= 'Accuracy'
      )

  return comparative_dict

# utility to allowing output organization in project folder
def new_project_folder(folder_name: str)-> None:
  '''
  Utility method for creating a new directory in the project folder so that outputs can be organized

  parameter: name of folder to be added
  return: Nothing
  '''
  cwd = os.getcwd()
  new_dir = os.path.join(cwd,folder_name)
  if os.path.exists(new_dir):
    print(f"{new_dir} already exists")
  else:
    os.mkdir(new_dir)
  return

# ################## Driver Code ################## 
if __name__ == "__main__":

  # imports the results of training from other scripts
  metadata, models = get_results("results.pkl")

  # metadata for source data
  print("\n~~~~~~~~~~~~~~~~~~~~~~~~\nSource Data Information\n~~~~~~~~~~~~~~~~~~~~~~~~")

  print(f"""  Number of records: {metadata['nrows']}
  Number of features: {metadata['ncolumns']}

  Data Frame Sample Records:
  {metadata['head']}

  Feature Overview:
  {metadata['summary_table']}
  """)


  print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\nModel Performance Information\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")  

  # stores csvs
  new_project_folder("performance_tables_plots")

  #### performance tables for reporting ####

  # Full table
  performance_table = create_performance_table(models)
  performance_table.to_csv("performance_tables_plots/performance_table.csv")

  print(f"""  Performance Table:
  {performance_table}
  """)

  # most accurate models table
  top_accuracy = performance_table.sort_values("Test Accuracy", ascending=False)
  top_accuracy = top_accuracy[top_accuracy["Test Accuracy"] >= .80]
  top_accuracy.to_csv("performance_tables_plots/performance_table_topAccuracy.csv")
  print(f""" Top Accuracy:
  {top_accuracy}
  """)


  # best training time table
  top_time = performance_table.sort_values("Training Time", ascending=True).iloc[:15,:]
  top_time.to_csv("performance_tables_plots/performance_table_topTime.csv")
  print(f""" Top Training Time:
  {top_time}
  """) 


  # least complex table (fewest support vectors)
  top_SVs = performance_table.copy()
  top_SVs['first_sv'] = top_SVs["# Support Vectors"].apply(lambda x: x[0])
  top_SVs = top_SVs.sort_values("first_sv", ascending = True).iloc[:15,:-1]
  


  top_SVs.to_csv("performance_tables_plots/performance_table_topSVs.csv")

  print(f""" Top Complexity:
  {top_SVs}
  """) 

  # # for report
  # print(top_SVs.to_latex())


  #### figures for reporting ####

  # uniform color scaling values for heatmaps
  color_min = min(performance_table['Test Accuracy'])
  color_max = max(performance_table['Test Accuracy'])

  comp_accuracy = create_comparative_accuracy(models)

  new_project_folder("model_accuracy_comparison")

  # creates figures for accuracy comparisons
  keys = comp_accuracy.keys()
  
  for k in keys:
    df = comp_accuracy[k]
    if k == 'linear':
      sns.barplot(df, x= df['C'], y= df['Accuracy'], color= '#5CA462')
      plt.title(f"{k} kernel accuracy distribution".title(), fontsize = 14)
      plt.xlabel('C scalar value')
      plt.ylabel('Accuracy')
      plt.savefig(f"model_accuracy_comparison/{k}_accuracy_bplot.pdf")
      plt.show()
      print(f"model_accuracy_comparison/{k}_accuracy_bplot.pdf created")
    else:
      sns.heatmap(df, annot= True, 
                  cmap="Greens", 
                  vmin=color_min, 
                  vmax=color_max, 
                  cbar_kws = {'label': "Accuracy"})
      
      plt.title(f"{k} kernel accuracy distribution".title(), fontsize = 14)
      plt.xlabel('Gamma Value')
      plt.ylabel('C Scalar Value')
      plt.savefig(f"model_accuracy_comparison/{k}_accuracy_hmap.pdf")
      plt.show()
      print(f"model_accuracy_comparison/{k}_accuracy_bplot.pdf created")

  #### best model decision boundary with support vectors
  print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\nBest Model Decision Boundary Plot \n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

  best_model_key = max(models, key=lambda k: models[k]["accuracy"])
  best_model = models[best_model_key]

  # extracts model training data
  X = best_model['feature_vectors']
  y = best_model['target_vector']

  # scales feature vectors
  scaler = best_model['model'].named_steps['scaler']
  X_scaled = scaler.transform(X)

  # projects feature vectors onto 2d plain
  pca = PCA(n_components=2)
  X_pca = pca.fit_transform(X_scaled)

  # 2d grid space for 
  xx, yy = np.meshgrid(
      np.linspace(X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1, 300),
      np.linspace(X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1, 300)
  )
  grid_pca = np.c_[xx.ravel(), yy.ravel()]

  # Map grid back to approximate 10D space
  grid_10D_approx = grid_pca @ pca.components_[:2, :] + X_scaled.mean(axis=0)

  # Predict decision values from original high-D model
  Z = best_model['model'].named_steps['svm'].decision_function(grid_10D_approx)
  Z = Z.reshape(xx.shape)

  # plots projection of decision boundary onto  2d space
  plt.figure(figsize=(8, 6))
  plt.contourf(xx, yy, Z, alpha=0.35, cmap=plt.cm.coolwarm)
  plt.contour(xx, yy, Z, levels=[0], colors='k', linewidths=1.25)
  plt.suptitle("Projected Multidimensional Decision Boundary", fontsize=16)
  plt.title(f"True 10D model projected into PCA space\n"
            f"Kernel={best_model['kernel_type']}, C={best_model['c_scalar']}, Gamma={best_model['gamma']}",
            fontsize=11)
  plt.xlabel('PC1')
  plt.ylabel('PC2')
  plt.tight_layout()
  plt.savefig("performance_tables_plots/best_model_boundary_projected.pdf")
  plt.close()



  # plots representative decision boundary with data
  plt.figure(figsize=(8, 6))
  svm_2d = SVC(kernel=best_model['kernel_type'],
                C=best_model['c_scalar'],
                gamma=best_model['gamma'])
  svm_2d.fit(X_pca, y)

  xx2, yy2 = np.meshgrid(
      np.linspace(X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1, 300),
      np.linspace(X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1, 300)
  )
  grid2 = np.c_[xx2.ravel(), yy2.ravel()]
  Z2 = svm_2d.predict(grid2).reshape(xx2.shape)

  plt.contourf(xx2, yy2, Z2, alpha=0.25, cmap=plt.cm.coolwarm)
  plt.contour(xx2, yy2, Z2, levels=[0], colors='k', linewidths=1.25)
  plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=plt.cm.coolwarm,
              edgecolor='k', s=10, label='Training Points')

  svs_2d = svm_2d.support_vectors_
  plt.scatter(svs_2d[:, 0], svs_2d[:, 1],
              facecolors='none', edgecolors='k', s=15,
              linewidths=1.5, label='Support Vectors')

  plt.suptitle("2D Visualization SVM Decision Boundary", fontsize=16)
  plt.title(f"SVM trained on PCA-reduced space\n"
            f"Kernel={best_model['kernel_type']}, C={best_model['c_scalar']}, Gamma={best_model['gamma']}",
            fontsize=11)
  plt.xlabel('PC1')
  plt.ylabel('PC2')
  plt.legend()
  plt.tight_layout()
  plt.savefig("performance_tables_plots/best_model_boundary_PCA2D.pdf")
  plt.close()

  print(f""" Decision boundary plots created:
        performance_tables_plots/best_model_boundary_projected.pdf
        performance_tables_plots/best_model_boundary_PCA2D.pdf
        """
  )


