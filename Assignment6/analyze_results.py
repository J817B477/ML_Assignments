import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
import seaborn as sns 
import ast

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

if __name__ == "__main__":
  source_df = pd.read_csv("WineQT.csv")

  overview = dataframe_overview(source_df)

  print(overview["summary_table"])


  df = pd.read_csv("test_train_results.csv")
  
  df = df.dropna()

  print(df.dtypes)

  df['probabilities'] = df['probabilities'].apply(ast.literal_eval)
  df['true_labels'] = df['true_labels'].apply(ast.literal_eval)
  df['confusion_matrix'] = df['confusion_matrix'].apply(ast.literal_eval)
  print(df.shape)

  print(df.dtypes)

  print(df['loss'].min())

  best_accuracy_df = df.sort_values(by="test_accuracy", ascending=False)
  
  best_model = best_accuracy_df.iloc[0]

  print(best_model["true_labels"])

  y_true = np.array(best_model["true_labels"])

  prob_list = best_model["probabilities"]      # list of rows
  prob_list = [row[0] if isinstance(row, list) and len(row) == 1 else row
              for row in prob_list]

  y_prob = np.vstack(prob_list)
  print(y_prob.shape)



  # Binarize labels for multiclass ROC
  num_classes = y_prob.shape[1]
  y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

  # Compute ROC curve and AUC for each class
  fpr = dict()
  tpr = dict()
  roc_auc = dict()
  for i in range(num_classes):
      fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
      roc_auc[i] = auc(fpr[i], tpr[i])

  # Plot ROC curves
  original_labels = source_df["quality"].unique()
  plt.figure()
  for i in range(num_classes):
      plt.plot(fpr[i], tpr[i], lw=2, 
               label=f'{original_labels[i]} (AUC = {roc_auc[i]:.2f})')

  plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Multiclass ROC')
  plt.legend(loc="lower right")
  plt.savefig("plots/AUC.pdf")
  plt.show()
  plt.close()

# confusion matrix
# order in your existing CM
# desired_labels = ['3','4','5','6','7','8']   # order you want
current_labels = [0,1,2,3,4,5]
# Find the indices that map desired_labels to current_labels
indices = [current_labels.index(label) for label in original_labels]

# Reorder rows and columns

cm = best_model["confusion_matrix"]
cm_reordered = cm[np.ix_(indices, indices)]
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=[3,4,5,6,7,8],
            yticklabels=[3,4,5,6,7,8],
          cbar=True)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix Heatmap")
plt.savefig("plots/cm.pdf")
plt.show()

