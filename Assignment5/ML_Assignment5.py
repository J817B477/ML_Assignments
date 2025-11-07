import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix, adjusted_rand_score


# function: makes new project folder
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

# function: fits a kmeans model
def a_kmeans(df:pd.DataFrame, k:int, max_training_iter = 300, cent_disp_threshold = np.e**(-4), n_init = 10, seed = None)->pd.DataFrame:
  '''
  function is an implementation of kmeans clustering algorithm

  parameters:
  - df: data to cluster (as a pandas df)
  - k: number of clusters to form
  - max_training_iter: maximum iterations of adjusting centroid positions and data object assignments
  - cent_disp_threshold: min total centroid displacement needed to stop training
  - n_init: the number of times a model of k clusters is trained for comparison to find minimum distortion
  - seed: values used for seeding randomization algorithm used

  return:
  - dictionary (includes):
    - labeled_df: passed df with labels column
    - cluster_SSE: dict of distortion for each cluster (key = label)
    - total_SSE: total SSE for the model
    - centroids: the central location of each cluster (key = label)
  '''
  # whether or not output is reproducible
  if seed == None:
    rng = np.random.default_rng()
  else:
    rng = np.random.default_rng(seed)

  # selects seed values: one for each model candidate
  # if seed parameter = none, then list is different each time function is called
  training_seeds = rng.integers(low = 0, high = 2**32, size=n_init)


  ## constant structures
  # converts to matrix for navigation with numpy
  labeled_df = df.copy()
  matrix = df.to_numpy()
  # stores n models that model with minimum totalSSE is selected from  
  model_candidates = []

  for n in range(0,n_init):

    ## handles the variation of candidate models
    model_rng = np.random.default_rng(training_seeds[n])

    ## these updated structures survive inner scope:
   
    # initial centroids
    centroids_idx = model_rng.choice(matrix.shape[0],k,replace=False)
    current_centroids= matrix[centroids_idx,:]

    # empty structures
    labels = np.empty(shape = (matrix.shape[0],), dtype=int)
    final_distances = np.empty(shape = (matrix.shape[0],), dtype=float) # reduces calculations
    clusterSSE = {} # returns individual clusters distortions
    TotalSSE = 0 # for elbow plot

   
    # iterations of training
    iteration = 0

    ####  from here down to "wrap up" is a single training iteration
    while iteration < max_training_iter:
      next_centroids = np.empty_like(current_centroids)

      for i, obj in enumerate(matrix):
        # creates array: stores iteration's distances from each centroid

        obj_distances = np.sum((current_centroids - obj)**2, axis = 1)**.5
        
        # updates training iteration's labels (index = label)
        dist_idx = np.argmin(obj_distances)
        labels[i] = dist_idx

        # updates final distances (rewrites final distances for each training iterations until convergence)
        final_distances[i] = obj_distances[dist_idx]

      # computes new centroid set iteratively for each cluster 
      cluster_ids = list(range(0,k))
      

      for id in cluster_ids:
        cluster_mask = labels == id
        cluster = matrix[cluster_mask,:]
        # array with mean of each column = center of current cluster
        next_centroids[id] = np.mean(cluster, axis=0) 

      # check the displacement of centroids
      total_centroid_movement = 0
      for i,j in zip(current_centroids, next_centroids):
        total_centroid_movement += np.sum((i-j)**2)**0.5
      
      if total_centroid_movement < cent_disp_threshold:
        break
      
      iteration += 1
      # if the loop gets here, updates the init_centroid of next iteration (minus edge case band aid)
      if iteration < max_training_iter:
        current_centroids = next_centroids.copy()

    #### results wrap up 

    # get distortion data
    cluster_ids = np.sort(np.unique(labels))
    for id in cluster_ids:
        cluster_mask = labels == id
        cluster_distortion = np.sum((final_distances**2)[cluster_mask])
        clusterSSE[id] = cluster_distortion
        TotalSSE += cluster_distortion

    # appends model object
    model_candidates.append(
            {'labels': labels, 
            'model_SSE': TotalSSE.item(), 
            'cluster_SSE': clusterSSE, 
            'centroids': current_centroids})
    
  # stores model candidate with smallest sse
  best_model = min(model_candidates, key = lambda m: m['model_SSE'])

  #replaces best model labels with updated df
  labeled_df['cluster_labels'] = best_model['labels']
  best_model['labeled_df'] = labeled_df
  del best_model['labels']

  # converts centroids matrix into a dictionary
  best_model['centroids'] = {i: row for i, row in enumerate(best_model['centroids'])}

  # cleans up object value types
  best_model['centroids']= {k: cent.tolist() for k, cent in best_model['centroids'].items()}

  best_model['cluster_SSE'] = {k.item(): v.item() for k,v in best_model['cluster_SSE'].items()}

  return best_model

########### driver code ###########
if __name__ == "__main__":
  # plots folder:
  new_project_folder("plots")

  # reads in df 
  df = pd.read_csv("data_2D.txt", sep='\s+', header=None, names=['x_1','x_2'])

  print("\n~~~~~~~~~~~~~~~~~~~~~~~~~\n Shows proper data import:\n~~~~~~~~~~~~~~~~~~~~~~~~~")

  print(f"""
  Dataframe rows and columns:{df.shape}

  Dataframe Preview:
  {df.head()}""")

  # tests my implementation of kmeans
  kmeans_model = a_kmeans(df,8)


  #### Non-optimized example of my implementation
  print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\nExample of model from my implementation:\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

  print(f"""
  dataframe with labels added:
    {kmeans_model['labeled_df']}

  model_SSE:
    {kmeans_model['model_SSE']}

  cluster_SSE:
    {kmeans_model['cluster_SSE']}

  cluster centroids:
    {kmeans_model['centroids']}
  """)

  trained_df = kmeans_model['labeled_df']
  centroids = pd.DataFrame(list(kmeans_model['centroids'].values()))

  plt.scatter(df['x_1'],
              df['x_2'],
              c = trained_df['cluster_labels'],
              cmap="coolwarm",
              s = 10)
  
  plt.scatter(centroids[0], centroids[1],facecolors = "none", edgecolors= "black",marker='o', s = 50)
  plt.suptitle("Eight Cluster Model with Centroids", fontsize = 16)
  plt.title("(my implementation)")
  plt.savefig("plots/example.pdf")
  plt.show()
  plt.close()

  print(f"plots/example.pdf has been created")
  # finds optimal k for model

  print(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\nBest K-value: Personal vs skLearn K-means implementation\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

  # personal implementation models
  models = {}

  for k in range(1,11):
    models[k] = a_kmeans(df, 
                         k, 
                         max_training_iter= 300, 
                         seed = 42)
  
  k_candidates = list(models.keys())

  candidate_SSEs = [k['model_SSE'] for k in models.values()]

  plt.plot(k_candidates,candidate_SSEs, marker = '.')
  plt.title("Personal K-means Implementation Elbow Plot")
  plt.xlabel("Number of Clusters")
  plt.xticks(list(range(0,11)))
  plt.ylabel("Model Distortion (SSE)")
  plt.tight_layout()
  plt.savefig("plots/elbow_MyKmeans.pdf")
  plt.show()
  plt.close()
  
  print(f"plots/elbow_MyKmeans.pdf has been created")


  # creates scatter plot of my best model 
  my_best_model = models[5]
  my_labeled_df = my_best_model['labeled_df']
  my_centroids = pd.DataFrame(list(my_best_model['centroids'].values()))
  my_k = len(my_labeled_df['cluster_labels'].unique())
  plt.scatter(df['x_1'],
              df['x_2'],
              c = my_labeled_df['cluster_labels'],
              cmap="coolwarm",
              s = 10)
  
  plt.scatter(my_centroids[0], my_centroids[1],facecolors = "none", edgecolors= "black",marker='o', s = 50)
  plt.suptitle(f"Kmeans Model with {my_k} Clusters", fontsize = 16)
  plt.title("(my implementation)")
  plt.savefig("plots/myBestScatter.pdf")
  plt.show()
  plt.close()

  print(f"plots/myBestScatter.pdf has been created")
  # sklearn implementation models
  skModels = {}
  df_X = df.to_numpy()

  for k in range(1,11):
    skmodel = KMeans(k, n_init=10, random_state=42)

    skModels[k] = skmodel.fit(df_X)

  skSSEs = [m.inertia_ for m in skModels.values()]

  # creates elbow plot
  plt.plot(list(skModels.keys()), skSSEs, marker = ".")
  plt.title("Sci-kit Learn K-means Implementation Elbow Plot")
  plt.xlabel("Number of Clusters")
  plt.xticks(list(range(0,11)))
  plt.ylabel("Model Distortion (SSE)")
  plt.tight_layout()
  plt.savefig("plots/elbow_skKmeans.pdf")
  plt.show()
  plt.close()

  print(f"plots/elbow_skKmeans.pdf has been created")

  # creates best model scatter plot
  sk_best_model = skModels[5]
  sk_labeled_df = df.copy()
  sk_labeled_df['cluster_labels'] = sk_best_model.labels_
  sk_centroids = sk_best_model.cluster_centers_
  sk_k = len(np.unique(sk_best_model.labels_))
  plt.scatter(df['x_1'], df['x_2'], c=sk_labeled_df['cluster_labels'], cmap='coolwarm', s=10)
  plt.scatter(sk_centroids[:,0], sk_centroids[:,1], facecolors='none', edgecolors='black', s=50, marker='o')
  plt.suptitle(f"KMeans Clustering with {sk_k} Clusters",fontsize = 16)
  plt.title("(sci-kit learn implementation)")
  plt.savefig("plots/skBestScatter.pdf")
  plt.show()
  plt.close()

  print(f"plots/skBestScatter.pdf has been created")

  print("\n~~~~~~~~~~~~~~\nSSE Distortion\n~~~~~~~~~~~~~~")

  print(f"""
    My Model SSE: {candidate_SSEs[4]}
    SKlearn SSE: {skSSEs[4]}
  """)

  ##### assignment cluster comparison #####


  print(f"\n~~~~~~~~~~~~~~~~~~\nCluster Comparison\n~~~~~~~~~~~~~~~~~~")

  # aligns sklearn's labels to my labels based on membership agreement
  myLabels = my_labeled_df['cluster_labels']
  skLabels = sk_labeled_df['cluster_labels']

  cm = confusion_matrix(myLabels, skLabels)

  # Hungarian algorithm implementation: finds label alignment 
  row_ind, col_ind = linear_sum_assignment(-cm)
  mapping = dict(zip(col_ind, row_ind))

  # remap sklearn labels
  sk_labeled_df['aligned_labels'] = np.array([mapping[l] for l in skLabels])

  print(f"""
    Comparative Clusters ARI score = {adjusted_rand_score(myLabels,skLabels)}
  """)



    
