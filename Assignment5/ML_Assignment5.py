import pandas as pd
import numpy as np

def a_kmeans(df:pd.DataFrame, k:int, seed = None)->pd.DataFrame:
  # whether or not reproducibility is used
  np.random.seed(seed)

  ## constant structures
  # converts to matrix for navigation with numpy
  matrix = df.to_numpy()

  ## updated structures that should survive scope
  # initial centroids
  centroids_idx = np.random.randint(0,matrix.shape[0],k)
  current_centroids= matrix[centroids_idx,:]

  # empty structures that should survive scope (used in output)
  labels = np.empty(shape = (matrix.shape[0],), dtype=int)
  final_distances = np.empty(shape = (matrix.shape[0],), dtype=float) # reduces calculations
  clusterSSE = {} # returns individual clusters distortions
  TotalSSE = 0 # for elbow plot

  ## utility structures that must survive loops
  # what the next iterations loops start at
  next_centroids = np.empty((matrix.shape[1],k))
  # iterations of training
  iteration = 0

  ####  from here down to "wrap up" is a single training iteration
  while iteration < 300:
    
    for obj in matrix:
      obj_distances = np.empty(shape=(centroids_idx.shape[0],),dtype=float)
      for c in current_centroids:
        #finds distance between current row and current centroid
        obj_distance = np.sum((obj-c)**2)**0.5
        #keeps distances array parallel to current_centroids array
        obj_distances[np.where(current_centroids == c)] = obj_distance
      
      # updates training iteration's labels (index = label)
      dist_idx = np.argmin(obj_distances)
      labels[np.where(matrix == obj)] = dist_idx
      # updates final distances
      final_distances[np.where(matrix == obj)] = obj_distances[dist_idx]

    # computes new centroid set iteratively for each cluster 
    cluster_ids = np.sort(np.unique(labels)) # update to arange at some point

    for id in cluster_ids:
      cluster_mask = labels == id
      cluster = matrix[cluster_mask,]
      # array with mean of each column = center of current cluster
      next_centroids[id] = np.mean(cluster, axis=0) 

    # check the displacement of centroids
    total_centroid_movement = 0
    for i,j in zip(current_centroids, next_centroids):
      total_centroid_movement += np.sum((i-j)**2)**0.5
    
    if total_centroid_movement < (np.e**(-4)):
      break
    
    # if the loop gets here, updates the init_centroid of next iteration (minus edge case band aid)
    if iteration < 300:
      current_centroids = next_centroids

    iteration += 1

  #### results wrap up 
  # get distortion data
  cluster_ids = np.sort(np.unique(labels))
  for id in cluster_ids:
      cluster_mask = labels == id
      cluster_distortion = np.sum(final_distances**2[cluster_mask])
      clusterSSE[id] = cluster_distortion
      TotalSSE += cluster_distortion

  # append labels to df
  df["cluster_labels"] = labels

  return df, TotalSSE, clusterSSE, current_centroids


########### driver code ###########
if __name__ == "__main__":
  df = pd.read_csv("data_2D.txt", sep='\s+', header=None, names=['x_1','x_2'])

  print(f"""  
  Dataframe rows and columns:{df.shape}

  Dataframe Preview:
  {df.head()}""")

  a_kmeans(df,10)
