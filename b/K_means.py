'''Algoritmo k-Means
1. Seleccionar el número de k grupos (clusters)
2. Generar aleatoriamente k puntos que llamaremos centroides
3. Asignar cada elemento del conjunto de datos al centroide más cercano para
   formar k grupos
4. Reasignar la posición de cada centroide
5. Reasignar los elementos de datos al centroide más cercano nuevamente
   5.1 Si hubo elementos que se asignaron a un centroide distinto al original,
   regresar al paso 4, de lo contrario, el proceso ha terminado'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import sys
from sklearn import metrics

def init_random_centroids(n_clusters):
  random_centroids = {}
  for i in range(0,n_clusters):
    centroid = []
    for j in range(0,26):
      centroid.append(random.randint(1,5))
    random_centroids[i] = centroid
  return random_centroids


def silhoutte_score(clusters):
  X = []
  labels = []
  for k in clusters:
    X.extend(clusters[k])
    for i in range (0, len(clusters[k])):
      labels.append(k)
  sil_score = metrics.silhouette_score(X, labels, metric = 'euclidean')
  return sil_score

def ARI_score(clusters):
  dataset = pd.read_csv('data.csv')
  candidates_ids = dataset.iloc[:, 1].sort_values().tolist()
  labels_true = [0]
  for candidate_id in candidates_ids:
    if candidate_id < 5:
      labels_true.append(0)
    elif candidate_id < 12:
      labels_true.append(1)
    elif candidate_id < 18:
      labels_true.append(2)
    elif candidate_id == 18:
      labels_true.append(3)
    elif candidate_id == 19:
      labels_true.append(4)
    elif candidate_id == 20:
      labels_true.append(5)
    elif candidate_id == 21:
      labels_true.append(6)
    elif candidate_id == 22:
      labels_true.append(7)
    elif candidate_id == 23:
      labels_true.append(8)
    elif candidate_id == 24:
      labels_true.append(9)
    elif candidate_id == 25:
      labels_true.append(10)
  labels_pred = []
  for k in clusters:
    for i in range (0, len(clusters[k])):
      labels_pred.append(k)
  ari_score = metrics.adjusted_rand_score(labels_true, labels_pred)
  return ari_score

def init_empty_clusters(n_clusters):
  clusters = {}
  for i in range(0, n_clusters):
    clusters[i] = []

  return clusters

def add_to_nearest_cluster(centroids, dataset):
  clusters = init_empty_clusters(len(centroids.keys()))
  dataset_list = dataset.apply(lambda x: x.tolist(), axis = 1)
  for i in range(0, len(dataset_list)):
    v = np.array(dataset_list[i])
    distances = []
    for centroid in centroids:
      u = np.array(centroids[centroid])
      dist = np.linalg.norm(v-u)
      distances.append(dist)

    cluster = distances.index(min(distances))
    clusters[cluster].append(dataset_list[i])

  return clusters

def recalculate_centroids(clusters, n_clusters):
  new_centroids = {}
  for i in range(0, n_clusters):
    cluster_df = pd.DataFrame(clusters[i])
    mean = cluster_df.mean()
    new_centroids[i] = list(map(lambda x: x, mean))

  return new_centroids

def k_means(n_clusters):
  dataset = pd.read_csv('data.csv')
  dataset = dataset.iloc[:, range(2,28)]

  centroids = init_random_centroids(n_clusters)

  iterations = 0
  while iterations < 999:
    iterations += 1
    clusters = add_to_nearest_cluster(centroids, dataset)
    new_centroids = recalculate_centroids(clusters, n_clusters)
    if centroids == new_centroids:
      break
    else:
      centroids = new_centroids.copy()

  return clusters, iterations

if __name__ == "__main__":
  n_clusters = int(sys.argv[1])

  clusters, iterations = k_means(n_clusters)

  if iterations != 999:
    print('Total iterations: ', iterations)
    silh_score_reached = silhoutte_score(clusters)
    print("Silhouette score reached: " + str(silh_score_reached))

    ari_score_reached = ARI_score(clusters)
    print("ARI score reached: " + str(ari_score_reached))
  else:
    print("Limit of iterations reached")
