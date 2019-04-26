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

def init_random_centroids(n_clusters):
  random_centroids = {}
  for i in range(0,n_clusters):
    centroid = []
    for j in range(0,26):
      centroid.append(random.randint(1,5))
    random_centroids[i] = centroid
  return random_centroids


def silhoutte_score(clusters):
  return 1

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

def k_means(n_clusters, init_centroids):
  dataset = pd.read_csv('data.csv')
  dataset = dataset.iloc[:, range(2,28)]

  if init_centroids == 'random':
    centroids = init_random_centroids(n_clusters)
  else:
    centroids = init_heuristic_centroids(n_clusters)

  iterations = 0
  while True:
    iterations += 1
    clusters = add_to_nearest_cluster(centroids, dataset)
    new_centroids = recalculate_centroids(clusters, n_clusters)
    if centroids == new_centroids:
      break
    else:
      centroids = new_centroids.copy()

  print('Total iterations: ', iterations)
  return clusters

if __name__ == "__main__":
  n_clusters = int(sys.argv[1])
  init_centroids = sys.argv[2]

  clusters = k_means(n_clusters, init_centroids)
  print(clusters)
