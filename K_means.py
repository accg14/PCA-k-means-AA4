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

def init_random_centroids():

def init_heuristic_centroids():

def add_to_nearest_centroid(centroids, dataset):
  for row in dataset.iterrows():


def k_means(n_clusters, init_centroids):
  dataset = pd.read_csv('data.csv')
  dataset = dataset.iloc[:, range(2,28)]

  if init_centroids == 'random':
    init_random_centroids()
  else:
    init_heuristic_centroids()
