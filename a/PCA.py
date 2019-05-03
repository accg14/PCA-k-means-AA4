import pandas as pd
import numpy as np

def pca(dataset):
  matrix = dataset.values.transpose()
  matrix_rescaled = matrix - matrix.mean(axis = 1, keepdims = True)
  matrix_covariance = np.cov(matrix_rescaled)
  eig_val, eig_vec = np.linalg.eig(matrix_covariance)
  eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(len(eig_val))]
  eig_pairs.sort()
  eig_pairs.reverse()
  matrix_w = np.hstack((eig_pairs[0][1].reshape(26,1), eig_pairs[1][1].reshape(26,1)))
  print(np.dot(matrix_rescaled.T, matrix_w).transpose())

if __name__ == '__main__':
  dataset = pd.read_csv('data.csv')
  dataset = dataset.iloc[:, range(2,28)]
  pca(dataset)
