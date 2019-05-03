import numpy as np
import pandas as pd
import sys
from matplotlib import pyplot as plt


def get_color(index):
    switcher = {
        0: "red",
        1: "green",
        2: "blue",
        3: "yellow",
        4: "cyan",
        5: "magenta",
        6: "black",
    }
    max_color = len(switcher)
    return switcher.get(index % max_color, "black")


def get_shape(index):
    switcher = {
        0: "o",
        1: "^",
        2: "s",
    }
    max_shape = len(switcher)
    return switcher.get(index % max_shape, "o")


# python PCA.py data.csv
# dataset filename: "data.csv"
if __name__ == "__main__":
    dataset = pd.read_csv(sys.argv[1], delimiter=",")
    dataset = dataset.iloc[:, range(1, 28)].sort_values(by=["candidatoId"])
    candidates = dataset["candidatoId"].value_counts().sort_index()
    dataset = dataset.iloc[:, range(1, 27)]

    num_rows = dataset.shape[0]
    num_columns = dataset.shape[1]

    datamatrix = np.array(dataset).transpose()
    datamatrix = datamatrix - datamatrix.mean(axis=1, keepdims=True)

    cvm = np.cov(datamatrix)
    eig_val_cov, eig_vec_cov = np.linalg.eig(cvm)

    for i in range(len(eig_val_cov)):
        eigvec_cov = eig_vec_cov[:, i].reshape(1, num_columns).T

    eig_pairs = [(np.abs(eig_val_cov[i]), eig_vec_cov[:, i])
                 for i in range(len(eig_val_cov))]

    eig_pairs.sort()
    eig_pairs.reverse()

    matrix_w = np.hstack((eig_pairs[0][1].reshape(
        num_columns, 1), eig_pairs[1][1].reshape(num_columns, 1)))

    transformed = np.dot(datamatrix.T, matrix_w).transpose()

    index = 0
    counter = 0
    for candidate in candidates:
        if not (index):
            plt.plot(transformed[0, 0:int(candidate)], transformed[1, 0:int(
                candidate)], get_shape(index), markersize=7, color=get_color(counter), alpha=0.5, label=candidate)
            prev_candidate = candidate
        else:
            plt.plot(transformed[0, int(prev_candidate):int(candidate)], transformed[1, int(
                prev_candidate):int(candidate)], get_shape(index), markersize=7, color=get_color(counter), alpha=0.5, label=candidate)
            prev_candidate = candidate
        index += candidate
        counter += 1

    plt.xlabel("x_values")
    plt.ylabel("y_values")
    plt.title("Instancias transformadas (con etiquetas)")
    plt.show()
