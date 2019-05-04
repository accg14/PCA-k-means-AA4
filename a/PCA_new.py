import numpy as np
import pandas as pd
import sys
from matplotlib import pyplot as plt


import pdb


def get_plot(index):
    switcher = {
        1: ["s", "blue"],
        2: ["s", "blue"],
        3: ["s", "blue"],
        4: ["s", "blue"],

        5: ["^", "lightblue"],
        6: ["^", "lightblue"],
        7: ["^", "lightblue"],
        8: ["^", "lightblue"],
        9: ["^", "lightblue"],
        10: ["^", "lightblue"],
        11: ["^", "lightblue"],

        12: ["o", "red"],
        13: ["o", "red"],
        14: ["o", "red"],
        15: ["o", "red"],
        16: ["o", "red"],
        17: ["o", "red"],

        18: ["d", "purple"],

        19: ["+", "black"],

        20: ["*", "green"],

        21: ["x", "orange"],

        22: ["D", "yellow"],

        23: ["X", "lightgreen"],

        24: ["1", "brown"],

        25: ["P", "grey"],
    }
    len_switcher = len(switcher) + 1
    shape = switcher.get(index % len_switcher, ["-", "black"])[0]
    color = switcher.get(index % len_switcher, ["-", "black"])[1]
    return shape, color


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

    pdb.set_trace()

    counter = 1
    prev_index = 0
    index = 0
    for candidate in candidates:
        index += candidate
        shape, color = get_plot(counter)
        plt.plot(transformed[0, prev_index:index], transformed[1, prev_index:index],
                 shape, markersize=7, color=color, alpha=0.5, label=candidate)
        prev_index = index
        counter += 1

    plt.xlabel("x_values")
    plt.ylabel("y_values")
    plt.legend(loc='upper left')
    plt.title("Instancias transformadas (con etiquetas)")
    plt.show()
