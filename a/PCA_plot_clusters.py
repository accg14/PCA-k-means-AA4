from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd
import sys
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
np.set_printoptions(suppress=True, precision=3)


def get_plot_values(value, unicity):
    dummy = ["-", "black", "Partido Desconocido"]
    switcher = {
        1: ["s", "blue", "Clúster 1"],
        2: ["o", "red", "Clúster 2"],
        3: ["^", "lightblue", "Clúster 3"],
        4: ["d", "purple", "Clúster 4"],
        5: ["+", "black", "Clúster 5"],
        6: ["*", "green", "Clúster 6"],
        7: ["x", "orange", "Clúster 7"],
        8: ["D", "yellow", "Clúster 8"],
        9: ["X", "lightgreen", "Clúster 9"],
        10: ["1", "brown", "Clúster 10"],
        11: ["P", "grey", "Clúster 11"],
    }
    switch = value % (len(switcher) + 1)
    shape = switcher.get(switch, dummy)[0]
    color = switcher.get(switch, dummy)[1]
    party = switcher.get(switch, dummy)[2]
    if (unicity):
        return shape, color, party
    else:
        return "+", color, party


def get_subplot_position(value):
    if (value == 1):
        return 1, "1"
    elif (value == 2):
        return 2, "2"
    elif (value == 3):
        return 3, "3"
    elif (value == 4):
        return 4, "4"
    elif (value == 5):
        return 5, "5"
    elif (value == 6):
        return 6, "6"
    elif (value == 7):
        return 7, "7"
    elif (value == 8):
        return 8, "8"
    else:
        return 9, "9"


# python PCA_plot_clusters.py x.txt 2 1
if __name__ == "__main__":
    dataset = pd.read_csv(sys.argv[1], delimiter=",")
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

    dimensions = int(sys.argv[2])
    unicity = bool(int(sys.argv[3]))
    fig = plt.figure(figsize=(15, 5))

    if (dimensions == 1):
        matrix_w = np.hstack((eig_pairs[0][1].reshape(num_columns, 1),))
    elif (dimensions == 2):
        matrix_w = np.hstack((eig_pairs[0][1].reshape(
            num_columns, 1), eig_pairs[1][1].reshape(num_columns, 1)))
    else:
        matrix_w = np.hstack((eig_pairs[0][1].reshape(num_columns, 1), eig_pairs[1][1].reshape(
            num_columns, 1), eig_pairs[2][1].reshape(num_columns, 1)))
        if (unicity):
            ax = fig.add_subplot(111, projection='3d')

    transformed = np.dot(datamatrix.T, matrix_w).transpose()

    value = 1
    prev_index = 0
    index = 0
    for candidate in candidates:
        index += candidate
        shape, color, party = get_plot_values(value, unicity)
        if (dimensions == 1):
            if not (unicity):
                position, title = get_subplot_position(value)
                plt.subplot(330 + position)
                plt.title(title)
            plt.plot(transformed[0, prev_index:index], shape, markersize=7,
                     color=color, alpha=0.5, label=party + " - (" + str(candidate) + ")")
        elif (dimensions == 2):
            if not (unicity):
                position, title = get_subplot_position(value)
                plt.subplot(330 + position)
                plt.title(title)
            plt.plot(transformed[0, prev_index:index], transformed[1, prev_index:index], shape,
                     markersize=7, color=color, alpha=0.5, label=party + " - (" + str(candidate) + ")")
        else:
            if not (unicity):
                position, title = get_subplot_position(value)
                ax = fig.add_subplot(330 + position, projection='3d')
                plt.title(title)
            ax.plot(transformed[0, prev_index:index], transformed[1, prev_index:index], transformed[2, prev_index:index],
                    shape, markersize=7, color=color, alpha=0.5, label=party + " - (" + str(candidate) + ")")
        prev_index = index
        value += 1
    plt.subplots_adjust(hspace=0.75, wspace=0.35)

    if (unicity):
        plt.legend(loc='upper center', bbox_to_anchor=(
            0.5, -0.15), shadow=False, ncol=2)

    plt.show()
