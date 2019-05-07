import numpy as np
import pandas as pd
import sys
from matplotlib import pyplot as plt


def get_plot_values(value, unicity):
    dummy = ["-", "black", "Partido Desconocido"]
    switcher = {
        1: ["s", "blue", "Frente Amplio"],
        2: ["s", "blue", "Frente Amplio"],
        3: ["s", "blue", "Frente Amplio"],
        4: ["s", "blue", "Frente Amplio"],
        5: ["^", "lightblue", "Partido Nacional"],
        6: ["^", "lightblue", "Partido Nacional"],
        7: ["^", "lightblue", "Partido Nacional"],
        8: ["^", "lightblue", "Partido Nacional"],
        9: ["^", "lightblue", "Partido Nacional"],
        10: ["^", "lightblue", "Partido Nacional"],
        11: ["^", "lightblue", "Partido Nacional"],
        12: ["o", "red", "Partido Colorado"],
        13: ["o", "red", "Partido Colorado"],
        14: ["o", "red", "Partido Colorado"],
        15: ["o", "red", "Partido Colorado"],
        16: ["o", "red", "Partido Colorado"],
        17: ["o", "red", "Partido Colorado"],
        18: ["d", "purple", "La Alternativa"],
        19: ["+", "black", "Unidad Popular"],
        20: ["*", "green", "Partido de la Gente"],
        21: ["x", "orange", "PERI"],
        22: ["D", "yellow", "Partido de los Trabajadores"],
        23: ["X", "lightgreen", "Partido Digital"],
        24: ["1", "brown", "Partido Verde"],
        25: ["P", "grey", "Partido de Todos"],
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
    if (value < 5):
        return 1, "Frente Amplio"
    elif (value < 12):
        return 4, "Partido Nacional"
    elif (value < 18):
        return 5, "Partido Colorado"
    elif (value == 18):
        return 2, "La Alternativa"
    elif (value == 19 or value == 22):
        return 3, "Unidad Popular - Partido de los Trabajadores"
    elif (value == 20):
        return 6, "Partido de la Gente"
    else:
        return 7, "Otros"


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

    value = 1
    prev_index = 0
    index = 0
    unicity = bool(int(sys.argv[2]))
    for candidate in candidates:
        index += candidate
        shape, color, party = get_plot_values(value, unicity)
        if not (unicity):
            position, title = get_subplot_position(value)
            plt.subplot(330 + position)
            plt.title(title)
        plt.plot(transformed[0, prev_index:index], transformed[1, prev_index:index],
                 shape, markersize=7, color=color, alpha=0.5, label=party + " - (" + str(candidate) + ")")
        prev_index = index
        value += 1
    plt.subplots_adjust(hspace=0.75, wspace=0.35)

    plt.show()
