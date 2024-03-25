import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

"""Classificador linear"""

def __calculate_pseudo_inverse(data: np.array):
    """
    Calcula a pseudoinversa
    :param data: dataset sem vetor de classe
    :return:
    """

    x = np.copy(data)
    x_transpose = np.copy(x.T)

    x = np.dot(x_transpose, x)
    x_inverse = np.linalg.inv(x)
    x_transpose = np.copy(data.T)
    x = np.dot(x_inverse, x_transpose)

    return x

def calculate_weight_vector(data: np.array, class_vector):
    """
    Calcula vetor de pesos, de acordo com a pesudoinversa
    :param data: dataset sem vetor de classe
    :param class_vector: vetor de classe
    :return:
    """

    pseudo_inverse = __calculate_pseudo_inverse(data)
    weight_vector = np.dot(pseudo_inverse, class_vector)
    return weight_vector

def calculate_results(weight_vector, test_data):
    """
    Calcula resultados (testa o modelo). Limiar definido como 0.5
    :param weight_vector: vetor de pesos
    :param test_data: dataset sem vetor de classe
    :return:
    """

    predicted_results = []

    for index in range(0, len(test_data)):
        result = np.dot(weight_vector, test_data[index])
        if result >= 0.5:
            predicted_results.append(1)
        else:
            predicted_results.append(0)

    return predicted_results

def error_rate(predicted_results, test_class_vector):
    """
    calcula taxa de erro
    :param predicted_results:
    :param test_class_vector:
    :return:
    """

    rate = 0
    for index in range(0, len(predicted_results)):
        if predicted_results[index] != test_class_vector[index]:
            rate += 1
    return 1-(rate/len(predicted_results))

def conf_matrix(predicted_results, test_class_vector):
    """calcula matriz de confusao"""

    cm = confusion_matrix(test_class_vector, predicted_results)
    return cm


def split_training_test_subsets(df):
    """divide o dataset"""

    pivot = len(df)
    pivot = int(pivot * (2/3))
    train = df[:pivot]
    test = df[pivot:]

    return train, test

def print_complete_dataset(dataset):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(dataset)
    return