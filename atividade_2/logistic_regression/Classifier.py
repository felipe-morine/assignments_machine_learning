import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import math

def calculate_weight_vector(data: np.array, class_vector, learning_rate=0.1, epochs=600, min_gradient_value=10**-9):
    """
    Calcula o vetor de pesos
    :param data: dataset sem o vetor de classes
    :param class_vector: vetor de classe
    :param learning_rate: taxa de aprendizado
    :param epochs: numero de epocas
    :param min_gradient_value: valor minimo para o gradiente
    :return:
    """

    weight_vector = np.random.randn(len(data[0]))
    learning_rate *= -1

    for epochs in range(0, epochs):
        gradient = __calculate_gradient(weight_vector, np.copy(data), np.copy(class_vector))
        weight_vector = np.add(weight_vector, learning_rate*gradient)

        # print(np.linalg.norm(gradient))
        if min_gradient_value:
            if not __is_gradient_big(weight_vector, min_gradient_value):
                return weight_vector

    return weight_vector

def __calculate_gradient(weight_vector, data, class_vector):
    """
    calcula o gradiente
    :param weight_vector: vetor de pesos
    :param data: conjunto de dados sem o vetor de classe
    :param class_vector: vetor de classe
    :return:
    """
    n = len(data)
    sum = np.zeros(len(weight_vector))
    for index in range(0, n):
        w = np.copy(weight_vector)
        y = class_vector[index]
        x = data[index]
        numerator = y * x
        e_exponent = y * w
        e_exponent = np.dot(e_exponent, x)
        denominator = 1 + math.exp(e_exponent)

        partial_sum = np.true_divide(numerator, denominator)
        sum = np.add(sum, partial_sum)
    sum = -1 * sum
    gradient = np.true_divide(sum, n)
    return gradient

def __is_gradient_big(gradient, min_gradient_value):
    """
    verifica o tamanho do gradiente
    :param gradient:
    :param min_gradient_value:
    :return:
    """
    if np.linalg.norm(gradient) > min_gradient_value:
        return True
    return False


def calculate_results(weight_vector, test_data):
    """
    calcula os resultados. Foi definido um limiar de 0.5
    :param weight_vector: vetor de pesos
    :param test_data: dataset de teste sem o vetor de classes
    :return:
    """
    predicted_results = []

    for index in range(0, len(test_data)):
        s = np.dot(weight_vector, test_data[index])
        exp_s = math.exp(s)
        result = exp_s/(1 + exp_s)
        # print(result)
        if result >= 0.5:
            predicted_results.append(1)
        else:
            predicted_results.append(-1)

    return predicted_results

def error_rate(predicted_results, test_class_vector):
    """
    calcula a acuracia
    :param predicted_results: resultados do teste
    :param test_class_vector: vetor de classe original
    :return:
    """
    rate = 0
    for index in range(0, len(predicted_results)):
        # print(predicted_results[index], test_class_vector[index])
        if predicted_results[index] != test_class_vector[index]:
            rate += 1
    return 1-(rate/len(predicted_results))

def conf_matrix(predicted_results, test_class_vector):
    """
    calcula a matriz de confusao
    :param predicted_results: resutlados do teste
    :param test_class_vector: vetor de classe original
    :return:
    """
    cm = confusion_matrix(test_class_vector, predicted_results)
    return cm


def split_training_test_subsets(df):
    """
    divide o dataset para holdout
    :param df:
    :return:
    """
    pivot = len(df)
    pivot = int(pivot * (2/3))
    train = df[:pivot]
    test = df[pivot:]

    return train, test

def print_complete_dataset(dataset):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(dataset)
    return