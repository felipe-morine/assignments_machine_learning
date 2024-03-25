import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import math


def calculate_weight_vector(data: np.array, class_vector, learning_rate=0.5, epochs=100, min_gradient_value=1e-5):
    """
    Calcula o vetor de pesos
    :param data: dataset sem o vetor de classes
    :param class_vector: vetor de classe
    :param learning_rate: taxa de aprendizado
    :param epochs: numero de epocas
    :param min_gradient_value: valor minimo para o gradiente
    :return:
    """


    weight_vector = np.matrix(np.random.randn(len(data[0]))/100)
    learning_rate *= -1

    for epochs in range(0, epochs):
        print(epochs)
        gradient = __calculate_gradient(weight_vector, np.copy(data), np.copy(class_vector))
        hessian = __calculate_hessian(weight_vector, np.copy(data), np.copy(class_vector))
        direction = __calculate_direction(gradient, hessian)
        weight_vector = np.add(weight_vector, learning_rate*direction)

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
        x = np.matrix(data[index])
        numerator = y * x
        e_exponent = y * w
        e_exponent = np.dot(e_exponent, x.T)
        denominator = 1 + np.exp(e_exponent)

        partial_sum = np.true_divide(numerator, denominator)
        sum = np.add(sum, partial_sum)
    sum = -1 * sum
    gradient = np.true_divide(sum, n)
    return gradient

def __calculate_hessian(weight_vector, data, class_vector):
    """
    Calcula a hessiana
    :param weight_vector: vetor de pesos
    :param data: conjunto de dados sem o vetor de classe
    :param class_vector: vetor de classe
    :return:
    """

    N = data.shape[0]
    D = weight_vector.shape[1]
    hessian_matrix = np.zeros(shape=(D, D))

    for index_d_first_derivative_term in range(0, D):
        for index_d_second_derivative_term in range(0, D):
            sum = 0
            for index_n in range(0, N):
                w = np.copy(weight_vector)
                y = class_vector[index_n]
                x = np.matrix(data[index_n])
                exp = calculate_logistic_exp(y, w, x)

                x1 = data[index_n, index_d_first_derivative_term]
                x2 = data[index_n, index_d_second_derivative_term]

                numerator = x1 * x2 * exp
                denominator = math.pow((1 + exp), 2)

                sum += (numerator/denominator)
            hessian_matrix[index_d_first_derivative_term, index_d_second_derivative_term] = sum/N
    return hessian_matrix


def __calculate_direction(gradient, hessian):
    """
    calcula a direcao para a atualização dos pesos
    :param gradient:
    :param hessian:
    :return:
    """

    hessian = np.linalg.inv(__modified_newton(hessian))

    return np.dot(hessian, gradient.T).T


def __modified_newton(hessian: np.matrix):
    """
    Atribui um valor para a positivacao da hessiana para sua inversao
    :param hessian:
    :return:
    """
    eigenvalues, eigenvectors = np.linalg.eig(hessian)
    smallest_eigenvalue = np.min(eigenvalues)

    if (smallest_eigenvalue > 0):
        return hessian
    else:
        add_term = abs(smallest_eigenvalue) + (np.random.rand()/100)
        return hessian + (add_term * np.eye(hessian.shape[0]))


def calculate_logistic_exp(y, weight_vector, x):
    """
    Calcula a funcao logistica e o expoente da funcao em termos de x, w e y
    :param y:
    :param weight_vector:
    :param x:
    :return:
    """
    result = y * weight_vector
    result = np.dot(result, x.T)
    result = np.exp(result)
    return result

def __is_gradient_big(gradient, min_gradient_value):
    """
    verifica o tamanho do gradiente
    :param gradient:
    :param min_gradient_value:
    :return:
    """
    norm = np.linalg.norm(gradient)
    if norm > min_gradient_value:
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
        exp_s = np.exp(s)
        result = exp_s/(1 + exp_s)
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