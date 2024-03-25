import numpy as np
import math

"""
Utiliza a estrategioa One vs. All para tratar o problema multiclasse do Iris.
"""

def calculate_results(weight_vector, test_data):
    """
    Calcula a saida, sem o limiar de 0.5
    :param weight_vector: vetor de pesos
    :param test_data: dataset de teste, sem o vetor de classes
    :return:
    """
    predicted_results = []

    for index in range(0, len(test_data)):
        s = np.dot(weight_vector, test_data[index])
        exp_s = math.exp(s)
        result = exp_s/(1 + exp_s)
        predicted_results.append(result)

    return predicted_results

def calculate_iris_results(setosa_results, versicolor_results, virginica_results):
    """
    Calcula o resultado de acordo com os modelos treinados. Aquele cuja saida for maior eh a classe
    (estrategia one vs. all)
    :param setosa_results: saidas do modelo de setosa
    :param versicolor_results: saidas do modelo de versicolor
    :param virginica_results: saidas do modelo viriginca
    :return:
    """
    predicted_results = []
    for index in range(0, len(setosa_results)):
        setosa = setosa_results[index]
        versicolor = versicolor_results[index]
        virginica = virginica_results[index]
        if setosa > versicolor:
            if setosa > virginica:
                predicted_results.append('setosa')
            else:
                predicted_results.append('virginica')
        elif versicolor > virginica:
            predicted_results.append('versicolor')
        else:
            predicted_results.append('virginica')
    return predicted_results

def return_iris_classes(class_vector):
    """
    retorna a codificao certa para o problema de multiclasses
    :param class_vector: o vetor de classes (de teste)
    :return:
    """
    real_class_vector = []
    for index in class_vector:
        setosa = index[0]
        versicolor = index[1]
        virginica = index[2]
        if setosa > versicolor:
            if setosa > virginica:
                real_class_vector.append('setosa')
            else:
                real_class_vector.append('virginica')
        elif versicolor > virginica:
            real_class_vector.append('versicolor')
        else:
            real_class_vector.append('virginica')
    return real_class_vector