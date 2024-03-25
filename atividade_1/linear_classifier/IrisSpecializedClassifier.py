import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

"""similar ao Classifier.py, mas faz o processamento para multiclasse"""

def __calculate_pseudo_inverse(data: np.array):
    x = np.copy(data)
    x_transpose = np.copy(x.T)

    x = np.dot(x_transpose, x)
    x_inverse = np.linalg.inv(x)
    x_transpose = np.copy(data.T)
    x = np.dot(x_inverse, x_transpose)

    return x

def calculate_weight_vector(data: np.array, class_vector):
    pseudo_inverse = __calculate_pseudo_inverse(data)
    weight_vector = np.dot(pseudo_inverse, class_vector)
    return weight_vector

def calculate_results(weight_vector, test_data):
    predicted_results = []

    for index in range(0, len(test_data)):
        result = np.dot(weight_vector.T, test_data[index].T)
        predicted_results.append(result)

    return predicted_results

def return_iris_classes(class_vector):
    """processamento para o problema multiclasse do iris. retorna o maior valor"""

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


def error_rate(predicted_results, test_class_vector):
    rate = 0
    for index in range(0, len(predicted_results)):
        if predicted_results[index] != test_class_vector[index]:
            rate += 1
    return 1-(rate/len(predicted_results))

def conf_matrix(predicted_results, test_class_vector):
    cm = confusion_matrix(test_class_vector, predicted_results)
    return cm


def split_training_test_subsets(df):
    pivot = len(df)
    pivot = int(pivot * (2/3))
    train = df[:pivot]
    test = df[pivot:]

    return train, test

def print_complete_dataset(dataset):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(dataset)
    return