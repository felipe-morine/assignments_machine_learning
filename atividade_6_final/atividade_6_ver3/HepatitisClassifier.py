from Perceptron import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd
from pandas_ml import ConfusionMatrix
import numpy as np


def normalize(X_train, X_test):
    """
    normaliza os datasets de treinamento e teste
    :param X_train:
    :param X_test:
    :return:
    """
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test

def revert_multilabel(class_vector):
    """
    reverte a dummificacao, ou seja, (1, 0) = 0 e (0,1) = 1,
    :param class_vector: vetor de classe a ser revertido
    :return: vetor de classe correto
    """
    reverted_class_vector = np.zeros(len(class_vector))

    for i in range(len(class_vector)):
        if np.all(class_vector[i] == [0.0, 1.0]):
            reverted_class_vector[i] = 1
        else:
            reverted_class_vector[i] = 0
    return reverted_class_vector

if __name__=="__main__":
    # carrega o conjunto de treinamento
    hepatitis_training = pd.read_csv('dataset/hepatitis/hepatitis_training.csv')
    class_label = 'class'

    # separa o vetor de classes
    X_training = hepatitis_training.drop(class_label, axis=1)
    y_training = hepatitis_training[class_label]

    # o mesmo para o conjunto de teste
    hepatitis_test = pd.read_csv('dataset/hepatitis/hepatitis_test.csv')
    X_test = hepatitis_test.drop(class_label, axis=1)
    y_test = hepatitis_test[class_label]

    X_training = X_training.values
    X_test = X_test.values

    # normaliza
    X_training, X_test = normalize(X_training, X_test)

    # dummifica
    y_training = pd.get_dummies(y_training)

    y_train = y_training.values

    # cria o modelo. MUDAR OS PARAMETROS LIVRES AQUI
    perceptron = Perceptron()
    perceptron.fit(X_training, y_train)

    y_predicted = perceptron.predict(X_test)
    y_predicted = revert_multilabel(y_predicted)

    acc = accuracy_score(y_test.values, y_predicted)

    cm = ConfusionMatrix(y_test, y_predicted)

    print(acc)
    print(cm)