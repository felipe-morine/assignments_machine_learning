from SVM import SVM
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

if __name__=="__main__":
    # carrega o dataset
    hepatitis_training = pd.read_csv('dataset/hepatitis/hepatitis_training.csv')
    class_label = 'class'

    # separa o vetor de classes
    X_training = hepatitis_training.drop(class_label, axis=1)
    y_training = hepatitis_training[class_label]

    # troca o valor 0 para -1, como pede o SVM
    y_training = y_training.replace(0.0, -1.0)

    # faz o mesmo para o dataaet de teste
    hepatitis_test = pd.read_csv('dataset/hepatitis/hepatitis_test.csv')
    X_test = hepatitis_test.drop(class_label, axis=1)
    y_test = hepatitis_test[class_label]

    y_test = y_test.replace(0.0, -1.0)

    # normaliza
    X_training, X_test = normalize(X_training, X_test)
    y_train = y_training.values

    # cria o modelo. MUDAR OS PARAMETROS LIVRES AQUI
    svm = SVM(kernel_name='polynomial', C=10, kernel_param=3)
    svm.fit(X_training, y_train)

    y_predicted = svm.predict(X_test)

    acc = accuracy_score(y_test.values, y_predicted)

    # troca as saidas 1 e -1 por true e false. apenas para motivos de vsualizacao, nao necessario
    y_test = y_test.values
    y_test[y_test == 1.0] = True
    y_test[y_test == -1.0] = False

    y_predicted[y_predicted == 1.0] = True
    y_predicted[y_predicted == -1.0] = False

    cm = ConfusionMatrix(y_test, y_predicted)

    print(acc)
    print(cm)