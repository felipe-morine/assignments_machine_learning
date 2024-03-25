from Perceptron import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from pandas_ml import ConfusionMatrix

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
    # carrega o dataset
    diabetes_training = pd.read_csv('dataset/diabetes/diabetes_training.csv')
    class_label = 'Outcome'

    # separa o vetor de classes
    X_training = diabetes_training.drop(class_label, axis=1)
    y_training = diabetes_training[class_label]

    # faz o mesmo para o conjunto de teste
    diabetes_test = pd.read_csv('dataset/diabetes/diabetes_test.csv')
    X_test = diabetes_test.drop(class_label, axis=1)
    y_test = diabetes_test[class_label]

    # normaliza o dataset
    X_training, X_test = normalize(X_training, X_test)

    # dummifica o vetor de classes
    y_training = pd.get_dummies(y_training)
    y_training = y_training.values

    # cria o modelo. MUDAR OS PARAMETROS LIVRES AQUI
    perceptron = Perceptron()
    perceptron.fit(X_training, y_training)

    y_predicted = perceptron.predict(X_test)
    y_predicted = revert_multilabel(y_predicted)

    acc = accuracy_score(y_test.values, y_predicted)

    cm = ConfusionMatrix(y_test, y_predicted)

    print(acc)
    print(cm)