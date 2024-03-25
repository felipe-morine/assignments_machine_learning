from CustomKeras import CustomKeras
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
    reverte a codificacao do Keras, ou seja, 0 = setosa, 1 = versicolor e 2 = virginica
    :param class_vector: vetor de classe a ser revertido
    :return: vetor de classe correto
    """
    iris_class_vector = []

    for i in range(len(class_vector)):
        if class_vector[i] == 0:
            iris_class_vector.append('Iris-setosa')
        elif class_vector[i] == 1:
            iris_class_vector.append('Iris-versicolor')
        else:
            iris_class_vector.append('Iris-virginica')
    return iris_class_vector


if __name__=="__main__":
    # carrega o dataset de treinamento
    iris_training = pd.read_csv('dataset/iris/iris_training.csv')
    class_label = 'class'

    # separa o vetor de classes
    X_training = iris_training.drop('class', axis=1)
    y_training = iris_training['class']

    # o mesmo para o dataset de teste
    iris_test = pd.read_csv('dataset/iris/iris_test.csv')
    X_test = iris_test.drop('class', axis=1)
    y_test = iris_test['class']

    # normaliza
    X_training, X_test = normalize(X_training, X_test)

    # dummifica o vetor de classe
    y_training = pd.get_dummies(y_training)
    y_training = y_training.values

    # cria o modelo. MUDAR OS PARAMETROS LIVRES AQUI
    keras_deep_network = CustomKeras()
    keras_deep_network.fit(X_training, y_training)

    y_predicted = keras_deep_network.predict(X_test)
    y_predicted = revert_multilabel(y_predicted)

    acc = accuracy_score(y_test.values, y_predicted)
    cm = ConfusionMatrix(y_test, y_predicted)

    print(acc)
    print(cm)