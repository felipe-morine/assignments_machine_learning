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

if __name__=="__main__":
    # carrega o dataset
    diabetes_training = pd.read_csv('dataset/diabetes/diabetes_training.csv')
    class_label = 'Outcome'

    # separa o vetor de classes
    X_training = diabetes_training.drop(class_label, axis=1)
    y_training = diabetes_training[class_label]

    # o mesmo para o dataset de teste
    diabetes_test = pd.read_csv('dataset/diabetes/diabetes_test.csv')
    X_test = diabetes_test.drop(class_label, axis=1)
    y_test = diabetes_test[class_label]

    # normaliza
    X_training, X_test = normalize(X_training, X_test)

    # dummifica
    y_training = pd.get_dummies(y_training)
    y_training = y_training.values

    # cria o modelo. MUDAR OS PARAMETROS LIVRES AQUI
    keras_deep_network = CustomKeras()
    keras_deep_network.fit(X_training, y_training)

    y_predicted = keras_deep_network.predict(X_test)

    acc = accuracy_score(y_test.values, y_predicted)
    cm = ConfusionMatrix(y_test, y_predicted)

    print(acc)
    print(cm)