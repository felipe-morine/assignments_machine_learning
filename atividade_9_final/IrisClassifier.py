from SVM import SVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from pandas_ml import ConfusionMatrix
from collections import Counter


def normalize(X_train, X_test, X_s_ve, X_s_vi, X_ve_vi):
    """
    normaliza os datasets
    :param X_train: 
    :param X_test: 
    :param X_s_ve: treinamento setosa vs. versicolor
    :param X_s_vi:  treinamento setosa vs. virginica
    :param X_ve_vi:  treinamento versicolor vs. virginica
    :return: 
    """

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    X_s_ve = scaler.transform(X_s_ve)
    X_s_vi = scaler.transform(X_s_vi)
    X_ve_vi = scaler.transform(X_ve_vi)

    return X_train, X_test, X_s_ve, X_s_vi, X_ve_vi

def revert_multilabel(class_vector, positive_label, negative_label):
    """
    reverte a classe do iris. 1 vira a classe positiva e -1 vira a negativa
    :param class_vector:
    :param positive_label:
    :param negative_label:
    :return:
    """

    iris_class_vector = []

    for i in range(len(class_vector)):
        if np.all(class_vector[i] == 1.0):
            iris_class_vector.append(positive_label)
        else:
            iris_class_vector.append(negative_label)
    return iris_class_vector

def one_vs_one(X_s_ve, X_s_vi, X_ve_vi):
    """
    one vs. one para votacao. retorna "tie" caso haja empate
    :param X_s_ve: classes para setosa vs. versicolor
    :param X_s_vi: classes para setosa vs. virginica
    :param X_ve_vi: classes para versicolor vs. virginica
    :return:
    """
    iris_class_vector = []

    for i in range(len(X_s_ve)):
        class_votes = [X_s_ve[i], X_s_vi[i], X_ve_vi[i]]
        counter = Counter(class_votes)
        most_common_list = counter.most_common(1)
        if len(most_common_list) > 1:
            iris_class_vector.append('tie')
        else:
            iris_class_vector.append(most_common_list[0][0])
    return iris_class_vector

if __name__=="__main__":
    # carrega o conjunto de treinamento total. Nao utilizado
    iris_training = pd.read_csv('dataset/iris/iris_training.csv')
    class_label = 'class'

    # separa o vetor de classes. Nao utilizado
    X_training = iris_training.drop('class', axis=1)
    y_training = iris_training['class']

    # carrega cada conjunto de treinamento, e separa vetor de classes
    # X_s_ve: setosa vs. versicolor
    # X_s_vi: setosa vs. virginica
    # X_ve_vi: versicolor vs. virginica
    iris_s_ve_training = pd.read_csv('dataset/iris/iris_setosa_versicolor.csv')
    X_s_ve_training = iris_s_ve_training.drop('class', axis=1)
    y_s_ve_training = iris_s_ve_training['class']

    iris_s_vi_training = pd.read_csv('dataset/iris/iris_setosa_virginica.csv')
    X_s_vi_training = iris_s_vi_training.drop('class', axis=1)
    y_s_vi_training = iris_s_vi_training['class']

    iris_ve_vi_training = pd.read_csv('dataset/iris/iris_versicolor_virginica.csv')
    X_ve_vi_training = iris_ve_vi_training.drop('class', axis=1)
    y_ve_vi_training = iris_ve_vi_training['class']


    # carrega conjunto de teste, e separa vetor de classes
    iris_test = pd.read_csv('dataset/iris/iris_test.csv')
    X_test = iris_test.drop('class', axis=1)
    y_test = iris_test['class']

    # normaliza
    X_training, X_test, X_s_ve_training, X_s_vi_training, X_ve_vi_training = normalize(X_training, X_test, X_s_ve_training, X_s_vi_training, X_ve_vi_training)

    # cria o modelo. MUDAR OS PARAMETROS LIVRES AQUI
    svm_setosa_vs_versicolor = SVM(kernel_name='polynomial', C=10, kernel_param=3)
    svm_setosa_vs_virginica = SVM(kernel_name='polynomial', C=10, kernel_param=3)
    svm_versicolor_vs_virginica = SVM(kernel_name='polynomial', C=10, kernel_param=3)

    svm_setosa_vs_versicolor.fit(X_s_ve_training, y_s_ve_training.values)
    svm_setosa_vs_virginica.fit(X_s_vi_training, y_s_vi_training.values)
    svm_versicolor_vs_virginica.fit(X_ve_vi_training, y_ve_vi_training.values)

    y_s_ve_predict = svm_setosa_vs_versicolor.predict(X_test)
    y_s_vi_predict = svm_setosa_vs_virginica.predict(X_test)
    y_ve_vi_predict = svm_versicolor_vs_virginica.predict(X_test)

    # reverte as labels de 1 e -1 para as do iris original
    y_s_ve_predict = revert_multilabel(y_s_ve_predict, 'Iris-setosa', 'Iris-versicolor')
    y_s_vi_predict = revert_multilabel(y_s_vi_predict, 'Iris-setosa', 'Iris-virginica')
    y_ve_vi_predict = revert_multilabel(y_ve_vi_predict, 'Iris-versicolor', 'Iris-virginica')

    # conduz votacao por one vs. one
    y_predicted = one_vs_one(y_s_ve_predict, y_s_vi_predict, y_ve_vi_predict)

    acc = accuracy_score(y_test.values, y_predicted)

    cm = ConfusionMatrix(y_test, y_predicted)

    print(acc)
    print(cm)