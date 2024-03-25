from KNN import KNN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd

"""Cria um modelo KNN para o iris"""

def normalize(X_train, X_test):
    """
    Normaliza os conjuntos de dados
    :param X_train: conjunto de treinamento
    :param X_test: conjunto de teste
    :return: conjuntos normalizados
    """

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test


if __name__=="__main__":
    # carrega o dataset
    iris_training = pd.read_csv('../dataset/iris/iris_training.csv')
    class_label = 'class'

    # separa o vetor de classes
    X_training = iris_training.drop('class', axis=1)
    y_training = iris_training['class']

    # faz o mesmo para o conjunto de teste
    iris_test = pd.read_csv('../dataset/iris/iris_test.csv')
    X_test = iris_test.drop('class', axis=1)
    y_test = iris_test['class']

    # normaliza
    X_train, X_test = normalize(X_training, X_test)

    y_train = y_training.values

    # cria o modelo MUDAR PARAMETROS NESTA LINHA
    knn = KNN(k=4)
    knn.fit(X_train, y_train)

    y_predicted = knn.predict(X_test)

    acc = accuracy_score(y_test.values, y_predicted)
    cm = confusion_matrix(y_test.values, y_predicted)

    print(acc)
    print(cm)