from knn.KNN import KNN
from sklearn.metrics import accuracy_score
import pandas as pd
from pandas_ml import ConfusionMatrix
from knn import GeneralFunctions as gn

"""
Testa todos os modelos KNN criados para o relatorio, com as matrizes de suporte correspondentes ja calculadas.
Para mais detalhes, ver a classe KNN.
"""

def train(k, is_undersample, is_PCA):
    dataset_path = '../dataset/features/'

    dataset_training_name = gn.get_dataset_training_filename(is_undersample, is_PCA)

    dataset_test_name = gn.get_dataset_test_filename(is_PCA)
    dataset_training = pd.read_csv(dataset_path + dataset_training_name)
    dataset_test = pd.read_csv(dataset_path + dataset_test_name)

    X_training = dataset_training.drop('class', axis=1)
    y_training = dataset_training['class']

    X_test = dataset_test.drop('class', axis=1)
    y_test = dataset_test['class']

    knn = KNN(k)
    knn.fit(X_training.values, y_training.values)

    y_predicted = knn.predict(X_test.values)

    # SALVAR O MODELO
    gn.save_model(knn, is_undersample, is_PCA)

    acc = accuracy_score(y_test.values, y_predicted)
    cm = ConfusionMatrix(y_test, y_predicted)

    acc = '{:.2%}'.format(acc)

    dummies_columns = pd.get_dummies(y_test).columns

    print(acc)
    print(cm)


    return

dataset_path = '../dataset/features/'

is_undersample = True
is_PCA = True
k_list = [3, 4, 5]

for k in k_list:
    train(k, False, False)
    train(k, True, False)
    train(k, False, True)
    train(k, True, True)

