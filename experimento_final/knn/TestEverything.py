from knn.KNN import KNN
from sklearn.metrics import accuracy_score
import pandas as pd
from pandas_ml import ConfusionMatrix
from knn import GeneralFunctions as gn

"""
Testa todos os modelos KNN criados para o relatorio, com as matrizes de suporte correspondentes ja calculadas.
Para mais detalhes, ver a classe KNN
"""

def test(k, is_undersample, is_PCA):
    dataset_path = '../dataset/features/'

    dataset_test_name = gn.get_dataset_test_filename(is_PCA)
    dataset_test = pd.read_csv(dataset_path + dataset_test_name)

    X_test = dataset_test.drop('class', axis=1)
    y_test = dataset_test['class']

    knn = KNN(k)
    knn = gn.load_model(knn, is_undersample, is_PCA)

    y_predicted = knn.predict(X_test.values)

    acc = accuracy_score(y_test.values, y_predicted)
    cm = ConfusionMatrix(y_test, y_predicted)

    acc = '{:.2%}'.format(acc)

    dummies_columns = pd.get_dummies(y_test).columns

    print(acc)
    print(cm)

    # SALVAR OS RESULTADOS
    # gn.save_results(acc, y_test.values, y_predicted, dummies_columns.values, k, is_undersample, is_PCA

    return

dataset_path = '../dataset/features/'

is_undersample = True
is_PCA = True
k_list = [3, 4, 5]

for k in k_list:
    test(k, False, False)
    test(k, True, False)
    test(k, False, True)
    test(k, True, True)

