from mlp.MLP import MLP
from sklearn.metrics import accuracy_score
import pandas as pd
from pandas_ml import ConfusionMatrix
from mlp import GeneralFunctions as gn

"""
Testa todos os modelos gerados para o relatorio.
"""

def test(k, is_undersample, is_PCA):
    dataset_path = '../dataset/features/'

    dataset_test_name = gn.get_dataset_test_filename(is_PCA)
    dataset_test = pd.read_csv(dataset_path + dataset_test_name)

    X_test = dataset_test.drop('class', axis=1)
    y_test = dataset_test['class']

    dummies_columns = pd.get_dummies(y_test).columns

    mlp = gn.load_model(k, is_undersample, is_PCA)

    y_predicted = mlp.predict(X_test.values)
    y_predicted = gn.revert_multilabel(y_predicted, dummies_columns)

    acc = accuracy_score(y_test.values, y_predicted)
    cm = ConfusionMatrix(y_test.values, y_predicted.values)

    acc = '{:.2%}'.format(acc)

    print(acc)
    print(cm)

    # SALVAR OS RESULTADOS
    # gn.save_results(acc, y_test.values, y_predicted.values, dummies_columns.values, k, is_undersample, is_PCA)


dataset_path = '../dataset/features/'
is_undersample = True
is_PCA = True
k_list = [3, 6, 9]

for k in k_list:
    test(k, False, False)
    test(k, True, False)
    test(k, False, True)
    test(k, True, True)
