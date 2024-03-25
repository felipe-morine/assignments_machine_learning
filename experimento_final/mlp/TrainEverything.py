from mlp.MLP import MLP
from sklearn.metrics import accuracy_score
import pandas as pd
from pandas_ml import ConfusionMatrix
from mlp import GeneralFunctions as gn

"""Treina os modelos MLPs criados para o relatorio"""

def train(k, is_undersample, is_PCA):
    dataset_training_name = gn.get_dataset_training_filename(is_undersample, is_PCA)

    dataset_test_name = gn.get_dataset_test_filename(is_PCA)
    dataset_training = pd.read_csv(dataset_path + dataset_training_name)
    dataset_test = pd.read_csv(dataset_path + dataset_test_name)

    X_training = dataset_training.drop('class', axis=1)
    y_training = dataset_training['class']

    X_test = dataset_test.drop('class', axis=1)
    y_test = dataset_test['class']

    y_training = pd.get_dummies(y_training)
    dummies_columns = y_training.columns

    mlp = MLP()
    mlp.fit(X_training.values, y_training.values, hidden_layer_size=k)

    # SALVAR O MODELO
    # gn.save_model(mlp, k, is_undersample, is_PCA)

    y_predicted = mlp.predict(X_test.values)
    y_predicted = gn.revert_multilabel(y_predicted, dummies_columns)

    acc = accuracy_score(y_test.values, y_predicted)
    cm = ConfusionMatrix(y_test, y_predicted)

    acc = '{:.2%}'.format(acc)

    print(acc)
    print(cm)

    return

dataset_path = '../dataset/features/'

is_undersample = True
is_PCA = True
k_list = [3, 6, 9]

for k in k_list:
    train(k, False, False)
    train(k, True, False)
    train(k, False, True)
    train(k, True, True)

