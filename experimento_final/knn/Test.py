from knn.KNN import KNN
from sklearn.metrics import accuracy_score
import pandas as pd
from pandas_ml import ConfusionMatrix
from knn import GeneralFunctions as gn


"""
Testa o KNN com a matriz de suporte correspondente jah calculada. Para mais detalhes, ver a classe KNN
"""

dataset_path = '../dataset/features/'

# PARAMS
is_undersample = True
is_PCA = True
k = 3

# carrega o dataset
dataset_test_name = gn.get_dataset_test_filename(is_PCA)
dataset_test = pd.read_csv(dataset_path + dataset_test_name)

# separa o vetor de classes
X_test = dataset_test.drop('class', axis=1)
y_test = dataset_test['class']

# cria o modelo com a matriz de suporte carregada
knn = KNN(k)
knn = gn.load_model(knn, is_undersample, is_PCA)

# faz a predicao
y_predicted = knn.predict(X_test.values)

# calcula os resultados
acc = accuracy_score(y_test.values, y_predicted)
cm = ConfusionMatrix(y_test, y_predicted)
acc = '{:.2%}'.format(acc)
dummies_columns = pd.get_dummies(y_test).columns

print(acc)
print(cm)

# SALVAR OS RESULTADOS
# gn.save_results(acc, y_test.values, y_predicted, dummies_columns.values, k, is_undersample, is_PCA)