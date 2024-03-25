from knn.KNN import KNN
from sklearn.metrics import accuracy_score
import pandas as pd
from pandas_ml import ConfusionMatrix
from knn import GeneralFunctions as gn


"""
Treina um modelo. Para Ks maiores que 5, os diretorios necessarios devem ser criados e deve ser
aumentado o parametro max_support_instances. Para mais detalhes, ver classe KNN
"""


dataset_path = '../dataset/features/'

# PARAMS
is_undersample = True
is_PCA = True
k = 3

# carrega os datasets
dataset_training_name = gn.get_dataset_training_filename(is_undersample, is_PCA)
dataset_test_name = gn.get_dataset_test_filename(is_PCA)
dataset_training = pd.read_csv(dataset_path+dataset_training_name)
dataset_test = pd.read_csv(dataset_path+dataset_test_name)

# separa os vetores de classe
X_training = dataset_training.drop('class', axis=1)
y_training = dataset_training['class']
X_test = dataset_test.drop('class', axis=1)
y_test = dataset_test['class']

# treina o modelo
knn = KNN(k)
knn.fit(X_training.values, y_training.values)
# treina o modelo

# testa o modelo
y_predicted = knn.predict(X_test.values)

# SALVAR O MODELO
# gn.save_model(knn, is_undersample, is_PCA)

# imprime acuracia e matriz de confusao
dummies_columns = pd.get_dummies(y_test).columns
acc = accuracy_score(y_test.values, y_predicted)
cm = ConfusionMatrix(y_test, y_predicted)

acc = '{:.2%}'.format(acc)

print(acc)
print(cm)

