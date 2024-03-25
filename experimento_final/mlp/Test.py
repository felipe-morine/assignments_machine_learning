from mlp.MLP import MLP
from sklearn.metrics import accuracy_score
import pandas as pd
from pandas_ml import ConfusionMatrix
from mlp import GeneralFunctions as gn

"""testa um modelo MLP criado, carregando as matrizes de pesos"""

# PARAMS
dataset_path = '../dataset/features/'
is_undersample = True
is_PCA = True
k = 3
# PARAMS

# carrega o dataset
dataset_test_name = gn.get_dataset_test_filename(is_PCA)
dataset_test = pd.read_csv(dataset_path+dataset_test_name)

# separa o vetor de classe
X_test = dataset_test.drop('class', axis=1)
y_test = dataset_test['class']

dummies_columns = pd.get_dummies(y_test).columns

# carrega o modelo
mlp = gn.load_model(k, is_undersample, is_PCA)

# realiza os teste
y_predicted = mlp.predict(X_test.values)
y_predicted = gn.revert_multilabel(y_predicted, dummies_columns)


# mostra os resultados
acc = accuracy_score(y_test.values, y_predicted.values)
cm = ConfusionMatrix(y_test.values, y_predicted.values)

acc = '{:.2%}'.format(acc)

print(acc)
print(cm)

# SALVAR OS RESULTADOS
# gn.save_results(acc, y_test.values, y_predicted.values, dummies_columns.values, k, is_undersample, is_PCA)