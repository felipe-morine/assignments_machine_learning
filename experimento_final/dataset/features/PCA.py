import pandas as pd
from sklearn.decomposition import PCA

# aplica PCA para os conjuntos de dados

file_full = "dataset_training_full.csv"
file_teste = "dataset_test.csv"
file_undersample = "dataset_training_undersample.csv"

df = pd.read_csv(file_full)
df_under = pd.read_csv(file_undersample)
df_test = pd.read_csv(file_teste)

columns = df.columns

y = df['class']
y_test = df_test['class']
y_under = df_under['class']

X = df.drop('class', axis=1)
X_under = df_under.drop('class', axis=1)
X_test = df_test.drop('class', axis=1)

pca = PCA(n_components=4)
pca.fit(X)

X = pca.transform(X)
X_under = pca.transform(X_under)
X_test = pca.transform(X_test)

# GRAVAR
columns = ['pc1', 'pc2', 'pc3', 'pc4']
X = pd.DataFrame(X, columns=columns)
dataset = pd.concat((X, y), axis=1)
# dataset.to_csv('dataset_training_full_PCA.csv', index=False)

X_under = pd.DataFrame(X_under, columns=columns)
under_dataset = pd.concat((X_under, y_under), axis=1)
# under_dataset.to_csv('dataset_training_undersample_PCA.csv', index=False)

X_test = pd.DataFrame(X_test, columns=columns)
test_dataset = pd.concat((X_test, y_test), axis=1)
# test_dataset.to_csv('dataset_test_PCA.csv', index=False)