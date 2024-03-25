from sklearn.preprocessing import StandardScaler
import pandas as pd

# normaliza os dados

file_full = "dataset_training_full.csv"
file_teste = "dataset_test.csv"

df = pd.read_csv(file_full)
df_test = pd.read_csv(file_teste)

y = df['class']
y_test = df_test['class']

X = df.drop('class', axis=1)
X_test = df_test.drop('class', axis=1)

X_columns = X.columns

scaler = StandardScaler()
scaler.fit(X)
X_train = scaler.transform(X)
X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train, columns=X_columns)

X_test = pd.DataFrame(X_test, columns=X_columns)

training_df = pd.concat((X_train, y), axis=1)
test_df = pd.concat((X_test, y_test), axis=1)

# GRAVAR
# training_df.to_csv('dataset_training_full.csv', index=False)
# test_df.to_csv('dataset_test.csv', index=False)
