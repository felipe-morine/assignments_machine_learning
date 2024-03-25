import pandas as pd
from sklearn.model_selection import train_test_split

# Estratifica o dataset e cria os conjuntos de treinamento e teste

column_names = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'class']
filename = 'features.csv'
dataset = pd.read_csv(filename, sep='\t', names=column_names)

X = dataset.drop('class', axis=1)
y = dataset['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

training_df = pd.concat((X_train, y_train), axis=1)
test_df = pd.concat((X_test, y_test), axis=1)

# GRAVAR
# training_df.to_csv('dataset_training_full.csv', index=False)
# test_df.to_csv('dataset_test.csv', index=False)
