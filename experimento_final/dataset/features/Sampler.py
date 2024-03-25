import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

# cria o dataset reduzido

filename = 'dataset_training_full.csv'
dataset = pd.read_csv(filename)

X = dataset.drop('class', axis=1)
X_columns = X.columns
y = dataset['class']

values = dataset['class'].value_counts()

# classe 1 reduzida para 7000 instancias
rus = RandomUnderSampler(ratio={1: 7000})
X_rus, y_rus = rus.fit_sample(X, y)

y_rus_series = pd.Series(y_rus)

X_under = pd.DataFrame(X_rus, columns=X_columns)
y_under = pd.Series(y_rus, name='class')

under_dataset = pd.concat((X_under, y_under), axis=1)

# GRAVAR
# under_dataset.to_csv('dataset_training_undersample.csv', index=False)