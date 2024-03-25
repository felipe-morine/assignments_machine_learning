import pandas as pd
from sklearn.model_selection import train_test_split
from dataset.hepatitis import MissingValuesHandler


def format_binary_attributes(dataset: pd.DataFrame,  columns_list: []):
    """
    Formata atributos binarios com valores (1, 2) para (0, 1). Usado para o dataset de hepatite.
    :param dataset:
    :return:
    """
    for column in columns_list:
        dataset[column] = pd.to_numeric(dataset[column])
        dataset[column] = dataset[column].replace(1, 0)
        dataset[column] = dataset[column].replace(2, 1)
    return dataset

# carrega o dataset
filename = 'hepatitis.csv'
diabetes_df = pd.read_csv(filename)

# separa o vetor de classes
y = diabetes_df['class']
X = diabetes_df.drop('class', axis=1)

# separa o conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

# trata os missing values
missing_values_columns_mean = ['bilirubin', 'alk_phosphate', 'sgot', 'albumin', 'protime']
missing_values_columns_mode = ['steroid', 'fatigue', 'malaise', 'anorexia', 'liver_big', 'liver_firm',
                               'spleen_palpable',
                               'spiders', 'ascites', 'varices']
missing_values_value = '?'

X_train, X_test = MissingValuesHandler.set_numpy_missing_values(X_train, X_test, missing_values_value, (missing_values_columns_mode + missing_values_columns_mean))
X_train, X_test = MissingValuesHandler.imputation_with_mean(X_train, X_test, missing_values_columns_mean)
X_train, X_test = MissingValuesHandler.imputation_with_mode(X_train, X_test, missing_values_columns_mode)

# ver acima
format_binary_columns = missing_values_columns_mode

X_train = format_binary_attributes(X_train, format_binary_columns)
X_test = format_binary_attributes(X_test, format_binary_columns)

# faz o mesmo para o vetor de classes
y_train = y_train.replace(1, 0)
y_train = y_train.replace(2, 1)

y_test = y_test.replace(1, 0)
y_test = y_test.replace(2, 1)

# grava os datasets
training_df = pd.concat((X_train, y_train), axis=1)
test_df = pd.concat((X_test, y_test), axis=1)

training_df.to_csv('hepatitis_training.csv', index=False)
test_df.to_csv('hepatitis_test.csv', index=False)


