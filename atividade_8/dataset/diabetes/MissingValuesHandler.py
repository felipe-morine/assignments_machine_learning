import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer


def set_numpy_missing_values(dataset_training: pd.DataFrame, dataset_test: pd.DataFrame, missing_values_value, columns_list:[]):
    """
    Transforma os valores "nulos" do dataset em valores NaN definidos pela biblioteca numpy
    :param dataset
    :param missing_values_value: o valor do missing value no dataset. Ex: '?' para o dataset Hepatitis
    :param columns_list: lista de attributos em que a funcao deve ser aplicada. Ex: dataset Diabetes
        permite numero de gravidez  = 0, mas outros atributos como glicose nao (i.e. glicose == 0 -> dado faltante)
    :return dataset
    """
    for column in columns_list:
        dataset_training[column] = dataset_training[column].replace(missing_values_value, np.nan)
        dataset_test[column] = dataset_test[column].replace(missing_values_value, np.nan)

    return dataset_training, dataset_test

def imputation_with_mean(dataset_training: pd.DataFrame, dataset_test: pd.DataFrame, list_of_columns: []) -> (pd.DataFrame, pd.DataFrame):
    """
    Troca os valores NaN com a média de cada atributo na lista.
    """
    for column in list_of_columns:
        dataset_training[column] = dataset_training[column].astype(float)
        column_mean = dataset_training[column].mean()
        dataset_training[column] = dataset_training[column].replace(np.nan, column_mean)
        dataset_test[column] = dataset_test[column].astype(float)
        dataset_test[column] = dataset_test[column].replace(np.nan, column_mean)
    return dataset_training, dataset_test

def imputation_with_mode(dataset_training: pd.DataFrame, dataset_test: pd.DataFrame, list_of_columns: []) -> (pd.DataFrame, pd.DataFrame):
    """
    Troca os valores NaN com a moda de cada atributo na lista. Utilizado para atributos categóricos.
    """
    for column in dataset_training:
        if column in list_of_columns:
            column_mode = dataset_training[column].value_counts().idxmax()
            dataset_training[column] = dataset_training[column].replace(np.nan, column_mode)
            dataset_test[column] = dataset_test[column].replace(np.nan, column_mode)
    return dataset_training, dataset_test

def mean_imputation(df):
    """
    imputa os valores do dataset inteiro
    """
    imputer = Imputer(strategy='mean')
    df = imputer.fit_transform(df)
    return df





