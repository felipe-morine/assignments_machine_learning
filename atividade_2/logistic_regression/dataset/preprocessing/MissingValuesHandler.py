import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer


def set_numpy_missing_values(dataset: pd.DataFrame, missing_values_value, columns_list:[]):
    """
    Transforma os valores "nulos" do dataset em valores NaN definidos pela biblioteca numpy
    :param dataset
    :param missing_values_value: o valor do missing value no dataset. Ex: '?' para o dataset Hepatitis
    :param columns_list: lista de attributos em que a funcao deve ser aplicada. Ex: dataset Diabetes
        permite numero de gravidez  = 0, mas outros atributos como glicose nao (i.e. glicose == 0 -> dado faltante)
    :return dataset
    """
    for column in dataset:
        if column in columns_list:
            dataset[column] = dataset[column].replace(missing_values_value, np.nan)
    return dataset

def imputation_with_mean(dataset: pd.DataFrame, list_of_columns: []) -> pd.DataFrame:
    """
    Troca os valores NaN com a média de cada atributo na lista.
    """
    for column in dataset:
        if column in list_of_columns:
            dataset[column] = dataset[column].astype(float)
            column_mean = dataset[column].mean()
            dataset[column] = dataset[column].replace(np.nan, column_mean)
    return dataset

def imputation_with_mode(dataset: pd.DataFrame, list_of_columns: []) -> pd.DataFrame:
    """
    Troca os valores NaN com a moda de cada atributo na lista. Utilizado para atributos categóricos.
    """
    for column in dataset:
        if column in list_of_columns:
            column_mode = dataset[column].value_counts().idxmax()
            dataset[column] = dataset[column].replace(np.nan, column_mode)
    return dataset

def mean_imputation(df):
    """
    imputa os valores do dataset inteiro
    """
    imputer = Imputer(strategy='mean')
    df = imputer.fit_transform(df)
    return df





