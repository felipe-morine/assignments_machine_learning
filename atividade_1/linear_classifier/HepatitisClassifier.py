import numpy as np
import pandas as pd
from dataset.preprocessing import MissingValuesHandler, Normalizer
import Classifier

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

def preprocessing():
    filename = 'dataset/hepatitis/hepatitis.csv'
    hepatitis_df = pd.read_csv(filename)


    # algumas colunas com valor 0 na verdade sao missing values - transformar para NaN
    missing_values_columns_mean = ['bilirubin', 'alk_phosphate', 'sgot', 'albumin', 'protime']
    missing_values_columns_mode = ['steroid', 'fatigue', 'malaise', 'anorexia', 'liver_big', 'liver_firm',
                                   'spleen_palpable',
                                   'spiders', 'ascites', 'varices']
    missing_values_value = '?'
    hepatitis_df = MissingValuesHandler.set_numpy_missing_values(hepatitis_df, missing_values_value, (missing_values_columns_mode + missing_values_columns_mean))

    # label dos atributos
    hepatitis_header = ['age', 'sex', 'steroid', 'antivirals', 'fatigue', 'malaise', 'anorexia', 'liver_big', 'liver_firm', 'spleen_palpable',
         'spiders', 'ascites', 'varices', 'bilirubin', 'alk_phosphate', 'sgot', 'albumin', 'protime', 'histology', 'class']

    # imputar colunas de valores continuos com media
    hepatitis_df = MissingValuesHandler.imputation_with_mean(hepatitis_df, missing_values_columns_mean)
    # imputar coluna de valores categoricos com moda
    hepatitis_df = MissingValuesHandler.imputation_with_mode(hepatitis_df, missing_values_columns_mode)

    # tranforma valores categoricos 1-2 para 0-1, incluindo a classe
    binary_attributes = ['class', 'sex', 'steroid', 'antivirals', 'fatigue', 'malaise', 'anorexia', 'liver_big',
                         'liver_firm',
                         'spleen_palpable',
                         'spiders', 'ascites', 'varices', 'histology']
    hepatitis_df = format_binary_attributes(hepatitis_df, binary_attributes)

    # separar covariantes do vetor de classe
    hepatitis_df = hepatitis_df.values
    hepatitis_data = hepatitis_df[:, 1:]
    hepatitis_class_vector = hepatitis_df[:, 0]

    # normalizar
    hepatitis_data = Normalizer.standardize(hepatitis_data)

    # juntar dados, classes e label novamente
    hepatitis_df = np.c_[hepatitis_data, hepatitis_class_vector]
    hepatitis_df = pd.DataFrame(hepatitis_df, columns=hepatitis_header)

    # salvar arquivo
    preprocessed_filename = 'dataset/hepatitis/hepatitis_preprocessed.csv'
    hepatitis_df.to_csv(preprocessed_filename, index=False)

    return

def classifier_script():
    preprocessed_filename = 'dataset/hepatitis/hepatitis_preprocessed.csv'
    hepatitis_df = pd.read_csv(preprocessed_filename)

    hepatitis_df = hepatitis_df.values

    # adiciona coluna de vies w0 (valores =  1)
    bias_vector = np.ones(len(hepatitis_df))
    hepatitis_df = np.c_[bias_vector, hepatitis_df]

    # separa em treinamento, teste, 2/3 - 1/3
    hepatitis_training, hepatitis_test = Classifier.split_training_test_subsets(hepatitis_df)

    # separar covariantes do vetor de classe
    hepatitis_training_data = hepatitis_training[:, :-1]
    # vetor de classes ja transposto
    hepatitis_training_class_vector = hepatitis_training[:, -1]

    hepatitis_test_data = hepatitis_test[:, :-1]
    # vetor de classes ja transposto
    hepatitis_test_class_vector = hepatitis_test[:, -1]

    weight_vector = Classifier.calculate_weight_vector(hepatitis_training_data, hepatitis_training_class_vector)

    predicted_results = Classifier.calculate_results(weight_vector, hepatitis_test_data)

    error_rate = Classifier.error_rate(predicted_results, hepatitis_test_class_vector)
    confusion_matrix = Classifier.conf_matrix(predicted_results, hepatitis_test_class_vector)

    print(error_rate)
    print(confusion_matrix)

if __name__ == "__main__":
    # preprocessing()
    classifier_script()