import numpy as np
import pandas as pd
from dataset.preprocessing import MissingValuesHandler, Normalizer
import Classifier


def preprocessing():
    filename = 'dataset/diabetes/diabetes.csv'
    diabetes_df = pd.read_csv(filename)


    # algumas colunas com valor 0 na verdade sao missing values - transformar para NaN
    missing_values_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction']
    missing_values_value = 0
    diabetes_df = MissingValuesHandler.set_numpy_missing_values(diabetes_df, missing_values_value, missing_values_columns)

    # label dos atributos
    diabetes_header = list(diabetes_df.columns)

    # separar covariantes do vetor de classe
    diabetes_df = diabetes_df.values
    diabetes_data = diabetes_df[:, :-1]
    diabetes_class_vector = diabetes_df[:, -1]

    # imputar valores faltantes com media
    diabetes_data = MissingValuesHandler.mean_imputation(diabetes_data)

    # normalizar
    diabetes_data = Normalizer.standardize(diabetes_data)

    # juntar dados, classes e label novamente
    diabetes_df = np.c_[diabetes_data, diabetes_class_vector]
    diabetes_df = pd.DataFrame(diabetes_df, columns=diabetes_header)

    # salvar arquivo
    preprocessed_filename = 'dataset/diabetes/diabetes_preprocessed.csv'
    diabetes_df.to_csv(preprocessed_filename, index=False)

    return

def classifier_script():
    preprocessed_filename = 'dataset/diabetes/diabetes_preprocessed.csv'
    diabetes_df = pd.read_csv(preprocessed_filename)

    diabetes_df = diabetes_df.values

    # adiciona coluna de vies w0 (valores =  1)
    bias_vector = np.ones(len(diabetes_df))
    diabetes_df = np.c_[bias_vector, diabetes_df]

    # separa em treinamento, teste, 2/3 - 1/3
    diabetes_training, diabetes_test = Classifier.split_training_test_subsets(diabetes_df)

    # separar covariantes do vetor de classe
    diabetes_training_data = diabetes_training[:, :-1]
    # vetor de classes ja transposto
    diabetes_training_class_vector = diabetes_training[:, -1]

    diabetes_test_data = diabetes_test[:, :-1]
    # vetor de classes ja transposto
    diabetes_test_class_vector = diabetes_test[:, -1]



    weight_vector = Classifier.calculate_weight_vector(diabetes_training_data, diabetes_training_class_vector)

    predicted_results = Classifier.calculate_results(weight_vector, diabetes_test_data)

    error_rate = Classifier.error_rate(predicted_results, diabetes_test_class_vector)
    confusion_matrix = Classifier.conf_matrix(predicted_results, diabetes_test_class_vector)

    print(error_rate)
    print(confusion_matrix)


if __name__ == "__main__":
    # preprocessing()
    classifier_script()