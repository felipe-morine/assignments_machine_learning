import numpy as np
import pandas as pd
from dataset.preprocessing import MissingValuesHandler, Normalizer
import Classifier
import One_vs_All


def format_class_values(dataset: pd.DataFrame,  columns_list: []):
    """
    Formata valores da classe de (0, 1) para (-1, 1).
    :param dataset:
    :return:
    """
    for column in columns_list:
        dataset[column] = pd.to_numeric(dataset[column])
        dataset[column] = dataset[column].replace(0, -1)
    return dataset

def preprocessing():
    filename = 'dataset/iris/iris.csv'
    iris_df = pd.read_csv(filename)

    # labels dos atributos e das classes
    attrs = ['sepal_l', 'sepal_w', 'petal_l', 'petal_w']
    class_labels = ['setosa', 'versicolor', 'virginica']

    # coloca as classes em variaveis dummy
    iris_df = pd.get_dummies(iris_df)

    # separar covariantes do vetor de classe
    iris_df = iris_df.values
    iris_data = iris_df[:, :-3]
    iris_setosa_vector = iris_df[:, -3]
    iris_versicolor_vector = iris_df[:, -2]
    iris_virginica_vector = iris_df[:, -1]

    # normalizar
    iris_data = Normalizer.standardize(iris_data)

    # juntar dados, classes e label novamente
    iris_df = np.c_[iris_data, iris_setosa_vector]
    iris_df = np.c_[iris_df, iris_versicolor_vector]
    iris_df = np.c_[iris_df, iris_virginica_vector]

    iris_df = pd.DataFrame(iris_df, columns=(attrs+class_labels))

    iris_df = format_class_values(iris_df, ['setosa', 'versicolor', 'virginica'])

    # salvar arquivo
    preprocessed_filename = 'dataset/iris/iris_preprocessed.csv'
    iris_df.to_csv(preprocessed_filename, index=False)

    return

def classifier_script():
    # classes ja separads em teste e treinamento
    training_filename = 'dataset/iris/iris_training.csv'
    iris_training = pd.read_csv(training_filename)

    test_filename = 'dataset/iris/iris_test.csv'
    iris_test = pd.read_csv(test_filename)

    iris_training = iris_training.values
    iris_test = iris_test.values

    # adiciona coluna de vies w0 (valores =  1)
    bias_vector = np.ones(len(iris_training))
    iris_training = np.c_[bias_vector, iris_training]
    bias_vector = np.ones(len(iris_test))
    iris_test = np.c_[bias_vector, iris_test]

    # separar covariantes do vetor de classe
    iris_training_data = iris_training[:, :-3]
    iris_training_setosa_vector = iris_training[:, -3]
    iris_training_versicolor_vector = iris_training[:, -2]
    iris_training_virginica_vector = iris_training[:, -1]

    iris_test_data = iris_test[:, :-3]
    # iris_test_setosa_vector = iris_test[:, -3]
    # iris_test_versicolor_vector = iris_test[:, -2]
    # iris_test_virginica_vector = iris_test[:, -1]
    iris_test_class_vector = iris_test[:, -3:]

    # calcula os 3 modelos (para one vs. all) OS PARAMETROS LIVRES DEVEM SER MUDADOS AQUI
    setosa_weight_vector = Classifier.calculate_weight_vector(iris_training_data, iris_training_setosa_vector)
    versicolor_weight_vector = Classifier.calculate_weight_vector(iris_training_data, iris_training_versicolor_vector)
    virginica_weight_vector = Classifier.calculate_weight_vector(iris_training_data, iris_training_virginica_vector)

    # calcula os resultados do  one vs all
    setosa_predicted_results = Classifier.calculate_results(setosa_weight_vector, iris_test_data)
    versicolor_predicted_results = Classifier.calculate_results(versicolor_weight_vector, iris_test_data)
    virginica_predicted_results = Classifier.calculate_results(virginica_weight_vector, iris_test_data)

    real_predicted_results = One_vs_All.calculate_iris_results(setosa_predicted_results, versicolor_predicted_results, virginica_predicted_results)
    real_test_classes = One_vs_All.return_iris_classes(iris_test_class_vector)

    error_rate = Classifier.error_rate(real_predicted_results, real_test_classes)
    confusion_matrix = Classifier.conf_matrix(real_predicted_results, real_test_classes)

    print(error_rate)
    print(confusion_matrix)

    return


if __name__ == "__main__":
    # preprocessing()
    classifier_script()