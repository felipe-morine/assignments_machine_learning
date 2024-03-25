import numpy as np
import pandas as pd
from dataset.preprocessing import MissingValuesHandler, Normalizer
import IrisSpecializedClassifier


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

    print(iris_data)

    # juntar dados, classes e label novamente
    iris_df = np.c_[iris_data, iris_setosa_vector]
    iris_df = np.c_[iris_df, iris_versicolor_vector]
    iris_df = np.c_[iris_df, iris_virginica_vector]

    iris_df = pd.DataFrame(iris_df, columns=(attrs+class_labels))

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
    iris_training_class_vector = iris_training[:, -3:]

    iris_test_data = iris_test[:, :-3]
    iris_test_class_vector = iris_test[:, -3:]


    weight_vector = IrisSpecializedClassifier.calculate_weight_vector(iris_training_data, iris_training_class_vector)

    predicted_results = IrisSpecializedClassifier.calculate_results(weight_vector, iris_test_data)

    real_predicted_results = IrisSpecializedClassifier.return_iris_classes(predicted_results)
    real_test_classes = IrisSpecializedClassifier.return_iris_classes(iris_test_class_vector)

    error_rate = IrisSpecializedClassifier.error_rate(real_predicted_results, real_test_classes)
    confusion_matrix = IrisSpecializedClassifier.conf_matrix(real_predicted_results, real_test_classes)

    print(error_rate)
    print(confusion_matrix)

    return


if __name__ == "__main__":
    # preprocessing()
    classifier_script()