import numpy as np
from collections import Counter


class KNN:

    """
    Classe para modelos KNN
    """

    def __init__(self, k):
        """

        :param k: numero de vizinhos a serem considerados
        """
        self.k = k
        self.X: None
        self.y: None

    def fit(self, X, y):
        """
        Treinamento (apenas grava o ocnjunto de treinamento e o vetor de classes
        :param X: conjunto de treinamento
        :param y: vetor de classes
        :return:
        """
        self.X = X
        self.y = y

        return

    def predict(self, X):
        """
        Faz a predicao das classes para o conjunto de teste
        :param X: conjunto de teste
        :return: vetor de classes preditas
        """
        n_instances, n_features = X.shape

        y_predict = []

        for i in range(n_instances):
            distance_list = self.calculate_distance_vector(X[i])
            support_instances_class_list = self.get_support_instances_class_list(distance_list)
            predicted_class = self.get_class(support_instances_class_list)
            y_predict.append(predicted_class)
        return y_predict


    def calculate_distance_vector(self, x):
        """
        Calcula distancia de x para as instancias do conjunto de treinamento
        :param x: instancia a ser predita
        :return: vetor de distancia
        """

        n_instances, n_features = self.X.shape
        distance_map = []
        for i in range(n_instances):
            instance_distance = np.linalg.norm(x - self.X[i])
            # adiciona tanto o indice da instancia quanto a distancia
            distance_map.append((i, instance_distance))
        dtype = [('index', int), ('distance', float)]
        distance_vector = np.array(distance_map, dtype=dtype)
        # ordena pela distancia
        distance_vector = np.sort(distance_vector, order='distance')

        return distance_vector

    def get_support_instances_class_list(self, distance_vector):
        """
        retorna a classes das k instancias mais proximas
        :param distance_vector: vetor de distancia calculado na funcao acima
        :return: vetor de classes mais proximas
        """
        support_instances_class_list = []

        for i in range(self.k):
            support_instances_class_list.append(self.y[distance_vector[i][0]])
        return support_instances_class_list

    def get_class(self, support_instances_class_list):
        """
        votacao. Caso de empate, retorna para k-1
        :param support_instances_class_list: vetor de classes mais proximas, calculado na funcao acima
        :return: a classe mais votada
        """
        while True:
            counter = Counter(support_instances_class_list)
            most_common_list = counter.most_common(1)
            if len(most_common_list) > 1:
                support_instances_class_list = support_instances_class_list[:len(support_instances_class_list)-1]
            else:
                predicted_class = most_common_list[0][0]
                break
        return predicted_class
