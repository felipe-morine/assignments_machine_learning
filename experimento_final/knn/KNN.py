import numpy as np
from collections import Counter
from scipy.spatial.distance import cdist
from tqdm import tqdm


class KNN:
    """
    Classe do modelo KNN.
    """


    def __init__(self, k, max_support_instances=5):
        """

        :param k: numero de vizinhos mais proximos a serem considerados
        :param max_support_instances: numero maximo de instancias suporte a serem gravados na matriz

        self.support_instances_matrix eh a matriz que contem as classes das MAX_SUPPORT_INSTANCES mais proximas do conjunto de teste
        Por exemplo, para Iris, a primeira linha indica a classe das três instancias mais proximas da primeira
        instancia do conjunto de teste, que poderia ser (setosa, setosa, virginica) para um maximo de 3 instancias suporte.

        Se testes com k maiores que 5 quiserem ser realizados, eh necessario aumentar MAX_SUPPORT_INSTANCES

        A ideia eh que a mesma matriz de suporte com MAX_SUPPORT_INSTANCES = D possa ser utilizada para prever modelos
        com qualquer k <= D. Isso acelera o treinamento.

        Eh possivel que aumentar demais esse tamanho possa dar um estouro de memoria
        """

        self.k = k
        self.X = None
        self.y = None

        self.support_instances_matrix = None
        # maximo de instancias suporte a serem gravados.
        self.MAX_SUPPORT_INSTANCES = max_support_instances



    def fit(self, X, y):

        """
        "Treina o modelo". Apenas salva X e y.

        :param X: Dataset sem o vetor de classes
        :param y: vetor de classes.
        :return:
        """

        self.X = X
        self.y = y

        return

    def predict(self, X, support_instances_matrix: np.array=None):
        """
        Faz a predição do modelo

        :param X: Conjunto de teste, sem o vetor de classe.
        :param support_instances_matrix: matriz com as instâncias mais próximas.
        :return: o vetor de classes preditas.
        """

        n_instances, n_features = X.shape

        y_predict = np.zeros(n_instances)

        # se a matriz de classes mais próximas já existe
        if support_instances_matrix is not None:
            self.support_instances_matrix = support_instances_matrix

        # senao calcula matriz de classes mais proximas
        if self.support_instances_matrix is None:
            self.support_instances_matrix = np.zeros((n_instances, self.MAX_SUPPORT_INSTANCES))

            for i in tqdm(range(n_instances)):
                x_reshape = X[i].reshape((1, n_features))

                # calcula a distancia. Similar a tirar a norma da diferenca, mas mais rapido.
                distance_vector = cdist(self.X, x_reshape).flatten()
                support_instances_list = distance_vector.argsort()[:self.MAX_SUPPORT_INSTANCES]

                # dado o índice das classes mais proximas, pega a classe dessas instancias e grava na matriz de instancias suporte
                for k in range(support_instances_list.shape[0]):
                    support_instances_list[k] = self.y[support_instances_list[k]]
                self.support_instances_matrix[i] = support_instances_list

        # prediz a classe de cada x
        for i in range(n_instances):
            predicted_class = self.get_class(self.support_instances_matrix[i])
            y_predict[i] = predicted_class
        return y_predict


    def get_class(self, support_instances_class_list):
        """
        Realiza a votacao de acordo com um k
        :param support_instances_class_list:
        :return:
        """

        support_instances_class_list = support_instances_class_list[:self.k]
        while True:
            counter = Counter(support_instances_class_list)
            most_common_list = counter.most_common(1)
            if len(most_common_list) > 1:
                support_instances_class_list = support_instances_class_list[:len(support_instances_class_list)-1]
            else:
                predicted_class = most_common_list[0][0]
                break
        return predicted_class
