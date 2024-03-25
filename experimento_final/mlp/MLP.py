import numpy as np
from tqdm import tqdm
import sys


class MLP:

    """
    Classe Multilayer Perceptron. Contem apenas uma camada escondida e utiliza apenas funcoes de ativacao
    sigmoide
    """

    def __init__(self, learning_rate=0.6, epochs=6000, min_gradient_value=1e-5):
        """

        :param learning_rate: taxa de aprendizado
        :param epochs: numero de peocas
        :param min_gradient_value: valor minimo do gradiente
        """

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.min_gradient_value = min_gradient_value

        # matrizes de pesos da camada escondida e camada de saida
        self.a_weight_matrix: np.array = None
        self.b_weight_matrix: np.array = None

        return

    def fit(self, X, y, hidden_layer_size=6):
        """

        :param X: conjunto de treinamento sem o vetor de classe
        :param y: vetor de classe
        :param hidden_layer_size: numero de neuronios na camada escondida
        :return:
        """

        X = self.add_bias_vector(X)

        num_instances, num_features = X.shape
        num_instances, num_classes = y.shape

        self.a_weight_matrix = np.random.randn(hidden_layer_size, num_features)/10
        self.b_weight_matrix = np.random.randn(num_classes, hidden_layer_size+1)/10

        best_a_w = self.a_weight_matrix
        best_b_w = self.b_weight_matrix

        error = sys.maxsize
        epoch_error_count = 0

        for epoch in tqdm(range(self.epochs)):
            epoch_error, hidden_gradient, output_gradient = self.train(X, y)

            hidden_gradient /= num_instances
            output_gradient /= num_instances

            # atualiza os pesos
            self.a_weight_matrix += self.learning_rate * hidden_gradient
            self.b_weight_matrix += self.learning_rate * output_gradient

            # verifica se o erro da epoca foi maior do que o da epoca passada. Caso o erro nao diminua com as epocas, modelo para
            if error > epoch_error:
                epoch_error_count = 0
                best_a_w = np.copy(self.a_weight_matrix)
                best_b_w = np.copy(self.b_weight_matrix)

                error = epoch_error

            else:
                epoch_error_count+=1

            if epoch_error_count > 50:
                self.a_weight_matrix = best_a_w
                self.b_weight_matrix = best_b_w
                print('Erro de validacao nao diminui apos 50 epocas.')
                break

            # calcula se o gradiente eh pequeno. Se for, para o treinamento
            grad_size = np.linalg.norm(hidden_gradient) + np.linalg.norm(output_gradient)
            if grad_size < self.min_gradient_value:
                print('break')
                break

    def add_bias_vector(self, X):
        """
        adiciona uma coluna de bias na matriz X
        :param X:
        :return:
        """


        num_instances = X.shape[0]
        bias_vector = np.ones((num_instances, 1))
        X = np.hstack((bias_vector, X))

        return X


    def train(self, X, y):
        """
        Treina o modelo
        :param X: dataset sem o vetor de classes
        :param y: vetor de classes
        :return: 
        """

        z, output = self.feedfoward(X)

        error = y - output
        hidden_gradient, output_gradient = self.backpropagate(error, z, output)

        # calcula o "delta" de atualizacao dos pesos da camada escondida
        hidden_gradient = np.dot(hidden_gradient.T, X)
        # calcula o "delta" de atualizacaodos pesos da camada de saida
        output_gradient = np.dot(output_gradient.T, z)

        # calcula o erro total
        error = np.sum(error*error)/len(error)

        return error, hidden_gradient, output_gradient


    def feedfoward(self, X):
        """
        calcula o feedfoward
        :param X: dataset sem o vetor de classe
        :return:
        """

        # calcula as saidas da camada escondida
        zin = np.inner(X, self.a_weight_matrix)
        z = self.sigmoid_function(zin)
        z = self.add_bias_vector(z)

        # calcula as saidas da camada de saida
        yin = np.inner(z, self.b_weight_matrix)
        output = self.sigmoid_function(yin)

        return z, output

    def backpropagate(self, error_vector, z_vector, output_vector):
        output_gradient = self.outer_layer_backpropagate(error_vector, output_vector)
        hidden_gradient = self.inner_layer_backpropagate(z_vector, output_gradient)

        return hidden_gradient, output_gradient

    def outer_layer_backpropagate(self, error, output):
        """
        calcula "delta" do backpropagation da camada de saida
        :param error: erro do classificador (saida esperada - saida dos neuronios de saida)
        :param output: saidas dos neuronios de saida
        :return:
        """

        gradient = error * output * (1-output)
        return gradient

    def inner_layer_backpropagate(self, z, outer_gradient):
        """
        calcula o "delta" do backpropagation da camada escondida
        :param z: saidas para os neuronios da camada escondida
        :param outer_gradient: "delta" da funcao acima
        :return:
        """
        # tem que tirar o bias
        outer_gradient = np.dot(outer_gradient, self.b_weight_matrix[:, 1:])

        gradient = z[:, 1:] * (1-z[:, 1:]) * (outer_gradient)
        return gradient

    def predict(self, X):
        """
        Faz a predicao
        :param X: dataset para ser classificado
        :return:
        """

        X = self.add_bias_vector(X)
        z, y_predict = self.feedfoward(X)
        y_predict = self.format_prediction(y_predict)
        return y_predict

    def format_prediction(self, y):
        """
        atribui 1 para a maior saída, e 0 caso contrario
        :param y: vetor de classes
        :return:
        """

        prediction = np.zeros_like(y)
        for i in range(prediction.shape[0]):
            prediction[i, y[i].argmax()] = 1
        return prediction

    def sigmoid_function(self, x):
        """funcao sigmoide"""
        return 1 / (1 + np.exp(-x))
