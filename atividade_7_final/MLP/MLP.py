import pandas as pd
import numpy as np
from tqdm import tqdm


class MLP:

    """
    Classe do modelo MLP
    """

    def __init__(self, learning_rate=0.2, epochs=600, min_gradient_value=1e-5):
        """
        :param learning_rate: taxa de aprendizado
        :param epochs: numero de epocas
        :param min_gradient_value: tamanho minimo do gradiente
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.min_gradient_value = min_gradient_value

        self.a_weight_matrix: np.array = None
        self.a_bias_vector: np.array = None

        self.b_weight_matrix: np.array = None
        self.b_bias_vector: np.array = None

        return

    def fit(self, X, y, hidden_layer_size=4):
        """
        Treina o modelo
        :param X: conjunto de treinamento sem o vetor de classes
        :param y: vetor de classes
        :param hidden_layer_size: numero de neuronios na camada escondida
        :return:
        """
        num_instances, num_features = X.shape
        num_instances, num_classes = y.shape

        # matrizes de pesos e vetores de pesos para o bias
        self.a_weight_matrix = np.random.randn(hidden_layer_size, num_features)/10
        self.a_bias_vector = np.random.randn(hidden_layer_size)/10

        self.b_weight_matrix = np.random.randn(num_classes, hidden_layer_size)/10
        self.b_bias_vector = np.random.randn(num_classes)/10

        for epoch in tqdm(range(self.epochs)):
            a_gradient = np.zeros(self.a_weight_matrix.shape)
            a_bias_gradient = np.zeros(self.a_bias_vector.shape)

            b_gradient = np.zeros(self.b_weight_matrix.shape)
            b_bias_gradient = np.zeros(self.b_bias_vector.shape)


            for i in range(num_instances):
                # "delta" de atualizacao dos pesos para as matrizes de pesos da camada escondida, de saida e seus respectivos bias
                a_g, a_bias_g, b_g, b_bias_g  = self.train(X[i], y[i])

                # soma as atualizacoes encontradas para a instancia
                a_gradient += a_g
                a_bias_gradient += a_bias_g

                b_gradient += b_g
                b_bias_gradient += b_bias_g

            a_gradient /= num_instances
            a_bias_gradient /= num_instances

            b_gradient /= num_instances
            b_bias_gradient /= num_instances

            # atualizacao dos pesos
            self.a_weight_matrix += self.learning_rate * a_gradient
            self.a_bias_vector += self.learning_rate * a_bias_gradient

            self.b_weight_matrix += self.learning_rate * b_gradient
            self.b_bias_vector += self.learning_rate * b_bias_gradient

            # verifica tamanho do gradiente
            grad_size = np.linalg.norm(a_gradient) + np.linalg.norm(a_bias_gradient) + np.linalg.norm(b_gradient) + np.linalg.norm(b_bias_gradient)

            if grad_size < self.min_gradient_value:
                # print('break')
                break

    def train(self, x, y):
        """
        Treina o modelo
        :param x: instancia a ser usada no treino
        :param y: classe (dummificada)
        :return:
        """
        z_vector, output_vector = self.feedfoward(x)

        hidden_gradient, output_gradient = self.backpropagate(x, y, z_vector, output_vector)

        # calcula o "delta" das atualizacoes de pesos
        a_gradient = np.outer(hidden_gradient, x)
        a_bias_gradient = hidden_gradient

        b_gradient = np.outer(output_gradient, z_vector)
        b_bias_gradient = output_gradient

        return a_gradient, a_bias_gradient, b_gradient, b_bias_gradient

    def feedfoward(self, x):
        """
        alimenta a instancia x e calcula a saida da rede
        :param x:
        :return: saidas dos neuronios da camada escondida, saidas dos neuronios da camada de saida
        """
        zin_vector = np.inner(x, self.a_weight_matrix)
        zin_vector += self.a_bias_vector

        z_vector = self.softmax_function(zin_vector)

        yin_vector = np.inner(z_vector, self.b_weight_matrix)
        yin_vector += self.b_bias_vector

        output_vector = self.softmax_function(yin_vector)

        return z_vector, output_vector

    def backpropagate(self, x, y, z_vector, output_vector):
        output_gradient = self.outer_layer_backpropagate(x, y, output_vector)
        hidden_gradient = self.inner_layer_backpropagate(x, y, z_vector, output_vector)

        return hidden_gradient, output_gradient

    def outer_layer_backpropagate(self, x, y, output_vector):
        """
        calcula o "delta" do backpropagation para a camada de saida
        :param x: instancia de treinamento
        :param y: classe da instancia
        :param output_vector: saidas dos neuronios da camada de saida
        :return:
        """
        error_vector = y - output_vector

        gradient = -np.outer(output_vector, output_vector)
        gradient += np.diag(output_vector)
        gradient = np.inner(error_vector, gradient)

        return gradient

    def inner_layer_backpropagate(self, x, y, z_vector, output_vector):
        """
        calcula o "delta" do backpropagation para a camada escondida
        :param x: instancia de treinamento
        :param y: classe da instancia
        :param z_vector: saidas dos neuronios da camada escondida
        :param output_vector: saida dos neuronios da camada esconndida
        :return:
        """
        error_vector = y - output_vector

        f_grad = -np.outer(z_vector, z_vector)
        f_grad += np.diag(z_vector)
        yin_grad = np.inner(f_grad, self.b_weight_matrix)
        yin_grad_delta = np.inner(output_vector, yin_grad)

        gradient = np.add(yin_grad.T, -yin_grad_delta).T
        gradient = np.multiply(gradient, output_vector)
        gradient = np.inner(error_vector, gradient)

        return gradient

    def predict(self, X):
        """
        prediz as classes de instancias desconhecidas
        :param X: conjunto de teste
        :return:
        """
        num_instances, num_features = X.shape

        y_predict = np.zeros((num_instances, self.b_weight_matrix.shape[0]))

        for i in range(num_instances):
            z_vector, output = self.feedfoward(X[i])
            formatted_output = self.format_prediction(output)
            y_predict[i] = formatted_output

        return y_predict

    def format_prediction(self, y):
        """
        formata as saidas: a saida com maior valor = 1, o resto = 0
        :param y:
        :return:
        """
        prediction = np.zeros_like(y)
        prediction[y.argmax()] = 1

        return prediction

    def softmax_function(self, yin_vector):
        """
        calcula a softmax para todos os valores do vetor
        :param yin_vector: vetor das entradas para cada neuronio de uma camada
        :return: as saidas de cada neuronio dessa camada
        """

        # retirar o y_max ajuda a evitar o problema de overflow pelo calculo da expoente. Note que subtraimos em cima e embaixo = multiplicar por 1
        yin_max = yin_vector.max()
        exp_f = lambda x:  np.exp(x-yin_max)
        yin_vector2 = exp_f(yin_vector)

        sum_denominator = np.sum(yin_vector2)

        yin_vector2 /= sum_denominator

        return yin_vector2
