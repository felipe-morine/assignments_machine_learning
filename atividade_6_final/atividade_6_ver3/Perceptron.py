import numpy as np


class Perceptron:

    """
    classe para criacao do modelo Perceptron
    """

    def __init__(self, learning_rate=0.2, epochs=600, min_gradient_value=1e-6):
        """
        :param learning_rate: taxa de aprendizado
        :param epochs: numero de epocas
        :param min_gradient_value: minimo valor para o gradiente
        """

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.min_gradient_value = min_gradient_value

        # matriz de pesos
        self.weight_matrix: np.array = None

        return

    def fit(self, X, y):
        """
        treina o modelo
        :param X: conjunto de treinamento se o vetor de classes
        :param y: vetor de classes
        :return:
        """
        X = self.add_bias_vector(X)
        num_instances, num_features = X.shape
        num_instances, num_classes = y.shape

        self.weight_matrix = np.random.randn(num_classes, num_features)/10


        for epoch in range(self.epochs):
            gradient = np.zeros(self.weight_matrix.shape)

            for i in range(num_instances):
                delta = self.train(X[i], y[i])
                gradient += np.outer(delta, X[i])

            self.weight_matrix += self.learning_rate * gradient
            grad_size = np.linalg.norm(gradient)

            if grad_size < self.min_gradient_value:
                break


    def add_bias_vector(self, X):
        """
        adiciona uma coluna de bias para a matriz
        :param X:
        :return:
        """
        num_instances = X.shape[0]
        bias_vector = np.ones((num_instances, 1))
        X = np.hstack((bias_vector, X))

        return X

    def train(self, x, y):
        output_vector = self.feedfoward(x)
        gradient = self.backpropagate(x, y, output_vector)

        return gradient

    def feedfoward(self, x):
        """
        calcula o feedfoward
        :param x: instancia a ser alimentada
        :return:
        """

        yin_vector = np.zeros(self.weight_matrix.shape[0])
        output_vector = np.zeros(self.weight_matrix.shape[0])

        for i in range(len(yin_vector)):
            yin = np.inner(x, self.weight_matrix[i])
            yin_vector[i] = yin

        for i in range(len(output_vector)):
            output_vector[i] = self.softmax_function(i, yin_vector)

        return output_vector

    def backpropagate(self, x, y, output_vector):
        """
        backpropagation
        :param x: instancia a ser treinada
        :param y: classe (dummificada)
        :param output_vector: saida dos neuronios de saida apos o feedfowarding
        :return:
        """
        error_vector = y - output_vector

        gradient = np.zeros(self.weight_matrix.shape[0])
        for i in range(len(output_vector)):
            backpropag_delta = 0
            for j in range(len(output_vector)):
                if i == j:
                    softmax_grad = output_vector[i]*(1-output_vector[i])
                else:
                    softmax_grad = -(output_vector[i] * output_vector[j])
                backpropag_delta += error_vector[j]*softmax_grad
            gradient[i] = backpropag_delta

        return gradient

    def predict(self, X):
        """
        retorna a classe predita para instancias desconhecidas
        :param X: conjunto de teste, sem o vetor de classes
        :return: classes preditas
        """
        X = self.add_bias_vector(X)
        num_instances, num_features = X.shape

        y_predict = np.zeros((num_instances, self.weight_matrix.shape[0]))

        for i in range(num_instances):
            output = self.feedfoward(X[i])
            formatted_output = self.format_prediction(output)
            y_predict[i] = formatted_output

        return y_predict


    def format_prediction(self, y):
        """
        define a saida do neuronio com maior valor como 1, e 0 para os demais
        :param y:
        :return:
        """
        prediction = np.zeros_like(y)
        prediction[y.argmax()] = 1

        return prediction


    def softmax_function(self, yin_index, yin_vector):
        """
        calcula a saida para a funcao softmax
        :param yin_index: o indice do neuronio
        :param yin_vector: a camada de neuronios
        :return:
        """
        numerator = np.exp(yin_vector[yin_index])
        denominator = 0

        for i in range(len(yin_vector)):
            denominator += np.exp(yin_vector[i])

        return numerator/denominator
