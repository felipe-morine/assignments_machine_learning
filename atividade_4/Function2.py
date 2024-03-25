import numpy as np


class Function2:
    """"(2x1 − x2)^2 + (3x1 − x3)^2 + (3x2 − 2x3)^2"""

    @staticmethod
    def func(input_vector: np.array):
        """
        Calcula a saida da funcao para um determinado input
        :param input_vector: vetor input. Deve ser um array de apenas tres posicoes
        :return:
        """
        Function2.is_correct_vector_size(input_vector)
        x1 = input_vector[0]
        x2 = input_vector[1]
        x3 = input_vector[2]
        return (np.power((2*x1 - x2), 2) + np.power((3*x1 - x3), 2) + np.power((3*x2 - 2*x3), 2))

    @staticmethod
    def r_vector(input_vector: np.array):
        """
        Calcula as saidas de cada "residuo"
        :param input_vector: vetor input. Deve ser um array de apenas tres posicoes
        :return:
        """
        Function2.is_correct_vector_size(input_vector)
        x1 = input_vector[0]
        x2 = input_vector[1]
        x3 = input_vector[2]

        r_vector = np.array(
            [(2 * x1 - x2),
             (3 * x1 - x3),
             (3 * x2 - 2 * x3)]
        )
        return r_vector.astype(np.float64)


    @staticmethod
    def gradient(input_vector: np.array):
        """
        Calcula o vetor gradiente da funcao para um determinado input
        :param input_vector: vetor input. Deve ser um array de apenas tres posicoes
        :return:
        """
        Function2.is_correct_vector_size(input_vector)
        x1 = input_vector[0]
        x2 = input_vector[1]
        x3 = input_vector[2]

        gradient = np.array(
            [(26*x1 -4*x2 -6*x3),
             (-4*x1 +20*x2 -12*x3),
             (-6*x1 -12*x2 +10*x3)]
        )

        return gradient.astype(np.float64)

    @staticmethod
    def residual_gradient_matrix(vector_x: np.array):
        """
        Calcula os vetores gradiente dos residuos para um determinado input
        :param input_vector: vetor input. Deve ser um array de apenas tres posicoes. Input nao influencia na matriz,
        apenas eh verificado se seu tamanho eh adequado
        :return:
        """
        gradient_matrix = np.array(
            [[2, -1, 0],
             [3, 0, -1],
             [0, 3, 2]]
        )

        return gradient_matrix.astype(np.float64)

    @staticmethod
    def is_correct_vector_size(vector: np.array):
        """
        verifica se o input tem tamanho 3
        :param vector:
        :return:
        """
        if len(vector) != 3:
            raise Exception('Dimensao errada de x_vetor')
        return