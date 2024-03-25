import numpy as np


class Function:

    """5x1^2 + x2^2 + 4*x1*x2 - 14*x1 - 6*x2 + 20"""

    @staticmethod
    def func(input_vector: np.array):
        """
        Calcula a saida da funcao para um determinado input
        :param input_vector: vetor input. Deve ser um array de apenas duas posicoes
        :return:
        """
        Function.is_correct_vector_size(input_vector)
        return ((5*(input_vector[0]**2)) + (input_vector[1]**2) + (4*input_vector[0]*input_vector[1]) + (-14*input_vector[0]) + (-4*input_vector[1]) + 20 )

    @staticmethod
    def gradient(input_vector: np.array):
        """
        retorna o vetor gradiente da funcao para um input
        :param input_vector: vetor input. Deve ser um array de apenas duas posicoes
        :return:
        """
        Function.is_correct_vector_size(input_vector)
        grad = np.array(
                [(10*input_vector[0]) + (4*input_vector[1]) - 14,
                (2*input_vector[1]) + (4*input_vector[0]) - 6
                 ]
        )
        return grad

    @staticmethod
    def hessian(input_vector: np.array):
        """
        retorna a matriz hessiana da funcao para um input. Neste caso, o input nao influencia nos valores da hessiana
        :param input_vector: vetor input. Deve ser um array de apenas duas posicoes
        :return:
        """
        Function.is_correct_vector_size(input_vector)
        h = np.array(
            [
                [10, 4],
                [4, 2]
            ]
        )
        return h

    @staticmethod
    def is_correct_vector_size(vector: np.array):
        """
        verifica se o input tem tamanho 2
        :param vector:
        :return:
        """
        if len(vector) != 2:
            raise Exception('Dimensao errada de x_vetor')
        return