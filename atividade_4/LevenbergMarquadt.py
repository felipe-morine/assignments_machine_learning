from Function2 import Function2
import numpy as np
import BisectionMethod

""""(2x1 − x2)^2 + (3x1 − x3)^2 + (3x2 − 2x3)^2"""
"""min = x2 = 2x1 or x3 = 3x1"""


def gradient_descent():
    f = Function2()

    max_iterations = 1000
    epsilon = 10 ** -4  # 10e-5

    #  INPUT ("CHUTE") INICIAL, MUDAR SE NECESSARIO
    vector_x = np.array(
        [5.0, 5.0, 5.0]
    )
    alpha = 0.1
    iteration = 0
    d_norm = np.linalg.norm(vector_x)


    while (iteration<max_iterations) and (d_norm >= epsilon):
        d = __calculate_direction(f, vector_x)

        # COMENTAR CASO NAO SE USE BISSECAO
        alpha = BisectionMethod.calculate_alpha(f, d, vector_x)

        vector_x += alpha*d
        d_norm = np.linalg.norm(d)
        print('Iteracao:', iteration, 'direcao:', d, 'x:', vector_x, 'norma direcao:', d_norm, 'alpha:', alpha)
        iteration+=1

def levenberg_marquadt():
    f = Function2()

    max_iterations = 1000
    epsilon = 10 ** -4  # 10e-5

    #  INPUT ("CHUTE") INICIAL, MUDAR SE NECESSARIO
    vector_x = np.array(
        [5.0, 5.0, 5.0]
    )

    alpha = 0.1
    iteration = 0
    d_norm = np.linalg.norm(vector_x)

    mi = np.abs(np.random.randn(1)[0]/10)
    matrix_adicional = np.identity(len(vector_x)) * mi

    while (iteration < max_iterations) and (d_norm >= epsilon):
        r_vector = f.r_vector(vector_x)
        matriz_gradiente = f.residual_gradient_matrix(vector_x)
        invert = np.inner(matriz_gradiente, matriz_gradiente)
        invert += matrix_adicional
        inverse = np.linalg.inv(invert)

        ajuste = np.inner(inverse, matriz_gradiente.T)
        ajuste = np.inner(ajuste, r_vector)

        # COMENTAR CASO NAO SE USE BISSECAO
        alpha = BisectionMethod.calculate_alpha(f, ajuste, vector_x)

        vector_x += alpha * ajuste
        d_norm = np.linalg.norm(ajuste)
        print('Iteracao:', iteration, 'ajuste:', ajuste, 'x:', vector_x, 'norma direcao:', d_norm, 'alpha:', alpha)
        iteration += 1

def __calculate_direction(func: Function2, vector_x):
    gradient = func.gradient(vector_x)

    return -(gradient)


if __name__ == "__main__":
    """MUDAR A FUNCAO DESEJADA"""

    levenberg_marquadt()
    # gradient_descent()