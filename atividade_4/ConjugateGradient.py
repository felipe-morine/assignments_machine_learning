from Function import Function
import numpy as np
import sys
import BisectionMethod


def conjugate_method():
    f = Function()

    max_iterations = 1000
    epsilon = 10 ** -4  # 10e-5

    #  INPUT ("CHUTE") INICIAL, MUDAR SE NECESSARIO
    vector_x = np.array(
        [5.0, 5.0]
    )

    alpha = 0.1

    iteration = 1
    d_previous = -f.gradient(vector_x)
    g_previous = np.copy(d_previous)
    d_norm = sys.maxsize

    while (iteration<max_iterations) and (d_norm >= epsilon):
        g_current = -f.gradient(vector_x)
        if (iteration - 1) % len(vector_x) == 0:
            d = g_current
        else:
            # DEFINIR A ESTRATEGIA A SER UTILIZADA
            # d = __calculate_direction(g_current, g_previous, d, beta_function=polak_ribiere_beta)
            d = __calculate_direction(g_current, g_previous, d, beta_function=fletcher_reeves_beta)

        # COMENTAR CASO NAO SE USE BISSECAO
        alpha = BisectionMethod.calculate_alpha(f, d, vector_x)

        vector_x += alpha * d
        g_previous = g_current
        d_norm = np.linalg.norm(d)
        print('Iteracao:', iteration, 'direcao:', d, 'x:', vector_x, 'norma direcao:', d_norm, 'alpha:', alpha)
        iteration+=1


def __calculate_direction(g_current, g_previous, d_previous, beta_function):
    beta = beta_function(g_current, g_previous)
    direction = beta * d_previous
    direction += g_current

    return direction

def polak_ribiere_beta(g_current, g_previous):
    beta = (( (np.inner(g_current, (g_current-g_previous)))  )  / (np.inner(g_previous, g_previous)))
    return beta

def fletcher_reeves_beta(g_current, g_previous):
    beta = ( np.inner(g_current, g_current) / np.inner(g_previous, g_previous) )
    return beta


if __name__ == "__main__":
    conjugate_method()

