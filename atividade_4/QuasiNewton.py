from Function import Function
import numpy as np
import sys
import BisectionMethod


def quasi_newton_method():
    f = Function()

    max_iterations = 1000
    epsilon = 10 ** -4  # 10e-5

    #  INPUT ("CHUTE") INICIAL, MUDAR SE NECESSARIO
    vector_x = np.array(
        [5.0, 6.0]
    )

    alpha = 0.1
    iteration = 1
    current_grad = -f.gradient(vector_x)

    h = np.identity(len(vector_x))
    d_norm = sys.maxsize


    while (iteration<max_iterations) and (d_norm >= epsilon):
        if (iteration-1) % len(vector_x) == 0:
            d = current_grad
            h = np.identity(len(vector_x))
        else:
            d = np.inner(h, current_grad)

        # COMENTAR CASO NAO SE USE BISSECAO
        alpha = BisectionMethod.calculate_alpha(f, d, vector_x)
        p = alpha * d

        vector_x += p

        previous_grad = current_grad
        current_grad = -f.gradient(vector_x)

        q = current_grad - previous_grad

        # DEFINIR A ESTRATEGIA A SER UTILIZADA
        # h = david_fletcher_powell_matrix(h, p, q)
        h = broyden_fletcher_goldfarb_shanno_matrix(h, p, q)

        d_norm = np.linalg.norm(d)
        print('Iteracao:', iteration, 'direcao:', d, 'x:', vector_x, 'norma direcao:', d_norm, 'alpha:', alpha)
        iteration+=1


def __calculate_direction(iteration, g_current, g_previous, d_previous, n_dimensions, beta_function):
    if iteration % n_dimensions != 0:
        beta = beta_function(g_current, g_previous)
        direction = beta * d_previous
        direction += g_previous
    else:
        direction = g_current
    return direction

def david_fletcher_powell_matrix(h_current, p, q):
    numerator_1 = np.outer(p, p)
    denominator_1 = np.inner(p, q)

    numerator_2 = np.outer(q, q)
    numerator_2 = np.inner(h_current, numerator_2)
    numerator_2 = np.inner(numerator_2, h_current)

    denominator_2 = np.inner(q, h_current)
    denominator_2 = np.inner(denominator_2, q)

    d1 = numerator_1/denominator_1
    d2 = numerator_2/denominator_2

    new_h = h_current + d1 - d2

    return new_h

def broyden_fletcher_goldfarb_shanno_matrix(h_current, p, q):
    numerator_1 = np.outer(p, p)
    denominator_1 = np.inner(p, q)

    numerator_extra = np.inner(q, h_current)
    numerator_extra = np.inner(numerator_extra, q)
    denominator_extra = np.inner(p, q)

    numerator_2_1 = np.inner(h_current, q)
    numerator_2_1 = np.outer(numerator_2_1, p)

    numerator_2_2 = np.outer(p, q)
    numerator_2_2 = np.inner(numerator_2_2, h_current)

    numerator_2 = numerator_2_1 + numerator_2_2
    denominator_2 = np.inner(p, q)

    d1 = numerator_1 / denominator_1
    d2 = numerator_2 / denominator_2
    d3 = numerator_extra / denominator_extra

    d3 = 1 + d3

    d1 = d1 * d3

    new_h = h_current + d1 - d2

    return new_h


if __name__ == "__main__":
    quasi_newton_method()

