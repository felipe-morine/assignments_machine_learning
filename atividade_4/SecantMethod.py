from Function import Function
import numpy as np
import sys
import BisectionMethod


def secant_method():
    f = Function()

    max_iterations = 1000
    epsilon = 10 ** -4  # 10e-5

    #  INPUT ("CHUTE") INICIAL, MUDAR SE NECESSARIO
    vector_x = np.array(
        [5.0, 5.0]
    )

    alpha = 0.1
    iteration = 1
    current_grad = -f.gradient(vector_x)

    d_norm = sys.maxsize

    s = vector_x
    q = current_grad


    while (iteration<max_iterations) and (d_norm >= epsilon):
        if (iteration-1) % len(vector_x) == 0:
            d = current_grad
        else:
            A, B = secant_values(current_grad, q, s)
            # MODIFICACAO DOS SLIDES
            d = current_grad - (A*s) - (B*q)
            # d = -current_grad + (A * s) + (B * q) nao converge

        # COMENTAR CASO NAO SE USE BISSECAO
        alpha = BisectionMethod.calculate_alpha(f, d, vector_x)

        s = alpha * d
        vector_x += s

        previous_grad = current_grad
        current_grad = -f.gradient(vector_x)

        q = current_grad - previous_grad

        d_norm = np.linalg.norm(d)
        print('Iteracao:', iteration, 'direcao:', d, 'x:', vector_x, 'norma direcao:', d_norm, 'alpha:', alpha)
        iteration+=1

def secant_values(g, q, s):
    first_term = np.inner(q, q)/np.inner(s, q)
    first_term += 1
    second_term = np.inner(s, g)/np.inner(s, q)
    third_term = np.inner(q, g)/np.inner(s, q)

    A = -(first_term*second_term)
    A += third_term

    B = np.inner(s, g)/np.inner(s, q)
    return A, B


if __name__ == "__main__":
    secant_method()

