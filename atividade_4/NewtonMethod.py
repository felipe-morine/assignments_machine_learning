from Function import Function
import numpy as np
import BisectionMethod


def newton_method():
    f = Function()

    max_iterations = 1000
    epsilon = 10 ** -4  # 10e-5

    #  INPUT ("CHUTE") INICIAL, MUDAR SE NECESSARIO
    vector_x = np.array(
        [5.0, 5.0]
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


def __calculate_direction(func: Function, vector_x):
    gradient = func.gradient(vector_x)
    hessian_inverse = np.linalg.inv(func.hessian(vector_x))
    return -(np.inner(hessian_inverse, gradient))


if __name__ == "__main__":
    newton_method()

