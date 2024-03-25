import numpy as np


def calculate_alpha(func, d_direction, x_vector, epislon=10**-4):
    #  "CHUTES" INICIAIS, MUDAR SE NECESSARIO
    alpha_lower = 0.0
    alpha_upper = 0.5

    x_test = __calculate_new_x(x_vector, d_direction, alpha_upper)
    galinha = __calculate_h_line(func, d_direction, x_test)

    while galinha < 0:
        alpha_upper *= 2
        x_test = __calculate_new_x(x_vector, d_direction, alpha_upper)
        galinha = __calculate_h_line(func, d_direction, x_test)

    alpha_mean = (alpha_upper + alpha_lower) / 2
    if abs(galinha) < epislon:
        return alpha_mean

    max_iterations = np.log2(alpha_upper / epislon)

    iteration = 0
    while iteration < max_iterations:
        alpha_mean = (alpha_lower + alpha_upper)/2
        x_test = __calculate_new_x(x_vector, d_direction, alpha_mean)
        galinha = __calculate_h_line(func, d_direction, x_test)

        if galinha > 0:
            alpha_upper = alpha_mean
        elif galinha < 0:
            alpha_lower = alpha_mean
        else:
            return alpha_mean
        iteration+=1
    return alpha_mean

def __calculate_h_line(func, d_direction, x_vector):
    grad = func.gradient(x_vector)
    return np.inner(grad, d_direction)

def __calculate_new_x(x_vector, d_direction, alpha):
    return x_vector + (alpha*d_direction)
