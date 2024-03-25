import numpy as np

"""
funcoes de kernel, autoexplicativas
"""

def get_kernel_function(kernel_name):
    return {
        'linear': linear_kernel,
        'polynomial': polynomial_kernel,
        'gaussian': gaussian_kernel
    }.get(kernel_name, linear_kernel)


def linear_kernel(x1, x2, gamma=None):
    # gamma not used
    k = np.dot(x1, x2)
    return k

def polynomial_kernel(x1, x2, gamma=3):
    if not gamma or gamma==0:
        gamma = 3
    k = (np.dot(x1, x2) + 1)
    k = np.power(k, gamma)
    return k

def gaussian_kernel(x1, x2, gamma=1.0):
    if not gamma or gamma == 0:
        gamma = 1.0

    numerator = np.linalg.norm(x1-x2)
    numerator = np.power(numerator, 2)

    denominator = 2 * np.power(gamma, 2)

    k = np.exp(-(numerator/denominator))

    return k
