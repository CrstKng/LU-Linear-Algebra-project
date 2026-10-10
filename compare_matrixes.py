import numpy as np
from scipy_func import lin_func

def is_projection(P):
    return np.allclose(P, np.dot(P, P), atol = 1e-08)

def is_orthogonal_projection(P):
    return is_projection(P) and np.allclose(P, np.transpose(P), atol = 1e-08)

def calc_P(x_arr, a, b):
    y_hat = a * x_arr + b
    A = np.outer(y_hat, y_hat) / (y_hat @ y_hat)
    return A

a = np.array([[1, 2], [3, 5]])
b = np.array([1, 2])
x = np.linalg.solve(a, b)

# def test():
#     mult = np.dot(A_t, A)
#     P = np.dot(reg.y0, np.dot(mult, A_t))

# test()