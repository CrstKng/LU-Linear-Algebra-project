import numpy as np
from scipy.optimize import fmin

def lin_func(x: float, a: float, b: float) -> float:
    return a * x + b

def objective(params, x_arr, y_arr):
    a, b = params
    return np.sum((y_arr - lin_func(np.array(x_arr), a, b))**2)

def scipy_linearization(x_arr: list[float], y_arr: list[float]) -> list[float]:
    result = fmin(objective, [1, 0], args=(x_arr, y_arr), maxiter=1000) #initial guess (a,b) = (1,0)
    a, b = result
    return a, b