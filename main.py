import numpy as np
from scipy_func import scipy_linearization
from compare_matrixes import calc_P, is_projection, is_orthogonal_projection
from plot import plot_func, plot_extra

#task 1
def main():
    # Opening regression files 
    data = np.loadtxt("regression_1-1.dat", delimiter=",", skiprows = 1)
    x0 = data[:, 0]
    y0= data[:, 1]

    # second regression file
    data = np.loadtxt("regression_2-1.dat", delimiter=",", skiprows = 1)
    x1 = data[:, 0]  
    y1 = data[:, 1]  
    y2 = data[:, 2]
    y3 = data[:, 3]

    result = scipy_linearization(x0, y0)
    a, b = result
    print(f"regression 1: a = {float(a)}, b = {float(b)}")
    #plot_func(x0, y0, a, b, name="regression_1_plot.png")

    P = calc_P(x0, a, b)
    print(f"P: {P}, is projection: {is_projection(P)}, is orthogonal projections: {is_orthogonal_projection(P)}")
    plot_extra(x0, y0, a, b, P, name="regression_1_plot.png")

    result = scipy_linearization(x1, y1)
    a, b = result
    print(f"regression 2.1: a = {float(a)}, b = {float(b)}")
    #plot_func(x1, y1, a, b, name="regression_2.1_plot.png")
    P = calc_P(x1, a, b)
    plot_extra(x1, y1, a, b, P, name="regression_2.1_plot.png")

    result = scipy_linearization(x1, y2)
    a, b = result
    print(f"regression 2.2: a = {float(a)}, b = {float(b)}")
    #plot_func(x1, y2, a, b, name="regression_2.2_plot.png")
    plot_extra(x1, y2, a, b, P, name="regression_2.2_plot.png")

    result = scipy_linearization(x1, y3)
    a, b = result
    print(f"regression 2.3: a = {a}, b = {float(b)}")
    #plot_func(x1, y3, a, b, name="regression_2.3_plot.png")
    plot_extra(x1, y3, a, b, P, name="regression_2.3_plot.png")

main()
