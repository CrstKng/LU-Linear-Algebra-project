import matplotlib.pyplot as plt
import numpy as np

def plot_func(x, y, a, b, name="regression_plot.png"):
    plt.plot(x, y, 'o', label='Original data', markersize=10)
    plt.plot(x, a*x + b, 'r', label='Fitted line')
    plt.legend()
    plt.show()
    plt.savefig(name)

def plot_extra(x, y, a, b, P, name="regression_plot.png"):
    plt.plot(x, y, 'o', label='Original data', markersize=10)
    plt.plot(x, a*x + b, 'r', label='Fitted line')
    plt.plot(x, np.dot(P, y), 'b', label='Projection matrix line')
    plt.legend()
    plt.show()
    plt.savefig(name)

