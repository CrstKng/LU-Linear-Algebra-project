import numpy as np
import matplotlib.pyplot as plt

class Plotter:
    """Plotting things"""

    @staticmethod
    def Regression_Plotter(X, Y, a, b):
        y = a * X + b
        plt.scatter(X, Y, color="red", marker="o", label="given data")
        plt.plot(X, y, color="blue", linestyle="-", label="regression line")

        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Linear Regression")
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def Surface_plot(mat, Y, coeffcients):
        Y = np.asarray(Y).flatten()
        coefficients = np.asarray(coeffcients).flatten()

        a, b = coefficients

        a_itv = np.linspace(a - 2, a + 2, 100)
        b_itv = np.linspace(b - 2, b + 2, 100)

        aa, bb = np.meshgrid(a_itv, b_itv)

        resid = (mat[:, 0, None, None] * aa + mat[:, 1, None, None] * bb - Y[:, None, None])

        err = np.linalg.norm(resid, axis=0)
        err_opti = np.linalg.norm(mat @ coefficients - Y)
        #sanity check
        #print(f"Minimum err is at: a= {a:.8f}, b= {b:.8f}, err= {err:.8f}")

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")

        ax.plot_surface(aa, bb, err, cmap="viridis", alpha=0.85)
        ax.scatter(a_itv, b_itv, err_opti, color="red", marker="o", s=150, label="LSQ Sol.")
        ax.set_xlabel("a")
        ax.set_ylabel("b")
        ax.set_zlabel("err")
        ax.set_title("LSQ Sol")
        ax.legend()
        plt.show()

    @staticmethod
    def Plotter(X, Y, Y_hat, Y_refl=None):
        plt.scatter(X, Y, color="red", marker="o", label="given data")
        plt.plot(X, Y_hat, color="blue", label="projection")
        if Y_refl is not None:
            plt.plot(X, Y_refl, color="green", label="reflection")

        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Projection and Reflection" if Y_refl is not None else "Projection")
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod #Same frame as the plotter but using vectors and displays the logarithmic scale
    def Conv_plot(epsilons, counts_theoretic, counts_num, title, ylabel="Number of iterations"):
        fix, ax = plt.subplots()

        ax.semilogy(epsilons, counts_theoretic, color="red", label="theoretic limit")
        ax.semilogy(epsilons, counts_num, color="blue", label="numerical limit")

        ax.set_xlabel("Tolerance and varepsilon")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="both")
        ax.legend()

        plt.tight_layout()
        plt.show()