import numpy as np
import matplotlib.pyplot as plt

class Plotter:
    """Plotting graphs"""

   # 
    def Regression_Plotter(X: np.ndarray, Y: np.ndarray, a: float, b: float, name: str="MISSING-NAME"):
        """Plots the regression plot using Y = aX+ b relationship"""
        y = a * X + b
        plt.scatter(X, Y, color="red", marker="o", label="given data")
        plt.plot(X, y, color="blue", linestyle="-", label="regression line")

        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(name)
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.savefig(f"images/regression_plot_{name}.png")

   # 
    def Surface_plot(mat: np.ndarray, Y: np.ndarray, coeffcients: float, name: str="MISSING-NAME"):
        """ Given an error matrix mat and a matrix Y of given dependent variables and the linear regresion coefficeints make a 3D error plot. We expect a quadratic surface with single critical point. """

        """z = ||aX+b - Y||"""
        """a and b are x and y axis"""
        Y = np.asarray(Y).flatten()
        coefficients = np.asarray(coeffcients).flatten()

        a, b = coefficients

        #making linspace for x and y spanning
        a_itv = np.linspace(a - 2, a + 2, 100)
        b_itv = np.linspace(b - 2, b + 2, 100)

        aa, bb = np.meshgrid(a_itv, b_itv)

        #essentially works as a Z axis => finding out the minimum distance between the Y and the regression function depending on a and b
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
        plt.savefig(f"images/surface_plot_{name}.png")

    #change to Plotter_reflection_projection
    def Plotter_reflection_projection(X: np.ndarray, Y: np.ndarray, Y_hat: np.ndarray, Y_refl: np.ndarray | None = None, name: str="MISSING-NAME"):
        """
        Now also plots the data points found using the projection and (optionally) reflection matrix.
        """

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
        plt.savefig(f"images/reflection_projection_plot_{name}.png")

    #Same frame as the plotter but using vectors and displays the logarithmic scale
    def Conv_plot(epsilons: float, counts_theoretic, counts_num, title: str, num: str = "0"):
        plt.rcParams['text.usetex'] = True
        fig, ax = plt.subplots()

        ax.semilogy(epsilons, counts_theoretic, color="red", label="theoretic limit")
        try:
            ax.semilogy(epsilons, counts_num, color="blue", label="numerical limit")
        except ValueError:
            print(f"{num} diverges")

        ax.set_xlabel(r"Tolerance ($\varepsilon$)")
        ax.set_ylabel("Number of iterations")
        ax.set_title(rf"{title}")
        ax.grid(True, which="both")
        ax.legend()

        fig.tight_layout()
        fig.show()
        fig.savefig(f"images/convergence_plots/convergence_plot_{num}.png")
