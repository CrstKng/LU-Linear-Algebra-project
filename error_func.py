from matplotlib.mpl_toolkits.mplot3d.axes3d.Axes3D import plot_surface
import matplotlib.pyplot as plt


def err_func():
    #plt.plot(reg.x0, err_func(m, c, reg.x0, reg.y0), label='Error')
    plt.legend()
    a = np.linspace(-5, 5, 20)
    b = np.linspace(-5, 5, 20)
    U, V = np.meshgrid(a, b)
    Z = err_func(U, V, reg.x0, reg.y0)

    # Plot the surface
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(U,V, Z, vmin=Z.min() * 2, cmap="viridis")
    plt.savefig("err.png")