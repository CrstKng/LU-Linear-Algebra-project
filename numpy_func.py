import numpy as np
import regression as reg
import matplotlib.pyplot as plt

#a @ x = b;
#a = ATA
#b = ATY

A_t = np.vstack([reg.x0, np.ones(len(reg.x0))]).T #AT

A = A_t.T

m, c = np.linalg.lstsq(A_t, reg.y0)[0]

print(f"{m}, {c}")

def err_func(m, c, x, y):
    return abs((m*x + c) - y)**2

#plt.plot(reg.x0, reg.y0, 'o', label='Original data', markersize=10)
#plt.plot(reg.x0, m*reg.x0 + c, 'r', label='Fitted line')
plt.legend()
plt.savefig("images/regression_plot.png")

#plt.plot(reg.x0, err_func(m, c, reg.x0, reg.y0), label='Error')
plt.legend()
a = np.linspace(-5, 5, 20)
b = np.linspace(-5, 5, 20)
U, V = np.meshgrid(a, b)
Z = err_func(U, V, reg.x0, reg.y0)

# Plot the surface
#fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#ax.plot_surface(U,V, Z, vmin=Z.min() * 2, cmap="viridis")
#plt.savefig("images/err.png")
#plt.show

def y_hat(m, c, x):
    return m*x + c

# Part 5
#A_t 2 by 20, A 20 by 2 => A_t A = 2 by 2
#reg.y0 20 by 1, mult * A_t => 2 by 2 * 2 by 20 => 2 by 20

P = A_t @ np.linalg.inv(A @ A_t) @ A
#(A_t @ np.linalg.inv(A @ A_t) size
print(P.shape)
#print(f"This is the size of P: {P.size}\nThis is the size of ATA inverse: {np.linalg.inv(A @ A_t).size}\nThis is the size: {(A_t @ np.linalg.inv(A @ A_t) @ A).size}")

# Part 6
def is_projection(P):
    return np.allclose(P, np.dot(P, P), atol = 1e-08)

def is_orthogonal_projection(P):
    return is_projection(P) and np.allclose(P, np.transpose(P), atol = 1e-08)

print(is_projection(P))
print(is_orthogonal_projection(P))

y_new_hat = P @ reg.y0
plt.plot(reg.x0, reg.y0, 'o', label='Original data', markersize=10)
plt.plot(reg.x0, m*reg.x0 + c, 'r', label='Fitted line')
plt.plot(reg.x0, 'g', label='Projected line')

#plt.savefig("images/projection_plot.png")

# Task 8
residuals = reg.y0 - y_new_hat

print(f"The dot product of y and y_hat is {np.dot(residuals,y_new_hat)}")

#Task 9
R = 2*P - np.identity(20)
plt.plot(reg.x0, R @ reg.y0, 'o', label='Reflected data', markersize=10)

plt.savefig("images/Reflection_plot.png")