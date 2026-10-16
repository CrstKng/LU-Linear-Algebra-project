import numpy as np
import scipy as sp

from Matrix_Operations import Matrix_Operations
from Functions import Functions
from Plotter import Plotter
from Eig_Operation import Eig_operation

# Import regression files

#regression 1-1 management
X, Y = Matrix_Operations.Reader("data/regression_1-1.dat")
A = Matrix_Operations.Matrix_Organiser(X)

#regression 2-1 management
X1, Y_mul = Matrix_Operations.Reader("data/regression_2-1.dat")
B = Matrix_Operations.Matrix_Organiser(X1)

#write down the datasets into dictionary to improve usability.
ds = [("DS 1", X, Y[:, 0],A),
      ("DS 2", X1, Y_mul[:, 0],B),
      ("DS 3", X1, Y_mul[:, 1],B),
      ("DS 4", X1, Y_mul[:, 2],B),]

init_guess = np.array([1.0, 1.0])


############################
### Task 1.1  & Task 1.2 ###
############################

for name, x, y, mat in ds: #mat will be the n by 2 matrix
     #plotting regression lines and errors
    Lin_const = Functions.Solver(mat, y)
    coefficients_min = sp.optimize.fmin(Functions.ERR, init_guess, args=(mat, y), disp=False)

    print(f"{name}\n"
          f"Lin const: {Lin_const}\n"
          f"fmin: {coefficients_min}\n"
          f"min.err: {Functions.ERR(coefficients_min, mat, y)}\n")

    Plotter.Regression_Plotter(x, y, Lin_const[0], Lin_const[1], f"Plot {name}")

################
### Task 1.3 ###
################

#err surface plot for DS 1
coeffs = Functions.Solver(A, Y[:, 0])
#Plotter.Surface_plot(A, Y[:, 0], coeffs)

################
### Task 1.5 ###
################


#proj. and refl. for DS 1 =? can be repeated for different data sets like DS 2, 3, and 4
P = Matrix_Operations.Projection_Matrix(A) #projection matrix
y_hat = P @ Y[:, 0]
coefficients = Functions.Solver(A, Y[:,0])
y_hat_reg = A @ coefficients

################
### Task 1.6 ###
################

# Sanity Check: proj. P == True ? refl. R == True ? data Y and y_hat == orth. proj ?
print(f"y_hat and y_hat_regression are all equal: {np.allclose(y_hat, y_hat_reg)}")

print(f"P is projection: {Functions.Is_Proj(P)}")

print(f"P is orthogonal projection: {Functions.Is_Ortho_Proj(P)}")

################
### Task 1.7 ###
################
#print(Functions.Is_Reflek(R))
Plotter.Plotter_reflection_projection(X, Y[:, 0], y_hat) # projeciton, task 1.7

################
### Task 1.8 ###
################
print(f"y hat is the orthogonal projection of y: {Functions.Verify_ortho_proj(Y[:,0], y_hat)}") #task 8

################
### Task 1.9 ###
################
R = Matrix_Operations.Refl_Matrix(P)
y_refl = R @ Y[:,0]
Plotter.Plotter_reflection_projection(X, Y[:, 0], y_hat, y_refl, "Task 1_9") # reflection, task 1.9

###### End of task 1 #######
#-------------------------------------------------------------------------------------------------------------------------------


################
### Task 2.1 ###
################

M = np.array(
    [[1, 3, 2],
    [-3, 4, 3],
    [2, 3, 1]], dtype=float)

eig_wert, v = Eig_operation.Eig_Dekom(M)

#print(f"eigenwert: {eig_wert}, eigenvektor: {v}")

################
### Task 2.2 ###
################

#initial value set z0
z0 = np.array([9, 1, 9], dtype=float)

values = Eig_operation.Iteration(M, z0, N=400)

# print(values[0]) #initial values
# print(values[1]) #M @ z0
# print(values[400]) #400th iteration

IV = [("Initial 1", np.array([9, 1, 9], dtype=float), r"Initial 1 $z_0 = [9, 1, 9]^T$"),
      ("Initial 2_1, alpha = 1", np.array([9, 1, 9], dtype=float), r"Initial 2.1 $z_0 = [9, 1, 9]^T$, $\alpha$ = 1, "),
      ("Initial 2_2, alpha = 1 over 19", np.array([1/19, 0, 1/19], dtype=float), r"Initial 2.2 $z_0 = [1, 0, 1]^T$, $\alpha$ = $\frac{1}{19}$"),
      ("Initial 3_1, alpha = 1", np.array([1, 12, -19], dtype=float), r"Initial 3.1  $z_0 = [1, 12, -19]^T$, $\alpha$ = 1"),
      ("Initial 3_2, alpha = 1 over 19", np.array([1/19, 12/19, -1], dtype=float), r"Initial 3.2 $z_0 = [1, 12, -19]^T$, $\alpha$ = $\frac{1}{19}$")]



###########################
### Task 2.4 & Task 2.5 ###
###########################

results = {}

N = 400
epsilon = 1e-8
epsilons = np.logspace(-1, -14, 14)

for file_name, z0, title in IV:
    
    # Task 5: normalized iterates and q_n in an array [0]: task 4, [1]: task 5
    values, q_values = Eig_operation.Norm_Iteration(M, z0, N=N)

    v400 = values[-1]
    q400 = q_values[-1]

    # Theoretical dominant eigenvector
    eigenvalues, eigenvectors = Eig_operation.Eig_Dekom(M)
    dominant_index = np.argmax(np.abs(eigenvalues))
    v_ast = np.real(eigenvectors[:, dominant_index])

    # Normalize the eigenvector
    v_ast = v_ast / np.linalg.norm(v_ast)

    # Align the sign with the numerical sequence
    if np.dot(v_ast, v400) < 0:
        v_ast = -v_ast

    q_ast = v_ast @ M @ v_ast
    
    ################
    ### Task 2.7 ###
    ################
    
    # Task 7: count until epsilon = 1e-8
    n_v_theory = Eig_operation.Convergence_counter(values, v_ast, epsilon)
    n_v_numerical = Eig_operation.Convergence_counter(values, v400, epsilon)

    n_q_theory = Eig_operation.Convergence_counter(q_values, q_ast, epsilon)
    n_q_numerical = Eig_operation.Convergence_counter(q_values, q400, epsilon)

    print(f"\n{file_name}")
    print("v_400 =", v400)
    print("q_400 =", q400)
    print("Iterations for v, theoretical limit:", n_v_theory)
    print("Iterations for v, numerical limit:", n_v_numerical)
    print("Iterations for q, theoretical limit:", n_q_theory)
    print("Iterations for q, numerical limit:", n_q_numerical)

    
    ################
    ### Task 2.8 ###
    ################
    counts_v_theory = []
    counts_v_numerical = []
    counts_q_theory = []
    counts_q_numerical = []

    for eps in epsilons:
        counts_v_theory.append(Eig_operation.Convergence_counter(values, v_ast, eps))
        counts_v_numerical.append(Eig_operation.Convergence_counter(values, v400, eps))

        counts_q_theory.append(Eig_operation.Convergence_counter(q_values, q_ast, eps))
        counts_q_numerical.append(Eig_operation.Convergence_counter(q_values, q400, eps))

    # Plot convergence of normalized vectors
    Plotter.Conv_plot(epsilons, counts_v_theory, counts_v_numerical, title=rf"{title}: convergence of $v_n$", num = f"{file_name}")

    # Plot convergence of q_n
    Plotter.Conv_plot(epsilons, counts_q_theory, counts_q_numerical, title=rf"{title}: convergence of $q_n$", num = f"{file_name}") #err occur? why no positive values?