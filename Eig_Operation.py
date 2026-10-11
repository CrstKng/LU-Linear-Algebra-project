import numpy as np
from scipy.linalg import eig

class Eig_operation:

    @staticmethod
    def Eig_Dekom(M):
        eigwert, eigvek = eig(M)
        return eigwert, eigvek

    @staticmethod
    def Iteration(M, z0, N=400):
        werten = np.zeros((N+1, len(z0)))
        werten[0] = z0

        for n in range(N):
            werten[n+1] = M @ werten[n]

        return werten

    @staticmethod #Iteration for normalising the vector
    def Norm_Iteration(M, z0, N=400):
        werten = np.zeros((N+1, len(z0)))
        q_werten = np.zeros(N+1)

        werten[0] = z0/np.linalg.norm(z0)
        q_werten[0] = werten[0] @ M @ werten[0]

        for n in range(N):
            z_1 = M @ werten[n]
            werten[n+1] = z_1/np.linalg.norm(z_1)
            q_werten[n+1] = werten[n+1] @ M @ werten[n + 1]

        return werten, q_werten

    @staticmethod #Counts the convergence iteration of the vector, incrementing the index X^i for X is the eigenvector we iteratively putting
    def Convergence_counter(values, v_ast, epsilon):
        values = np.asarray(values)
        v_ast = np.asarray(v_ast)

        if values.ndim == 1:
            dist = np.abs(values - v_ast)
        else:
            dist = np.linalg.norm(values - v_ast, axis=1)

        ind = np.where(dist < epsilon)[0]

        if len(ind) == 0:
            return None

        return ind[0]