import numpy as np
from scipy.linalg import eig

class Eig_operation:

    
    def Eig_Dekom(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Gets eigenvalues and eigenvectors for M
        """
        eigwert, eigvek = eig(M)
        return eigwert, eigvek

    def Iteration(M: np.ndarray, z0: np.array, N: int=400) -> np.ndarray:
        """
        Iteratively computes z_n in term of the eigenvalue and eigenvectors of M
        """
        values = np.zeros((N+1, len(z0)))
        values[0] = z0

        for n in range(N):
            values[n+1] = M @ values[n]

        return values

     #Iteration for normalising the vector
    def Norm_Iteration(M: np.ndarray, z0:np.array, N: int=400) -> tuple[float, float]:
        """
        Computes the preceeding eigenvectors and corresponding value, q, for some iteration, since the step size is controlled, vectors were calculated after normalisation
        """
        values = np.zeros((N+1, len(z0)))
        q_values = np.zeros(N+1)

        values[0] = z0/np.linalg.norm(z0)
        q_values[0] = values[0] @ M @ values[0]

        for n in range(N):
            z_1 = M @ values[n]
            values[n+1] = z_1/np.linalg.norm(z_1)
            q_values[n+1] = values[n+1] @ M @ values[n + 1]

        return values, q_values

     #Counts the convergence iteration of the vector, incrementing the index X^i for X is the eigenvector we iteratively putting
    def Convergence_counter(values: np.ndarray, v_ast: np.ndarray, epsilon: float= 1e-8) -> int:
        """ Runs a modified version of Norm_Iteration that checks if the difference between a theoretical v_ast* and a vector v_n.
            The method then stops when the difference is smaller than a specified epsilon.""" 
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