import numpy as np

class Functions:
    """Methods required for task 1. It includes a solver for the least square method, the error caclulations and checks for the matrices."""
    
    def Solver(A: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Solves the least square method for given matrices A and Y
        """
        numbers = np.linalg.lstsq(A, Y, rcond=None)[0]  # A is not a rectangular condition
        return numbers

    
    def ERR(numbers: float, A: np.ndarray, Y: np.ndarray) -> float:
        """
        Computes the scalar distance res between the matrix and the Y array (horizontal))
        """
        res = A @ numbers - Y.flatten()  # => finding the shortest distance between A*[a, b] and Y
        return np.linalg.norm(res) ** 2

    
    def Is_Proj(P: np.ndarray) -> bool:
        """
        Checks if P is projection
        """
        return np.allclose(P @ P, P)

    
    def Is_Ortho_Proj(P: np.ndarray) -> bool:
        """
        Check if a given matrix P is a orthogonal projection.
        """
        return Functions.Is_Proj(P) and np.allclose(P, P.T)

    
    def Is_Reflek(R: np.ndarray) -> bool:        
        """
        Checks if a given R is refletion
        """
        I = np.identity(R.shape[0])
        return (np.allclose(R @ R, I) and np.allclose(R.T, R))

    
    def Verify_ortho_proj(Y: np.ndarray, Y_hat: np.ndarray, tol: float=1e-8) -> bool:
        """
        Verifies if the projection of Y, Y_hat, is orthogonal to the Y via residual scalar
        """
        res = np.asarray(Y).flatten() - np.asarray(Y_hat).flatten()
        Y_hat = np.asarray(Y_hat).flatten()
        dot_prod = np.dot(Y_hat, res)
        return np.isclose(dot_prod, 0, atol=tol)

