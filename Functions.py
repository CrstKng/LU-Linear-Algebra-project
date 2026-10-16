import numpy as np

class Functions:
    """LSQ solving, ERR, Matrix Verifizierung"""
    @staticmethod
    def Solver(A, Y):
        numbers = np.linalg.lstsq(A, Y, rcond=None)[0]  # A is not a rectangular condition
        return numbers

    @staticmethod
    def ERR(numbers, A, Y):
        res = A @ numbers - Y.flatten()  # => finding the shortest distance between A*[a, b] and Y
        return np.linalg.norm(res) ** 2

    @staticmethod
    def Frage_Proj(P):
        return np.allclose(P @ P, P)

    @staticmethod
    def Frage_Ortho_Proj(P):
        return Functions.Frage_Proj(P) and np.allclose(P, P.T)

    @staticmethod
    def Frage_Reflek(R):
        I = np.identity(R.shape[0])
        return (np.allclose(R @ R, I) and np.allclose(R.T, R))

    @staticmethod
    def Verifiziert_ortho_proj(Y, Y_hat, tol=1e-8):
        res = np.asarray(Y).flatten() - np.asarray(Y_hat).flatten()
        Y_hat = np.asarray(Y_hat).flatten()
        dot_prod = np.dot(Y_hat, res)
        return np.isclose(dot_prod, 0, atol=tol)