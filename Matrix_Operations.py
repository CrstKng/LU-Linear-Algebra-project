import numpy as np

class Matrix_Operations:
    """ Operations to import data and preform basic operations required for Task 1."""
    def Reader(filename: str) -> tuple[np.ndarray, np.ndarray]:
        """Imports the directory of the regression files through a string and stores them in numpy matrices."""
        with open(filename, "r") as file:
            next(file)  # ommits header x, y

            data = [line.strip().split(",") for line in file if line.strip()]

        data = np.array(data, dtype=float)  # saving data from the splitting

        X = data[:, 0];
        Y = data[:, 1:];

        return X, Y

     #making matrix from .dat files
    def Matrix_Organiser(X: np.ndarray) -> np.ndarray:
        """ Adds a column of ones to the input matrix (X) so that it can be solved using the least square method."""
        return np.column_stack((X, np.ones_like(X)))

   #  #Using projection matrix formula
    def Projection_Matrix(A: np.ndarray) -> np.ndarray:
        """ Computes the projection matrix of a given matrix A. """ 
        return A @ np.linalg.inv(A.T @ A) @ A.T

     #V =U direkt sum U ortho => R = 2 P - I
    def Refl_Matrix(P: np.ndarray) -> np.ndarray:
        """Determines the reflection matriix given with the projection matrix P and the equation R = 2P - I"""
        I = np.identity(P.shape[0])
        R = 2 * P - I
        return R