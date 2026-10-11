import numpy as np

class Matrix_Operations:
    """for making matrix A"""
    # for making projection and reflection matrix

    @staticmethod
    def Reader(filename):
        with open(filename, "r") as file:
            next(file)  # ommits header x, y

            data = [line.strip().split(",") for line in file if line.strip()]

        data = np.array(data, dtype=float)  # saving data from the splitting

        X = data[:, 0];
        Y = data[:, 1:];

        return X, Y

    @staticmethod #making matrix from .dat files
    def Matrix_Organiser(X):
        return np.column_stack((X, np.ones_like(X)))

    @staticmethod #Using projection matrix formula
    def Projection_Matrix(A):
        return A @ np.linalg.inv(A.T @ A) @ A.T

    @staticmethod #V =U direkt sum U ortho => R = 2 P - I
    def Refl_Matrix(P):
        I = np.identity(P.shape[0])
        R = 2 * P - I
        return R