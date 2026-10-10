import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

def getRecurence(A,a0,n):
    """
    Get recurrence equation for a given matrix and it's starting values.
    Args:
        A (ndarray): A 3x3 diagonalizable matrix.
        a0 (ndarray): A 3x1 matrix of initial values.
        n (int): The nth iteration of the recurrence
    Returns:
        ndarray: The iterative step of the recursive sequence.
    """
    #Obtain eigen values and eigen vectors for A
    eigvals, eigvecs = sp.linalg.eig(A) 
    return a0[0]*eigvals[0]**n * eigvecs[0].T + a0[1]*eigvals[1]**n * eigvecs[1].T + a0[2]*eigvals[2]**n * eigvecs[2].T 





# Declare matrix A
A = np.array([[1,3,2],[-3,4,3],[2,3,1]])
# Store the initial values in an array 
z_val = [np.array([9,1,9]).T,np.array([1,0,1]).T,np.dot(1/19,np.array([1,12,-19]).T)]

for i in range(0,3):
    print(getRecurence(A,z_val[i],1))
    print(getRecurence(A,z_val[i],10))
    print(getRecurence(A,z_val[i],20))

# Debug
# print(f"The eigen values are: {eigvals}, the eigen vectors are: {eigvecs}")

