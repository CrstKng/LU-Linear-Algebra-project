import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

def getRecurence(A,u0,n):
    """
    Get recurrence equation for a given matrix and it's starting values. This solves the recurence relations by diagonalizing the matrix A where A = T* D *D⁻¹. Since a recurrence relationship can be represented in the form u_k = A^n * u_0 we want to find A = T*D^n*T^-1*u_0.
    Args:
        A (ndarray): A 3x3 diagonalizable matrix.
        a0 (ndarray): A 3x1 matrix of initial values.
        n (int): The nth iteration of the recurrence
    Returns:
        ndarray: The iterative step of the recursive sequence.
    """
    #Obtain eigen values and eigen vectors for A
    eigvals, T = sp.linalg.eig(A) 
    # Find T in terms of the starting conditions u_0
    c = np.linalg.solve(T, u0)

    return T @ (eigvals**n * c)
 
# Declare matrix A
A = np.array([[1,3,2],[-3,4,3],[2,3,1]])
# Store the initial values in an array 
z_val = [np.array([9,1,9]).T,np.array([1,0,1]).T,np.array([1,12,-19]).T]
alpha_val = [1,1/19]
# For task 1
# Compute for only the first z_0 value. 
print(f"For z_0 number 1 the the 1st value is: {getRecurence(A,z_val[0],1)}")
print(f"For z_0 number 1 the the 10th value is: {getRecurence(A,z_val[0],10)}")
print(f"For z_0 number 1 the the 20th value is: {getRecurence(A,z_val[0],20)}")

# Compute the matrices for each n-step and each alpha value.
for i in range(1,3):
    for j in range(0,2):
        print(f"For z_0 number {i+1} with alpha value number {alpha_val[j]}  the 1st value is: {getRecurence(A,alpha_val[j]*z_val[i],1)}")
        print(f"For z_0 number {i+1} with alpha value number {alpha_val[j]}  the 10th value is: {getRecurence(A,alpha_val[j]*z_val[i],10)}")
        print(f"For z_0 number {i+1} with alpha value number {alpha_val[j]}  the 20th value is: {getRecurence(A,alpha_val[j]*z_val[i],20)}")
     



