import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

# List of things missing
# We still need to use the given eigen vectors manually compute the sequence, this will be done later


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

def iterativeRecurrence(A,u_0,n):
    """
    Compute the recurrence equation iteratively rather than by using eigen values. 
    Args:
        A (ndarray): A 3x3 diagonalizable matrix.
        a0 (ndarray): A 3x1 matrix of initial values.
        n (int): The nth iteration at which the sequence will end
    Returns:
        ndarray: The nth term of the sequence
    """
    u_n = A @ u_0
    for i in range(0,n+1):
        #print(f"The {i}th value is  = {u_n}") # Task requirement
        u_n = A @ u_n
        #print(f"Term {i}") # debug
        
    return u_n
# Poor practice, iterativeRecurrence and normalizedRecurrence can probably be condensed into a single function.
# For problem 4, kinda
def normalizedRecurrence(A,u_0,n):
    """
    Similar to iterativeRecurrence it instead returns the normalized iteratres of the sequence
    Args:
        A (ndarray): A 3x3 diagonalizable matrix.
        u_0 (ndarray): A 3x1 matrix of initial values.
        n (int): The nth iteration at which the sequence will end
    Returns:
        ndarray: The nth normalized term of the sequence.
    """
    
    v_0 = u_0 / np.linalg.norm(u_0)
    v_n = A @ v_0 / np.linalg.norm(A @ v_0)
    for i in range(0,n+1):
        #print(f"The {i}th value is  = {v_n}") # Task requirement
        v_n = (A @ v_n) / np.linalg.norm(A @ v_n)
        #print(f"Term {i}") # debug
        
    return v_n

# Task 5 
def diagonalizedRecurrence(A,u_0,n):
    """
   Multiplies the transposed normalized iteration with an input matrix A.
    Args:
        A (ndarray): A 3x3 diagonalizable matrix.
        u_0 (ndarray): A 3x1 matrix of initial values.
        n (int): The nth iteration at which the sequence will end
    Returns:
        ndarray: The diagonalized term q_n
    """
    v_0 = u_0 / np.linalg.norm(u_0)
    v_n = A @ v_0 / np.linalg.norm(A @ v_0)
    for i in range(0,n+1):
        # print(f"The {i}th value is  = {v_n}") # Task requirement
        v_n = v_n.T * A @ v_n / np.linalg.norm(A @ v_n)
        #print(f"Term {i}") # debug
        
    return v_n    

def fullfillEpsilon(u_0,A,n,epsilon):
#    i = 1
#    result = np.linalg.norm(v - v_star)
#    while result <=epsilon:
#    Main idea run the regular normalized interative but use a while loop to log when the loop finds its end point. 
#    v* determined by n
    iteration = 0
    v_star = normalizedRecurrence(A,u_0,n) #  / np.linalg.norm(getRecurence(A,u_0,n))
    print(f"v* = {v_star}")
    v_n = A @ u_0
    print(v_n)
    while np.linalg.norm(v_n) - np.linalg.norm(v_star) <= epsilon:
        #print(f"The {i}th value is  = {v_n}") # Task requirement
        v_n = A @ v_n
        print(f"Term {iteration}") # debug
        iteration += 1 
        
    return f"Converged in {iteration} iterations."



# Declare matrix A (Use dtype float to avoid overflow errors)
A = np.array([[1,3,2],[-3,4,3],[2,3,1]], dtype = float)
# Store the initial values in an array 
z_val = [np.array([9,1,9], dtype = float).T,np.array([1,0,1], dtype = float).T,np.array([1,12,-19], dtype = float).T]
alpha_val = [1,1/19]
# For task 1
# Compute for only the first z_0 value. 
# Also commented out by default
# print(f"For z_0 number 1 the the 1st value is: {getRecurence(A,z_val[0],1)}")
# print(f"For z_0 number 1 the the 10th value is: {getRecurence(A,z_val[0],10)}")
# print(f"For z_0 number 1 the the 20th value is: {getRecurence(A,z_val[0],20)}")

# Compute the matrices for each n-step and each alpha value.
# Commented out by default. 
# for i in range(1,3):
#     for j in range(0,2):
#         print(f"For z_0 number {i+1} with alpha value number {alpha_val[j]}  the 1st value is: {getRecurence(A,alpha_val[j]*z_val[i],1)}")
#         print(f"For z_0 number {i+1} with alpha value number {alpha_val[j]}  the 10th value is: {getRecurence(A,alpha_val[j]*z_val[i],10)}")
#         print(f"For z_0 number {i+1} with alpha value number {alpha_val[j]}  the 20th value is: {getRecurence(A,alpha_val[j]*z_val[i],20)}")
     

# Task 2, compute the matrices iteratively. 
# make comparison between iterative recurrence and the regular one
# print(f"The recurrence equation for z_0[2] at point 399 and alpha 1/19 is {getRecurence(A,alpha_val[1]*z_val[2],399)}, while the iterative one is {iterativeRecurrence(A,alpha_val[1]*z_val[2],399)} while the normalized one is {normalizedRecurrence(A,alpha_val[1]*z_val[2],399)} ")
# print(f"The recurrence equation for z_0[2] at point 400 and alpha 1/19 is {getRecurence(A,alpha_val[1]*z_val[2],400)}, while the iterative one is {iterativeRecurrence(A,alpha_val[1]*z_val[2],400)} while the normalized one is {normalizedRecurrence(A,alpha_val[1]*z_val[2],400)} ")

# Display the output for task 5. 
print(f"The normalized recurrence is {normalizedRecurrence(A,alpha_val[1]*z_val[2],400)}, the diagonalized recurrence is {diagonalizedRecurrence(A,alpha_val[1]*z_val[2],400)} ")
print(f"The normalized recurrence is {normalizedRecurrence(A,alpha_val[1]*z_val[2],29)}, the diagonalized recurrence is {diagonalizedRecurrence(A,alpha_val[1]*z_val[2],29)} ") #Converges exactly after 30 iterations

# Task 7
print(fullfillEpsilon(alpha_val[0]*z_val[2],A,400,1e-18))
