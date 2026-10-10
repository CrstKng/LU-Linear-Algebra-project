import numpy as np
from scipy_func import scipy_linearization

#task 1

# Opening regression files 
data = np.loadtxt("regression_1-1.dat", delimiter=",", skiprows = 1)
x0 = np.array(data[:, 0])
y0= np.array(data[:, 1])

# second regression file
data = np.loadtxt("regression_2-1.dat", delimiter=",", skiprows = 1)
x1 = data[:, 0]  
y1 = data[:, 1]  
y2 = data[:, 2]
y3 = data[:, 3]

def test():
    result = scipy_linearization(x0, y0)
    print(result)
    return
