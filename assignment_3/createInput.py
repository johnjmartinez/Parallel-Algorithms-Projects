#!/bin/python
import sys
import numpy as np

n = int(sys.argv[1])
#x = np.random.randint(-99999, 99999, n)  
x = np.random.randint(10, 99, n)  
np.savetxt('input.txt', x, fmt='%d', delimiter=' ', newline=' ')  
