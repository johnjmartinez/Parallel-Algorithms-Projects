#!/bin/python
import sys
import numpy as np

n = int(sys.argv[1])
x = np.random.randint(-999999, 999999, n)  
#x = np.random.randint(100, 999, n)  
np.savetxt('input.txt', x, fmt='%d', delimiter=' ') #, newline=' ')  
