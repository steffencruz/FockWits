# diplacement
# rotation
# squeezing

import numpy as np

def a(n):
    # annihilation operator matrix for nxn
    a = np.zeros((n,n))

    for i in range(n-1):
        a[i,i+1]= np.sqrt(i+1)
    return a

def a_dag(n):
    # creation operator matrix for nxn
    a = np.zeros((n,n))

    for i in range(1,n):
        a[i,i-1]= np.sqrt(i)
    return a

def N(n):
    # Number operator matrix for nxn
    return np.matmul(a_dag(n),a(n))

def D(z,n):
    # Displacement operator matrix for nxn
    arg = z*a_dag(n)+z*a(n)
    return np.expm1(arg)
