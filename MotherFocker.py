# diplacement
# rotation
# squeezing

import numpy as np

# n is number of dimensions, which
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

def D(alpha,n):
    # Displacement operator matrix for nxn
    arg = alpha*a_dag(n)-np.conjugate(alpha)*a(n)
    return np.expm1(arg)

def S(z,n):
    #Single mode squeezing
    a2 = np.matmul(a(n),a(n))
    a2_dag = np.matmul(a_dag(n),a_dag(n))
    arg = np.conjugate(z)*a2 - z*a2_dag
    return np.expm1(arg)

def R(phi,n):
    j = np.complex(0,1)
    arg = j*phi*np.matmul(a_dag(n),a(n))
    return np.expm1(arg)
