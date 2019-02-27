# diplacement
# rotation
# squeezing

import numpy as np

class MultiMode():

    def __init__(self,n_modes,n_qubits_per_mode=2):
        pass

    def S2(self):
        # 2 mode squeezing
        # beam splitters
        pass

class QMode():
    """
    Class to handle single mode operations
    """

    def __init__(self,n_qubits_per_mode=2):

        self.n_qubits_per_mode = n_qubits_per_mode
        self.n_dim = 2**n_qubits_per_mode

    def a(self):
        # annihilation operator matrix for nxn
        a = np.zeros((self.n_dim,self.n_dim))
        for i in range(self.n_dim-1):
            a[i,i+1]= np.sqrt(i+1)
        return a

    def a_dag(self):
        # creation operator matrix for nxn
        a = np.zeros((self.n_dim,self.n_dim))
        for i in range(1,self.n_dim):
            a[i,i-1]= np.sqrt(i)
        return a

    def N(self):
        # Number operator matrix for nxn
        return np.matmul(a_dag(self.n_dim),a(self.n_dim))

    def D(self,alpha):
        # Displacement operator matrix for nxn
        arg = alpha*a_dag(self.n_dim)-np.conjugate(alpha)*a(self.n_dim)
        return np.expm1(arg)

    def S(self,z):
        #Single mode squeezing
        a2 = np.matmul(a(self.n_dim),a(self.n_dim))
        a2_dag = np.matmul(a_dag(self.n_dim),a_dag(self.n_dim))
        arg = np.conjugate(z)*a2 - z*a2_dag
        return np.expm1(arg)

    def R(self,phi):
        j = np.complex(0,1)
        arg = j*phi*np.matmul(a_dag(self.n_dim),a(self.n_dim))
        return np.expm1(arg)
