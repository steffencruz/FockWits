# diplacement
# rotation
# squeezing

import numpy as np
from scipy.linalg import expm
# import numpy.linalg.matrix_power as mat_pow

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

        # annihilation operator matrix for nxn
        self.a = np.zeros((self.n_dim,self.n_dim))

        for i in range(self.n_dim-1):
            self.a[i,i+1]= np.sqrt(i+1)

        # creation operator matrix for nxn
        self.a_dag = self.a.conj().T

        # Number operator matrix for nxn
        self.N = np.matmul(self.a_dag,self.a)

    def D(self,alpha):
        # Displacement operator matrix for nxn
        arg = alpha*self.a_dag-np.conjugate(alpha)*self.a
        return expm(arg)

    def S(self,z):
        #Single mode squeezing
        a2 = np.matmul(self.a,self.a)
        a2_dag = np.matmul(self.a_dag,self.a_dag)
        arg = np.conjugate(z)*a2 - z*a2_dag
        return expm(arg)

    def R(self,phi):
        j = np.complex(0,1)
        arg = j*phi*np.matmul(self.a_dag,self.a)
        return expm(arg)
