# diplacement
# rotation
# squeezing

import numpy as np
from scipy.linalg import expm
# import numpy.linalg.matrix_power as mat_pow

class TwoQMode():

    def __init__(self,n_qubits_per_mode=2):

        self.mode = QMode(n_qubits_per_mode)
        I = np.eye(self.mode.n_dim)

        self.a1 = np.kron(self.mode.a,I)
        self.a2 = np.kron(I,self.mode.a)
        self.a1_dag = self.a1.conj().T
        self.a2_dag = self.a2.conj().T

    def S2(self,z):
        #two mode squeeze
        a12 = np.matmul(self.a1,self.a2)
        a12_dag = np.matmul(self.a1_dag,self.a2_dag)
        arg = np.conjugate(z)*a12 - z*a12_dag
        return expm(arg)

    def BS(self,phi):
        a12dag = np.matmul(self.a1,self.a2_dag)
        a1dag2 = np.matmul(self.a1_dag,self.a2)
        arg = phi*a12dag - np.conjuagate(phi)*a1dag2
        return expm(arg)


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
