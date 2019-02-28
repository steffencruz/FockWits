# diplacement
# rotation
# squeezing
# from qiskit import QuantumCircuit,QuantumRegister,ClassicalRegister
# from qiskit import execute,Aer

import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

import argparse

class CVGates:
    def __init__(self,n_qubits_per_mode=2):

        self.mode = QMode(n_qubits_per_mode)
        self.n_dim = self.mode.n_dim**2
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
        arg = phi*a12dag - np.conjugate(phi)*a1dag2
        return expm(arg)

    def BHM(self,tmax,k,n=10,J=1,U=0.1):
        #Implement simple two-mode Bose-Hubbard simulation
        j = np.complex(0,1)
        BKR = {}
        for t in np.linspace(0,tmax,n):
            for layer in range(k):
                phi = -j*J*t/k
                r = -U*t/(2*k)
                BS = self.BS(phi)
                K = np.kron(self.mode.K(r),self.mode.K(r))
                R = np.kron(self.mode.R(-r),self.mode.R(-r))
                BK = np.matmul(BS,K)
                BKR[t] = np.matmul(BK,R)
        return BKR


class QMode():
    """
    Class to handle single mode operations
    """

    def __init__(self,n_qubits_per_mode=2):

        self.n_qubits_per_mode = n_qubits_per_mode
        self.n_dim = 2**n_qubits_per_mode

        I = np.eye(self.n_dim)

        # Annihilation operator
        self.a = np.zeros((self.n_dim,self.n_dim))
        for i in range(self.n_dim-1):
            self.a[i,i+1]= np.sqrt(i+1)

        # Creation operator
        self.a_dag = self.a.conj().T

        # Number operator
        self.N = np.matmul(self.a_dag,self.a)

        # 2-qumodes operators
        self.a1 = np.kron(self.a, I)
        self.a2 = np.kron(I, self.a)
        self.a1_dag = self.a1.conj().T
        self.a2_dag = self.a2.conj().T

    def D(self, alpha):
        # Displacement operator matrix for nxn
        arg = alpha*self.a_dag-np.conjugate(alpha)*self.a
        return expm(arg)

    def S(self, z):
        #Single mode squeezing
        a2 = np.matmul(self.a, self.a)
        a2_dag = np.matmul(self.a_dag, self.a_dag)
        arg = np.conjugate(z)*a2 - z*a2_dag
        return expm(arg)

    def R(self, phi):
        arg = 1j*phi*np.matmul(self.a_dag, self.a)
        return expm(arg)

    def K(self, kappa):
        j = np.complex(0,1)
        arg = j*kappa*np.matmul(self.N, self.N)
        return expm(arg)

    def S2(self, z):
        #two mode squeeze
        a12 = np.matmul(self.a1, self.a2)
        a12_dag = np.matmul(self.a1_dag, self.a2_dag)
        arg = np.conjugate(z)*a12 - z*a12_dag
        return expm(arg)

    def BS(self, phi):
        a12dag = np.matmul(self.a1, self.a2_dag)
        a1dag2 = np.matmul(self.a1_dag, self.a2)
        arg = phi*a12dag - np.conjuagate(phi)*a1dag2
        return expm(arg)

    def test(self,op='D',vals=None,v0=None):

        if vals is None:
            vals = [0.0,0.1,0.2,0.3,0.4]

        if v0 is None:
            v0 = np.zeros(self.n_dim)
            v0[0]=1

        allowed_op = ['S','D','R']
        if op not in allowed_op:
            print('Operation \'%s\' not recognized. Must be in'%(op),allowed_op)
            return

        for v in vals:
            mat = eval('self.{}({})'.format(op,v))
            state0 = np.matmul(mat,v0)
            state0 = state0*state0.conj()
            plt.plot(state0,'-o',ms=3,lw=1,alpha=0.5,label='{}'.format(round(v,3)))

        labels = ['{:04b}'.format(i) for i in range(self.n_dim)]
        plt.xticks(range(self.n_dim),labels,rotation=80)
        plt.xlabel('Qubit')
        plt.ylabel('Probability')
        plt.legend(title=op)
        plt.title('Results for {} Qubits per Mode'.format(self.n_qubits_per_mode))
        plt.pause(0.1)

if __name__=='__main__':

    # parse some optional arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-n','--number_per_mode', help='Number of qubits per mode',choices=[1,2,3,4],nargs='?',const=2,type=int)

    args = parser.parse_args()

    if args.number_per_mode is None:
        args.number_per_mode = 2
    m = CVGates(args.number_per_mode)

    # standard init
    v0 = np.zeros(m.n_dim)
    v0[0] = 1

    # displacement
    dmat = m.D(0.1)

    new_input= np.matmul(dmat,v0)

    # build circuit
    q = QuantumRegister(name='qr',size=args.number_per_mode)
    c = ClassicalRegister(name='cr',size=args.number_per_mode)
    circ = QuantumCircuit(q,c)

    circ.initialize(new_input, q)
