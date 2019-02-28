import sys
sys.path.append("/home/artix41/Toronto/qiskit-terra/")
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import Aer, execute

import numpy as np
import matplotlib.pyplot as plt

from CVCircuit import CVCircuit

def bose_hubard(n_layers=2, J=1, U=0.1, t=0):
    # ===== Constants =====

    n_qubits_per_mode = 3
    n_qumodes = 2
    alpha = 1
    phi = np.pi/16

    # ==== Initialize circuit =====

    qr = QuantumRegister(n_qubits_per_mode*n_qumodes)
    cr = ClassicalRegister(n_qubits_per_mode*n_qumodes)
    circuit = QuantumCircuit(qr, cr)
    cv_circuit = CVCircuit(circuit, qr, n_qubits_per_mode)

    # ==== Build circuit ====

    cv_circuit.initialize([0,0])

    for layer in range(n_layers):
        phi = -1j * J * t / n_layers
        r = -U * t / (2*n_layers)
        cv_circuit.BSGate(phi, (0,1))
        cv_circuit.KGate(r, 0)
        cv_circuit.KGate(-r, 1)
        cv_circuit.RGate(r, 0)
        cv_circuit.RGate(-r, 1)

    print(circuit)

    # ==== Compilation =====

    backend = Aer.get_backend('statevector_simulator')
    job = execute(circuit, backend)
    result = job.result()
    state = job.result().get_statevector(circuit)

    # ==== Tests ====

    print(state)

    prob = np.abs(state)**2
    plt.plot(prob,'-o')
    plt.show()
        
if __name__ == '__main__':
    bose_hubard(t=1)
# def BHM(self,tmax,k,J=1,U=0.1):
#     #Implement simple two-mode Bose-Hubbard simulation
#     j = np.complex(0,1)
#     BKR = {}
#     for t in np.linspace(0,tmax,10):
#         for layer in range(k):
#             phi = -j*J*t/k
#             r = -U*t/(2*k)
#             BS = self.BS(phi)
#             K = np.kron(self.mode.K(r),self.mode.K(r))
#             R = np.kron(self.mode.R(-r),self.mode.R(-r))
#             BK = np.matmul(BS,K)
#             BKR[t] = np.matmul(BK,R)
#     return BKR
