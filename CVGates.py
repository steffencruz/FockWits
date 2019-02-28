import os
import sys
sys.path.append("/home/artix41/Toronto/qiskit-terra/")

import matplotlib.pyplot as plt
import numpy as np
from qiskit.tools.visualization._circuit_visualization import circuit_drawer
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import Aer, execute, compile


from FockWit import QMode

def DGate(circuit, qr, cv, qumode, alpha):
    # U = np.identity(4)
    U = cv.S(alpha)
    # print(U@U.T.conj())
    circuit.unitary(U, qr[qumode], qr[qumode+1], qr[qumode+2])

if __name__ == '__main__':
    # ===== Constants =====

    n_qubits_per_mode = 3
    n_qumodes = 1
    alpha = 0.3

    # ==== Initialize circuit =====

    qr = QuantumRegister(n_qubits_per_mode*n_qumodes)
    cr = ClassicalRegister(n_qubits_per_mode*n_qumodes)
    circuit = QuantumCircuit(qr, cr)

    # ==== Build circuit ====

    cv = QMode(n_qubits_per_mode=n_qubits_per_mode)
    DGate(circuit, qr, cv, 0, alpha)
    # circuit.measure(qr, cr)
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