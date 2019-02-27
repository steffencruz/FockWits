from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from FockWit import QMode

def DGate(circuit, qr, cv, qumode, alpha):
    return circuit.unitary(cv.D(alpha), qr[qumode], qr[qumode+1])

if __name__ == '__main__':
    qr = QuantumRegister(2)
    circuit = QuantumCircuit(qr)

    cv = QMode()
    alpha = 1
    DGate(circuit, qr, cv, 0, alpha)