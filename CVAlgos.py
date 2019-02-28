from CVCircuit import CVCircuit
import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from ops import Interferometer

def gbs(z, U, n_qubits_per_mode=2, n_qumodes=4):
    qr = QuantumRegister(n_qubits_per_mode*n_qumodes)
    cr = ClassicalRegister(n_qubits_per_mode*n_qumodes)
    circuit = QuantumCircuit(qr, cr)

    cv_circuit = CVCircuit(circuit, qr, n_qubits_per_mode)

    interferometer = Interferometer(U)

    for n in range(4):
        cv_circuit.SGate(z, n)

    for n, m, theta, phi, N in interferometer.BS1:
        if np.round(phi, 13) != 0:
            cv_circuit.RGate(phi, n)
        if np.round(theta, 13) != 0:
            cv_circuit.BSGate(theta, [n, m])

    for n, expphi in enumerate(interferometer.R):
        if np.round(expphi, 13) != 1.0:
            q = log(expphi).imag
            cv_circuit.RGate(q, n)

    for n, m, theta, phi, N in reversed(interferometer.BS2):
        if np.round(theta, 13) != 0:
            cv_circuit.BSgate(-theta, [n, m])
        if np.round(phi, 13) != 0:
            cv_circuit.RGate(-phi, n)

    result = []
    for i in range(4):
        result.append(cv_circuit.Measure(i))

    return result


if __name__ == '__main__':
    U = np.array([
        [0.219546940711 - 0.256534554457j, 0.611076853957 + 0.524178937791j,
         -0.102700187435 + 0.474478834685j, -0.027250232925 + 0.03729094623j],
        [0.451281863394 + 0.602582912475j, 0.456952590016 + 0.01230749109j,
         0.131625867435 - 0.450417744715j, 0.035283194078 - 0.053244267184j],
        [0.038710094355 + 0.492715562066j, -0.019212744068 - 0.321842852355j,
         -0.240776471286 + 0.524432833034j, -0.458388143039 + 0.329633367819j],
        [-0.156619083736 + 0.224568570065j, 0.109992223305 - 0.163750223027j,
         -0.421179844245 + 0.183644837982j, 0.818769184612 + 0.068015658737j]
    ])



    """print(TiUs)
    print(TUs)
    print(phis)"""

    gbs(1, U)

