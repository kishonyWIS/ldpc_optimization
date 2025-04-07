from generate_memory_circuit import MemoryExperiment
from rotated_surface_code import RotatedSurfaceCode


def test_d3_rsc_memory_experiment():
    rsc = RotatedSurfaceCode(3)
    lz = []  # todo
    qubit_coords = {**rsc.data_qubits, **rsc.ancilla_qubits}
    m_exp = MemoryExperiment(len(rsc.data_qubits)+len(rsc.ancilla_qubits),
                             rsc.x_checks,
                             rsc.z_checks,
                             lz,
                             p=0.01,
                             x_detectors=True,
                             z_detectors=True,
                             qubit_coords=qubit_coords)

    print(m_exp.circuit)


#    MemoryExperiment()
test_d3_rsc_memory_experiment()
