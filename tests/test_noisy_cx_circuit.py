from src.noisy_cx_circuit import add_noise_to_circuit
import stim


def test_add_noise_to_circuit_small_circuit():
    # Create a simple example circuit.
    circ_in = stim.Circuit()
    circ_in.append_operation("R", [0, 1])
    circ_in.append_operation("RX", [2])
    circ_in.append_operation("TICK")
    # CX with control 0, targets 1 and 2 (will be split)
    circ_in.append_operation("CX", [0, 1])
    circ_in.append_operation("TICK")
    circ_in.append_operation("CX", [1, 2])
    circ_in.append_operation("TICK")
    circ_in.append_operation("M", [0, 1, 2])

    # Suppose qubits 0 and 1 are noisy.
    noisy_qubits = {0, 1, 2}

    new_circ, idling_time = add_noise_to_circuit(
        circ_in, noisy_qubits, p_idle=0.001, p_cx=0.01)
    assert new_circ == stim.Circuit("""R 0 1
                                    RX 2
                                    TICK
                                    DEPOLARIZE2(0.01) 0 1
                                    CX 0 1
                                    TICK
                                    DEPOLARIZE1(0.001) 2
                                    DEPOLARIZE2(0.01) 1 2
                                    CX 1 2
                                    TICK
                                    DEPOLARIZE1(0.001) 0
                                    M 0 1 2""")

    assert idling_time == 2

    new_circ, idling_time = add_noise_to_circuit(
        circ_in, noisy_qubits, p_idle=0.001, p_cx=0.01, p_measurement=0.02, p_initialization=0.03)
    assert str(new_circ) == """R 0 1
DEPOLARIZE1(0.03) 0 1
RX 2
DEPOLARIZE1(0.03) 2
TICK
DEPOLARIZE2(0.01) 0 1
CX 0 1
TICK
DEPOLARIZE1(0.001) 2
DEPOLARIZE2(0.01) 1 2
CX 1 2
TICK
DEPOLARIZE1(0.001) 0
M(0.02) 0 1 2"""
