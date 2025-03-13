from generate_memory_circuit import generate_memory_circuit

def test_unrotated_surface_code_d3():
    n_qubits = 16
    z_stabilizers = [
        [0, 1],
        [2, 3],
        [1, 2, 5, 6],
        [4, 5, 8, 9],
        [6, 7, 10, 11],
        [9, 10, 13, 14],
        [12, 13],
        [14, 15]]

    x_stabilizers = [
        [0, 1, 4, 5],
        [2, 3, 6, 7],
        [4, 8],
        [5, 6, 9, 10],
        [7, 11],
        [8, 9, 12, 13],
        [10, 11, 14, 15]]
    p = 0.01
