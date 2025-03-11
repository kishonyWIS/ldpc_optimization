from rotated_surface_code import RotatedSurfaceCode


def test_rotated_surface_code_d3():
    d3_rsc = RotatedSurfaceCode(3)
    print(d3_rsc.data_qubits)
    print(d3_rsc.ancilla_qubits)
    print(d3_rsc.z_checks)
    print(d3_rsc.x_checks)


test_rotated_surface_code_d3()
