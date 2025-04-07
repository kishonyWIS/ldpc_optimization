from rotated_surface_code import RotatedSurfaceCode


def test_rotated_surface_code_d3():
    d3_rsc = RotatedSurfaceCode(3)
    assert len(d3_rsc.data_qubits) == 9
    assert len(d3_rsc.ancilla_qubits) == 8
    assert len(d3_rsc.z_checks) == 4
    assert len(d3_rsc.x_checks) == 4
