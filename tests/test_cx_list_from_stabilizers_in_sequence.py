from src.cx_list_from_stabilizers_in_sequence import generate_optimally_ordered_rotated_surface_code_stabilizers


def test_generate_optimally_ordered_rotated_surface_code_stabilizers():
    """
    Test the generate_optimally_ordered_rotated_surface_code_stabilizers function.
    """
    # Define the parameters for the test
    L = 3

    # Generate the stabilizers
    x_stabilizers, z_stabilizers = generate_optimally_ordered_rotated_surface_code_stabilizers(
        L)
    assert x_stabilizers == [[3, 0, 4, 1], [7, 4, 8, 5], [6, 3], [2, 5]]
    assert z_stabilizers == [[2, 1, 5, 4], [4, 3, 7, 6], [1, 0], [8, 7]]
