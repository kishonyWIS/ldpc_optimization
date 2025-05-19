from cx_list_from_stabilizers_in_sequence import generate_optimally_ordered_rotated_surface_code_stabilizers


def test_generate_optimally_ordered_rotated_surface_code_stabilizers():
    """
    Test the generate_optimally_ordered_rotated_surface_code_stabilizers function.
    """
    # Define the parameters for the test
    L = 3
    num_stabilizers = 10

    # Generate the stabilizers
    stabilizers = generate_optimally_ordered_rotated_surface_code_stabilizers(
        L)

    print('x_stabilizers', stabilizers[0])
    print('z_stabilizers', stabilizers[1])


if __name__ == "__main__":
    test_generate_optimally_ordered_rotated_surface_code_stabilizers()
