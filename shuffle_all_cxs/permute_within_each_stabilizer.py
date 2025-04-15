import random
from typing import List, Tuple

# A CX gate is represented as a tuple (q, ancilla)
CXGate = Tuple[int, str]


def random_permutation_within_each_stabilizer(cx_list: List[CXGate]) -> None:
    """
    Modify cx_list in place by randomly permuting the order of gates within each block
    (i.e. all gates that share the same ancilla). The overall block order (based on the first
    appearance of each ancilla) is preserved.

    Parameters:
      cx_list : List[CXGate]
          A list of CX gates, where each gate is a tuple (q, ancilla).
    """
    # Create a dictionary mapping each ancilla to the list of indices where it appears.
    blocks = {}
    for i, gate in enumerate(cx_list):
        q, a = gate
        blocks.setdefault(a, []).append(i)

    # For each ancilla block, shuffle the gates in place.
    for a, indices in blocks.items():
        # Extract the sublist for this block.
        sublist = [cx_list[i] for i in indices]
        # Shuffle the sublist.
        random.shuffle(sublist)
        # Place the shuffled gates back into cx_list at the same indices.
        for i, idx in enumerate(indices):
            cx_list[idx] = sublist[i]


# --- Example usage ---
if __name__ == "__main__":
    cx_list = [
        (0, 'X0'),
        (1, 'X0'),
        (3, 'X0'),
        (4, 'X0'),
        (4, 'X1'),
        (5, 'X1'),
        (7, 'X1'),
        (8, 'X1'),
        (3, 'X2'),
        (6, 'X2'),
        (2, 'X3'),
        (5, 'X3'),
        (1, 'Z0'),
        (2, 'Z0'),
        (4, 'Z0'),
        (5, 'Z0'),
        (3, 'Z1'),
        (4, 'Z1'),
        (6, 'Z1'),
        (7, 'Z1'),
        (0, 'Z2'),
        (1, 'Z2'),
        (7, 'Z3'),
        (8, 'Z3')
    ]

    print("Original cx_list:")
    for gate in cx_list:
        print(gate)

    random_permutation_within_each_stabilizer(cx_list)

    print("\nPermuted cx_list (in-place):")
    for gate in cx_list:
        print(gate)
