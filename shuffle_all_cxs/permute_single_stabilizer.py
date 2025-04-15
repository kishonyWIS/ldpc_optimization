import random
from typing import List, Tuple

# A CX gate is represented as a tuple (q, ancilla)
CXGate = Tuple[int, str]


def permute_single_stabilizer_inplace(cx_list: List[CXGate], target_ancilla: str = None) -> None:
    """
    Modify cx_list in place by randomly permuting the order of gates within a single
    stabilizer block (i.e. all gates that share the same ancilla). If target_ancilla is None,
    a random ancilla is chosen from those present in cx_list.

    Parameters:
      cx_list : List[CXGate]
          A list of CX gates, where each gate is a tuple (q, ancilla).
      target_ancilla : str, optional
          The ancilla (stabilizer) identifier whose block should be permuted.
          If not provided, one is chosen at random.
    """
    # Find all ancillas present in the cx_list.
    ancillas = {a for _, a in cx_list}
    if not ancillas:
        return

    if target_ancilla is None:
        target_ancilla = random.choice(list(ancillas))

    # Get the indices in cx_list corresponding to the chosen ancilla.
    indices = [i for i, (_, a) in enumerate(cx_list) if a == target_ancilla]

    if not indices:
        return

    # Extract the sublist corresponding to this block.
    block = [cx_list[i] for i in indices]
    # Randomly permute the block.
    random.shuffle(block)
    # Put the permuted block back into the original cx_list.
    for i, idx in enumerate(indices):
        cx_list[idx] = block[i]


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

    # Permute only the block for ancilla "X1"
    permute_single_stabilizer_inplace(cx_list, target_ancilla="X1")

    print("\nAfter permuting block 'X1':")
    for gate in cx_list:
        print(gate)

    # Or, randomly choose one block and permute it.
    permute_single_stabilizer_inplace(cx_list)

    print("\nAfter randomly permuting one block:")
    for gate in cx_list:
        print(gate)
