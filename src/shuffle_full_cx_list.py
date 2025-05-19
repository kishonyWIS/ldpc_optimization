import random
from typing import List, Tuple, Dict

# A CX gate is represented as a tuple (q, a)
CXGate = Tuple[str, str]

def random_legal_local_change_inplace(cx_list: List[CXGate],
                                      ancilla_type: Dict[str, str]) -> bool:
    """
    Perform a random legal local change on the list of CX gates, modifying the list in place.
    The rules are:
      1) Randomly pick a location i (such that i < len(cx_list)-1). Let gate1=(q1,a1) and gate2=(q2,a2).
      2) If the two gates commute, then simply swap them.
         They commute if q1 != q2 or if a1 and a2 are of the same type.
      3) If they do not commute (i.e. q1==q2 and ancilla_type[a1] != ancilla_type[a2]),
         then try to find another data qubit q (neighbors of both a1 and a2) for which
         the ordering of the CX gates can be “adjusted” (swapped in blocks) as allowed.
         In particular, for each common qubit q (other than q1) that appears with both a1 and a2,
         if the ordering of the two edges (q,a1) and (q,a2) in the list (and any intervening edges
         for the same q) is “block ordered” (i.e. all X-type edges appear before all Z-type edges, or vice versa),
         then we bring the two edges together and perform the swap.
      4) If no such common neighbor can be found, give up on this attempt.
    Returns:
      True if a legal move was performed, False otherwise.
    """
    n = len(cx_list)
    if n < 2:
        return False

    # 1) Choose a random location (i) for a gate (ensure there's a successor).
    i = random.randint(0, n - 2)

    (q1, a1) = cx_list[i]

    # 2) Find the first subsequent edge that shares either the data qubit or the ancilla.
    j = None
    for candidate_j in range(i + 1, n):
        (qj, aj) = cx_list[candidate_j]
        if qj == q1 or aj == a1:
            j = candidate_j
            break

    if j is None:
        # No edge found that shares either q1 or a1; nothing to swap.
        return False

    # j = i+1



    (q2, a2) = cx_list[j]

    # 2) Check if the two consecutive gates commute.
    # They commute if they act on different data qubits or if the ancillas are of the same type.
    if q1 != q2 or ancilla_type[a1] == ancilla_type[a2]:
        # Legal swap: exchange positions i and i+1.
        # cx_list[i], cx_list[j] = cx_list[j], cx_list[i]
        # bring i to directly after j
        cx_list.insert(j, cx_list.pop(i))
        return True

    # 3) Otherwise, we have q1 == q2 and ancilla_type[a1] != ancilla_type[a2].
    # Build a mapping from each data qubit to a list of (position, ancilla) for all its occurrences.
    data_positions: Dict[str, List[Tuple[int, str]]] = {}
    for pos, (q, a) in enumerate(cx_list):
        data_positions.setdefault(q, []).append((pos, a))

    # Find all data qubits (other than q1) that are connected to both a1 and a2.
    common_qubits = []
    for q, pos_list in data_positions.items():
        if q == q1:
            continue
        ancillas = {a for (_, a) in pos_list}
        if a1 in ancillas and a2 in ancillas:
            common_qubits.append(q)

    # For each common qubit, try to see if we can adjust the order.
    for q in common_qubits:
        # Get all positions for (q,a1) and (q,a2)
        pos_a1 = sorted(pos for pos, a in data_positions[q] if a == a1)
        pos_a2 = sorted(pos for pos, a in data_positions[q] if a == a2)
        # Look for one pair where (q,a1) comes before (q,a2)
        for p1 in pos_a1:
            for p2 in pos_a2:
                if p1 < p2:
                    # For data qubit q, consider all edges between positions p1 and p2.
                    middle = [ (pos, a) for pos, a in data_positions[q] if p1 < pos < p2 ]
                    # Determine the required ordering.
                    # For instance, if ancilla a1 is X and a2 is Z, then we require that
                    # in the middle, all X-type edges appear before any Z-type edges.
                    if ancilla_type[a1] == 'X' and ancilla_type[a2] == 'Z':
                        needed_order = ('X', 'Z')
                    elif ancilla_type[a1] == 'Z' and ancilla_type[a2] == 'X':
                        needed_order = ('Z', 'X')
                    else:
                        continue

                    # Extract the types for edges (for qubit q) in the middle (ordered by their positions).
                    middle_sorted = sorted(middle, key=lambda x: x[0])
                    middle_types = [ancilla_type[a] for (_, a) in middle_sorted]

                    # Check that the sequence is block ordered: all needed_order[0] then all needed_order[1].
                    valid = True
                    seen_second = False
                    for t in middle_types:
                        if t == needed_order[1]:
                            seen_second = True
                        elif seen_second and t == needed_order[0]:
                            valid = False
                            break
                    if not valid:
                        continue

                    # If valid, "bring together" the edges (q,a1) and (q,a2).
                    # We remove these two edges from their current positions and reinsert them consecutively.
                    first_type_positions = [pos for pos, a in middle if ancilla_type[a] == needed_order[0]]
                    # Determine insertion index:
                    if first_type_positions:
                        # There are some intervening edges of the same type as a1.
                        # We want to reinsert immediately after the last such edge.
                        insertion_index = max(first_type_positions)
                    else:
                        # No intervening edge of type needed_order[0]: insert right after p1.
                        insertion_index = p1

                    # Remove the edges (q,a1) at p1 and (q,a2) at p2.
                    # Remove the edge at p2 first (to avoid index shifting), then p1.
                    removed_edge_q_a2 = cx_list.pop(p2)
                    removed_edge_q_a1 = cx_list.pop(p1)

                    # Reinsert the two edges in reversed order
                    cx_list.insert(insertion_index, removed_edge_q_a1)
                    cx_list.insert(insertion_index, removed_edge_q_a2)

                    # Now, swap the originally chosen consecutive gates (which are on data qubit q1).
                    # Find their current positions.
                    indices_q1 = [pos for pos, (q, a) in enumerate(cx_list) if q == q1 and a in (a1, a2)]
                    if len(indices_q1) < 2:
                        continue  # this candidate didn't work out
                    # Swap the two occurrences.
                    i1, i2 = indices_q1[0], indices_q1[1]
                    # cx_list[i1], cx_list[i2] = cx_list[i2], cx_list[i1]
                    # bring i1 to directly after i2
                    cx_list.insert(i2, cx_list.pop(i1))
                    return True

    # 4) If no valid common neighbor (and reordering) was found, give up on this attempt.
    return False


# Example usage:
if __name__ == "__main__":
    # Define a sample list of CX gates as (q, a)
    cx_list = [
        ("q1", "a1"),  # Suppose a1 is type X
        ("q1", "a2"),  # Suppose a2 is type Z
        ("q3", "a1"),
        ("q3", "a2"),
        ("q2", "a3")   # etc.
    ]
    ancilla_type = {
        "a1": "X",
        "a2": "Z",
        "a3": "X"  # for example
    }

    print("Original list:")
    for gate in cx_list:
        print(gate, ancilla_type[gate[1]])

    if random_legal_local_change_inplace(cx_list, ancilla_type):
        print("\nModified list:")
        for gate in cx_list:
            print(gate, ancilla_type[gate[1]])
    else:
        print("\nNo legal local change was found on this attempt.")
