def schedule_gates(cx_list):
    """
    Schedule a list of CX gates (each represented as a tuple (q, a)) and return a dictionary
    mapping each distinct gate (q, a) to its scheduled time. Gates that can act simultaneously
    (i.e. receive the same time) are collapsed into a single entry.

    The algorithm works as follows:
      - Initialize a dictionary `qubits_last_used` with each qubit (data or ancilla) set to 0.
      - For each gate in cx_list (in order), compute its time as:
            time = max(qubits_last_used[q], qubits_last_used[a]) + 1
      - If the gate (q, a) is already in the schedule with the same time, skip adding a duplicate.
      - Otherwise, record the gate with its time and update qubits_last_used for both q and a.

    Parameters:
      cx_list : list of tuple
          A list of CX gates, where each gate is a tuple (q, a) indicating the data qubit and the ancilla.

    Returns:
      schedule : dict
          Dictionary where keys are tuples (q, a) and values are the time at which the CX gate is scheduled.
    """
    # Collect all qubits that appear in the cx_list.
    all_qubits = set()
    for q, a in cx_list:
        all_qubits.add(q)
        all_qubits.add(a)
    qubits_last_used = {q: 0 for q in all_qubits}

    schedule = {}
    for gate in cx_list:
        q, a = gate
        t = max(qubits_last_used[q], qubits_last_used[a]) + 1
        schedule[gate] = t
        # Update the last used time for both qubits.
        qubits_last_used[q] = t
        qubits_last_used[a] = t
    return schedule