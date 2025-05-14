import stim


def add_noise_to_circuit(circ_in, noisy_qubits, p_idle=0.001, p_cx=0.01):
    """
    Take a Stim circuit (with operations: M, MX, R, RX, CX) and a collection of noisy qubits,
    and return a new Stim circuit with noise operations inserted.

    The algorithm is:
      - Create an empty new circuit.
      - Build a dictionary qubits_last_used = {q: 0 for q in all qubits in circ_in}.
      - Process each gate in circ_in one by one. For multi-target operations,
        split them into individual operations.
      - For each gate, compute its scheduled time as:
             t = max(qubits_last_used[q] for q in involved qubits) + 1.
      - For each involved qubit q in noisy_qubits, compute:
             idle_time = t - qubits_last_used[q] - 1.
         If idle_time > 0, add an idling DEPOLARIZE1 on that qubit.
      - Add the gate (as a single-target operation).
      - For a CX gate, if both qubits are in noisy_qubits, add a DEPOLARIZE2 gate after.
      - Update qubits_last_used for all involved qubits to t.

    Parameters:
      circ_in : stim.Circuit
          Input Stim circuit containing operations: M, MX, R, RX, CX.
      noisy_qubits : collection
          Collection of qubit identifiers that are noisy.

    Returns:
      new_circ : stim.Circuit
          New Stim circuit with noise operations interleaved.
    """
    new_circ = stim.Circuit()

    total_idling_time = 0

    # Collect all qubits present in the circuit.
    all_qubits = set()
    for op in circ_in:
        for q in op.targets_copy():
            all_qubits.add(q.value)
    qubits_last_used = {q: 0 for q in all_qubits}

    for op in circ_in:
        targets = op.targets_copy()
        args = op.gate_args_copy()
        # Split operations that have multiple targets.
        if op.name in {"R", "RX", "M", "MX"}:
            for target in targets:
                q = target.value
                t = qubits_last_used[q] + 1
                # For each noisy qubit, add idling noise for idle_time = t - last_used - 1.
                idle_time = t - qubits_last_used[q] - 1
                if q in noisy_qubits and idle_time > 0 and p_idle > 0 and qubits_last_used[q] > 0:
                    new_circ.append_operation(
                        "DEPOLARIZE1", target, idle_time*p_idle)
                    total_idling_time += idle_time
                new_circ.append_operation(op.name, target)
                qubits_last_used[q] = t
        elif op.name == "CX":
            # Assume op.targets[0] is control, and subsequent are targets.
            for i in range(0, len(targets), 2):
                q_c = targets[i].value
                q_t = targets[i + 1].value
                t = max(qubits_last_used[q_c], qubits_last_used[q_t]) + 1
                for q in [q_c, q_t]:
                    idle_time = t - qubits_last_used[q] - 1
                    if q in noisy_qubits and idle_time > 0 and p_idle > 0 and qubits_last_used[q] > 0:
                        new_circ.append_operation(
                            "DEPOLARIZE1", [q], idle_time*p_idle)
                        total_idling_time += idle_time
                # After a CX, if both qubits are noisy, add two-qubit noise.
                if q_c in noisy_qubits and q_t in noisy_qubits and p_cx > 0:
                    new_circ.append_operation("DEPOLARIZE2", [q_c, q_t], p_cx)
                new_circ.append_operation("CX", [q_c, q_t])
                qubits_last_used[q_c] = t
                qubits_last_used[q_t] = t
        elif op.name in ['DEPOLARIZE1', 'DEPOLARIZE2', 'X_ERROR', 'Z_ERROR', 'DETECTOR', 'OBSERVABLE_INCLUDE']:
            # don't count idling time for these operations, just add them to the circuit
            new_circ.append_operation(op.name, targets, args)
        else:
            raise NotImplementedError

#    print(f"Total idling time: {total_idling_time}")

    return new_circ, total_idling_time


# --- Example usage ---
if __name__ == "__main__":

    # Create a simple example circuit.
    circ_in = stim.Circuit()
    circ_in.append_operation("R", [0, 1])
    circ_in.append_operation("RX", [2])
    # CX with control 0, targets 1 and 2 (will be split)
    circ_in.append_operation("CX", [0, 1, 1, 2])
    circ_in.append_operation("M", [0, 1, 2])

    # Suppose qubits 0 and 1 are noisy.
    noisy_qubits = {0, 1, 2}

    new_circ = add_noise_to_circuit(circ_in, noisy_qubits)
    print(new_circ)
