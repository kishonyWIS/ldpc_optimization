import stim
import numpy as np
from typing import List, Tuple, Dict
from noisy_cx_circuit import add_noise_to_circuit

# A CX gate is represented as a tuple (q, a)
CXGate = Tuple[str, str]


def memory_experiment_circuit_from_cx_list(
        cx_list: List[CXGate],
        ancilla_type: Dict[str, str],
        data_mapping: Dict[str, int],
        ancilla_mapping: Dict[str, int],
        flag_mapping: Dict[str, int],
        logicals: np.ndarray,
        logical_type: str,
        p_cx: float,
        p_idle: float,
        both_detectors: bool = False,
        cycles_before_noise: int = 1,
        cycles_with_noise: int = 1,
        cycles_after_noise: int = 1,
        flag: bool = True
) -> Tuple[stim.Circuit, stim.Circuit, int]:
    """
    Build a Stim circuit from a global cx_list ordering (assumed to be the ordering for one measurement round)
    and then apply it repeatedly for multiple cycles.

    Each CX gate in cx_list is a tuple (q, a) where q is a data-qubit identifier and a is an ancilla identifier.
    The dictionary ancilla_type maps each ancilla id to its type ("X" or "Z").

    The dictionaries data_mapping and ancilla_mapping map abstract qubit identifiers to physical indices.
    In addition, flag_mapping maps each ancilla id to the physical index of its associated flag qubit.

    For each cycle:
      - For each ancilla present in cx_list, the first occurrence of a gate for that ancilla triggers an ancilla reset.
        If flag is enabled and noise is applied (p > 0), the flag qubit is prepared using the appropriate reset.
      - Each CX gate is appended. If p > 0 and noise is enabled, a DEPOLARIZE2 operation is inserted before the CX.
      - For flagging, if flag is enabled and p > 0, a flag CX is inserted immediately after the first occurrence for that ancilla.
      - After the last occurrence of a gate for a given ancilla, the flag measurement (if enabled) is inserted, followed
        by the main ancilla measurement.
      - Measurement indices (syndrome and flag) are recorded per ancilla.

    After all cycles, detector operations are appended (comparing consecutive syndrome measurements for each ancilla),
    followed by final data measurements and OBSERVABLE_INCLUDE operations for logical observables defined by lz.

    Parameters:
      cx_list: List of CX gates (tuples (q, a)) for one round.
      ancilla_type: Mapping ancilla id -> "X" or "Z".
      data_mapping: Mapping data qubit id -> physical index.
      ancilla_mapping: Mapping ancilla id -> physical index.
      flag_mapping: Mapping ancilla id -> physical index for the flag qubit.
      lz: A numpy array specifying the logical operators (each row corresponds to one logical observable).
      p: Noise parameter; if > 0, DEPOLARIZE2 operations are inserted.
      x_detectors, z_detectors: Whether to add detector operations for X or Z stabilizers.
      cycles_before_noise, cycles_with_noise, cycles_after_noise: Number of cycles in each phase.
      flag: Whether to insert flag operations.

    Returns:
      A tuple (circ, circ_without_flag_observables) where:
         - circ is the full Stim circuit including OBSERVABLE_INCLUDE operations for flag observables.
         - circ_without_flag_observables is a copy of the circuit before the OBSERVABLE_INCLUDE ops.
    """
    data_indices = [data_mapping[q] for q in sorted(data_mapping.keys())]
    n = len(data_indices)
    measurement_counter = 0
    noisy_qubits = set(data_mapping.values()) | set(ancilla_mapping.values())
    # Record measurement indices per ancilla.
    measurement_indexes = {
        'X_syndromes': {a: [] for a, t in ancilla_type.items() if t == "X"},
        'Z_syndromes': {a: [] for a, t in ancilla_type.items() if t == "Z"},
        'X_flags': {a: [] for a, t in ancilla_type.items() if t == "X"},
        'Z_flags': {a: [] for a, t in ancilla_type.items() if t == "Z"}
    }

    # Pre-calculate, for each ancilla, the first and last occurrence in cx_list.
    ancilla_positions = {}  # ancilla -> list of indices in cx_list
    for idx, (_, a) in enumerate(cx_list):
        ancilla_positions.setdefault(a, []).append(idx)
    first_occurrence = {a: min(pos_list) for a, pos_list in ancilla_positions.items()}
    last_occurrence = {a: max(pos_list) for a, pos_list in ancilla_positions.items()}

    # Define cycles: a list of booleans indicating whether noise is active.
    cycles_have_noise = ([False] * cycles_before_noise +
              [True] * cycles_with_noise +
              [False] * cycles_after_noise)

    # Process each cycle.
    all_cycles = []
    for cycle_no, noisy in enumerate(cycles_have_noise):
        # Iterate over the global cx_list.
        cycle, measurement_counter = build_syndrome_extraction_cycle(ancilla_mapping, ancilla_type, cx_list,
                                                                     data_mapping, first_occurrence, flag and noisy and (p_cx>0 or p_idle>0),
                                                                     flag_mapping,
                                                                     last_occurrence, measurement_counter, measurement_indexes)
        all_cycles.append(cycle)

    # Concatenate all cycles into a single circuit.
    circ = stim.Circuit()
    # initialize data qubits in logical basis
    circ.append_operation('R' if logical_type == "Z" else 'RX', data_indices)
    # add first noiseless cycles
    for cycle in all_cycles[:cycles_before_noise]:
        circ += cycle
    noisy_part_of_circ = stim.Circuit()
    # add noisy cycles
    for cycle in all_cycles[cycles_before_noise: cycles_before_noise+cycles_with_noise]:
        noisy_part_of_circ += cycle
    noisy_part_of_circ, idle_time = add_noise_to_circuit(noisy_part_of_circ, noisy_qubits=noisy_qubits, p_idle=p_idle, p_cx=p_cx)
    circ += noisy_part_of_circ
    # add last noiseless cycles
    for cycle in all_cycles[cycles_before_noise+cycles_with_noise:]:
        circ += cycle

    # Append detectors.
    # For each ancilla, add a detector comparing consecutive syndrome measurements.
    if logical_type == "X" or both_detectors:
        for a, meas_list in measurement_indexes['X_syndromes'].items():
            for cycle in range(len(meas_list) - 1):
                rec_targets = [stim.target_rec(meas_list[cycle] - measurement_counter),
                               stim.target_rec(meas_list[cycle + 1] - measurement_counter)]
                # Label detectors for X stabilizers (the label here is just an example).
                circ.append_operation("DETECTOR", rec_targets, [cycle, int(a[1:]), 0])
    if logical_type == "Z" or both_detectors:
        for a, meas_list in measurement_indexes['Z_syndromes'].items():
            for cycle in range(len(meas_list) - 1):
                rec_targets = [stim.target_rec(meas_list[cycle] - measurement_counter),
                               stim.target_rec(meas_list[cycle + 1] - measurement_counter)]
                circ.append_operation("DETECTOR", rec_targets, [cycle, int(a[1:]), 1])

    # Append final data measurements.
    circ.append_operation("M" if logical_type == "Z" else "MX", data_indices)
    measurement_counter += n

    # Append logical observables.
    for i_logical, logical in enumerate(logicals):
        qubits_in_logical = [i for i in range(n) if logical[i] == 1]
        circ.append_operation("OBSERVABLE_INCLUDE",
                              [stim.target_rec(i - n) for i in qubits_in_logical],
                              i_logical)

    # Make a copy of the circuit without flag OBSERVABLE_INCLUDE operations.
    circ_without_flag_observables = circ.copy()

    # add flag observables
    observable_index = logicals.shape[0]
    for i_flag, indexes_of_measurements in measurement_indexes['X_flags'].items():
        for i_cycle in range(len(indexes_of_measurements)):
            indexes = [ii - measurement_counter for ii in [indexes_of_measurements[i_cycle]]]
            circ.append_operation("OBSERVABLE_INCLUDE",
                                  list(map(stim.target_rec, indexes)),
                                  observable_index)
            observable_index += 1
    for i_flag, indexes_of_measurements in measurement_indexes['Z_flags'].items():
        for i_cycle in range(len(indexes_of_measurements)):
            indexes = [ii - measurement_counter for ii in [indexes_of_measurements[i_cycle]]]
            circ.append_operation("OBSERVABLE_INCLUDE",
                                  list(map(stim.target_rec, indexes)),
                                  observable_index)
            observable_index += 1

    return circ, circ_without_flag_observables, idle_time


def build_syndrome_extraction_cycle(ancilla_mapping, ancilla_type, cx_list, data_mapping, first_occurrence, flag,
                                    flag_mapping, last_occurrence, measurement_counter, measurement_indexes):
    cycle = stim.Circuit()
    for idx, (q, a) in enumerate(cx_list):
        # For this ancilla, determine if this is the first or last occurrence.
        first_idx = first_occurrence[a]
        last_idx = last_occurrence[a]
        # On the first occurrence, reset the ancilla and prepare the flag qubit if needed.
        if idx == first_idx:
            if ancilla_type[a] == "X":
                # For X stabilizers, reset the ancilla via RX.
                cycle.append_operation("RX", [ancilla_mapping[a]])
                if flag:
                    # Prepare the flag qubit (explicitly using flag_mapping).
                    cycle.append_operation("R", [flag_mapping[a]])
            elif ancilla_type[a] == "Z":
                # For Z stabilizers, reset the ancilla via R.
                cycle.append_operation("R", [ancilla_mapping[a]])
                if flag:
                    cycle.append_operation("RX", [flag_mapping[a]])
        # Append the CX gate.
        dq = data_mapping[q]
        aq = ancilla_mapping[a]
        if ancilla_type[a] == "X":
            # Insert a flag CX right before the last occurrence.
            if flag and idx == last_idx:
                cycle.append_operation("CX", [aq, flag_mapping[a]])
            cycle.append_operation("CX", [aq, dq])
            # Insert a flag CX right after the first occurrence.
            if flag and idx == first_idx:
                cycle.append_operation("CX", [aq, flag_mapping[a]])
        elif ancilla_type[a] == "Z":
            if flag and idx == last_idx:
                cycle.append_operation("CX", [flag_mapping[a], aq])
            cycle.append_operation("CX", [dq, aq])
            if flag and idx == first_idx:
                cycle.append_operation("CX", [flag_mapping[a], aq])
        # At the last occurrence, append flag measurement (if enabled) and then the main measurement.
        if idx == last_idx:
            if flag:
                if ancilla_type[a] == "X":
                    cycle.append_operation("M", [flag_mapping[a]])
                    measurement_indexes['X_flags'][a].append(measurement_counter)
                    measurement_counter += 1
                elif ancilla_type[a] == "Z":
                    cycle.append_operation("MX", [flag_mapping[a]])
                    measurement_indexes['Z_flags'][a].append(measurement_counter)
                    measurement_counter += 1
            if ancilla_type[a] == "X":
                cycle.append_operation("MX", [aq])
                measurement_indexes['X_syndromes'][a].append(measurement_counter)
                measurement_counter += 1
            elif ancilla_type[a] == "Z":
                cycle.append_operation("M", [aq])
                measurement_indexes['Z_syndromes'][a].append(measurement_counter)
                measurement_counter += 1
    # End of cycle.
    return cycle, measurement_counter

