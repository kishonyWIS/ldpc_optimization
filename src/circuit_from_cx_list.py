import stim
import numpy as np
from typing import List, Tuple, Dict
from .noisy_cx_circuit import add_noise_to_circuit

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
        number_of_cycles: int = 1,
        flag: bool = True,
        p_phenomenological_error: float = 0.0,
        p_measurement_error: float = 0.0,
        hook_errors={}  # ancilla_id: [(after_cx_number, p_error),...]
) -> stim.Circuit:
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
      number_of_cycles: Number of cycles.
      flag: Whether to insert flag operations.
      p_phenomenological_error: Probability of phenomenological errors before syndrome extraction on data qubits.

    Returns:
      A tuple (circ, circ_without_flag_observables) where:
         - circ is the full Stim circuit including OBSERVABLE_INCLUDE operations for flag observables.
         - circ_without_flag_observables is a copy of the circuit before the OBSERVABLE_INCLUDE ops.
    """
    data_indices = [data_mapping[q] for q in sorted(data_mapping.keys())]
    n = len(data_indices)
    noisy_qubits = set(data_mapping.values()) | set(ancilla_mapping.values())
    # Record measurement indices per ancilla.
    measurement_indexes = {
        'X_syndromes': {a: [] for a, t in ancilla_type.items() if t == "X"},
        'Z_syndromes': {a: [] for a, t in ancilla_type.items() if t == "Z"},
        'X_flags': {a: [] for a, t in ancilla_type.items() if t == "X"},
        'Z_flags': {a: [] for a, t in ancilla_type.items() if t == "Z"}
    }

    # Pre-calculate, for each ancilla, the first and last occurrence in cx_list.
    ancilla_positions_in_cx_list = {}  # ancilla -> list of indices in cx_list
    for idx, (_, a) in enumerate(cx_list):
        ancilla_positions_in_cx_list.setdefault(a, []).append(idx)
    # sort the positions
    for a in ancilla_positions_in_cx_list.keys():
        ancilla_positions_in_cx_list[a].sort()

    circ = stim.Circuit()

    # initialize data qubits in logical basis
    circ.append_operation('R' if logical_type == "Z" else 'RX', data_indices)

    def add_detectors(circuit_to_add_detectors, m_counter, cycle):
        # Append detectors.
        # For each ancilla, add a detector comparing consecutive syndrome measurements.
        if logical_type == "X" or both_detectors:
            for a, meas_list in measurement_indexes['X_syndromes'].items():
                rec_targets = [stim.target_rec(meas_list[cycle] - m_counter),
                               stim.target_rec(meas_list[cycle + 1] - m_counter)]
                # Label detectors for X stabilizers (the label here is just an example).

                circuit_to_add_detectors.append_operation("DETECTOR", rec_targets, [
                    cycle, int(a[1:]), 0])

        if logical_type == "Z" or both_detectors:
            for a, meas_list in measurement_indexes['Z_syndromes'].items():
                rec_targets = [stim.target_rec(meas_list[cycle] - m_counter),
                               stim.target_rec(meas_list[cycle + 1] - m_counter)]
                circuit_to_add_detectors.append_operation("DETECTOR", rec_targets, [
                    cycle, int(a[1:]), 1])

        return circuit_to_add_detectors

    def create_circuit_layer(flags: bool, noise: bool, detectors: bool, cycle: int = 0, m_counter=0):

        noiseless_circ, m_counter = build_syndrome_extraction_cycle(ancilla_mapping,
                                                                    ancilla_type,
                                                                    cx_list,
                                                                    data_mapping,
                                                                    ancilla_positions_in_cx_list,
                                                                    flags,
                                                                    flag_mapping,
                                                                    m_counter,
                                                                    measurement_indexes,
                                                                    p_phenomenological_error=p_phenomenological_error if noise else 0,
                                                                    p_measurement_error=0,
                                                                    hook_errors=hook_errors if noise else {})
        if detectors:
            noiseless_circ = add_detectors(
                noiseless_circ, m_counter, cycle)

        if noise == True:
            noisy_circ, idle_time = add_noise_to_circuit(
                noiseless_circ, noisy_qubits=noisy_qubits, p_idle=p_idle, p_cx=p_cx)
            return noisy_circ, m_counter
        else:
            return noiseless_circ, m_counter

    measurement_counter = 0

    noiseless_circ, measurement_counter = create_circuit_layer(
        flags=flag, noise=False, detectors=False, m_counter=measurement_counter)
    circ += noiseless_circ

    noisy_circ, measurement_counter = create_circuit_layer(
        flags=flag, noise=True, detectors=True, cycle=0, m_counter=measurement_counter)

    circ += number_of_cycles * noisy_circ

    if flag:
        observable_index = 0
        for flag in flag_mapping.keys():
            flag_type = flag[0]
            indexes_of_measurements = measurement_indexes[f"{flag_type}_flags"][flag]
            indexes = [
                ii - measurement_counter for ii in [indexes_of_measurements[1]]]

            for cycle_index in range(0, number_of_cycles):
                circ.append_operation("OBSERVABLE_INCLUDE",
                                      list(
                                          map(stim.target_rec, [
                                              index-cycle_index*(measurement_counter//2) for index in indexes])),
                                      observable_index)
                observable_index += 1

    noiseless_circ, measurement_counter = create_circuit_layer(
        flags=False, noise=False, detectors=True, m_counter=measurement_counter, cycle=1)

    circ += noiseless_circ
    # Append final data measurements.
    circ.append_operation("M" if logical_type == "Z" else "MX", data_indices)
    measurement_counter += n

    # Append logical observables.
    for i_logical, logical in enumerate(logicals):
        qubits_in_logical = [i for i in range(n) if logical[i] == 1]
        circ.append_operation("OBSERVABLE_INCLUDE",
                              [stim.target_rec(i - n)
                               for i in qubits_in_logical],
                              i_logical + circ.num_observables)

    return circ


def build_syndrome_extraction_cycle(ancilla_mapping,
                                    ancilla_type,
                                    cx_list,
                                    data_mapping,
                                    ancilla_positions_in_cx_list,
                                    flag,
                                    flag_mapping,
                                    measurement_counter,
                                    measurement_indexes,
                                    p_phenomenological_error: float = 0.0,
                                    p_measurement_error: float = 0.0,
                                    # ancilla_id: (after_cx_number, p_hook_error)
                                    hook_errors={}
                                    ):
    cycle = stim.Circuit()
    data_indices = [data_mapping[q] for q in sorted(data_mapping.keys())]
    if p_phenomenological_error > 0:
        cycle.append_operation(
            "DEPOLARIZE1", data_indices, p_phenomenological_error)

    for idx, (q, a) in enumerate(cx_list):
        # For this ancilla, determine if this is the first or last occurrence.
        first_idx = min(ancilla_positions_in_cx_list[a])
        last_idx = max(ancilla_positions_in_cx_list[a])
        hook_idx_to_p = {}
        if a in hook_errors:
            hook_idx_to_p = {}
            for hook_after_cxs_number, p_hook_error in hook_errors[a]:
                hook_idx_to_p[ancilla_positions_in_cx_list[a]
                              [hook_after_cxs_number]] = p_hook_error
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
        # insert hook error on ancilla if on the right position
        p_hook_error = hook_idx_to_p.get(idx, 0)
        if p_hook_error > 0:
            cycle.append_operation("DEPOLARIZE1", [aq], p_hook_error)
        # At the last occurrence, append flag measurement (if enabled) and then the main measurement.
        if idx == last_idx:
            if flag:
                if ancilla_type[a] == "X":
                    cycle.append_operation("M", [flag_mapping[a]])
                    measurement_indexes['X_flags'][a].append(
                        measurement_counter)
                    measurement_counter += 1
                elif ancilla_type[a] == "Z":
                    cycle.append_operation("MX", [flag_mapping[a]])
                    measurement_indexes['Z_flags'][a].append(
                        measurement_counter)
                    measurement_counter += 1
            if ancilla_type[a] == "X":
                if p_measurement_error > 0:
                    cycle.append_operation(
                        "Z_ERROR", [aq], p_measurement_error)
                cycle.append_operation("MX", [aq])
                measurement_indexes['X_syndromes'][a].append(
                    measurement_counter)
                measurement_counter += 1
            elif ancilla_type[a] == "Z":
                if p_measurement_error > 0:
                    cycle.append_operation(
                        "X_ERROR", [aq], p_measurement_error)
                cycle.append_operation("M", [aq])
                measurement_indexes['Z_syndromes'][a].append(
                    measurement_counter)
                measurement_counter += 1
    # End of cycle.
    return cycle, measurement_counter


if __name__ == "__main__":
    import numpy as np
    import stim
    from cx_list_from_stabilizers_in_sequence import StabilizerCode, RotatedSurfaceCode

    from quits.qldpc_code import *
    from quits.circuit import get_qldpc_mem_circuit
    from quits.decoder import sliding_window_bposd_circuit_mem
    from quits.simulation import get_stim_mem_result
    """
    lift_size, factor = 15, 3
    p1 = [0, 1, 5]   # e + x + x^5
    p2 = [0, 2, 7]   # e + x^2 + x^7
    code = BpcCode(p1, p2, lift_size, factor)  # Define the BpcCode object
    # Build the Tanner graph and assign directions to its edges.
    code.build_graph(seed=1)

    num_zcheck, num_data = code.hz.shape
    num_xcheck, num_data = code.hx.shape
    num_logical = code.lz.shape[0]
    depth = sum(list(code.num_colors.values()))

    stabilizer_x = []
    for i in range(code.hx.shape[0]):
        row = []
        for j in range(code.hx.shape[1]):
            if code.hx[i, j] == 1:
                row.append(j)
        stabilizer_x.append(row)

    stabilizer_z = []
    for i in range(code.hz.shape[0]):
        row = []
        for j in range(code.hz.shape[1]):
            if code.hz[i, j] == 1:
                row.append(j)
        stabilizer_z.append(row)

    bpc_code = StabilizerCode(stabilizer_x, stabilizer_z, code.lx, code.lz)
    cx_list = bpc_code.generate_cx_list()
    ancilla_type, data_mapping, ancilla_mapping, flag_mapping = bpc_code.build_mappings()
    """

    code = RotatedSurfaceCode(L=3)

    cx_list = code.generate_cx_list()
    ancilla_type, data_mapping, ancilla_mapping, flag_mapping = code.build_mappings()
    lz = code.lz
    lx = code.lx
    data_coords = code.data_coords
    ancilla_coords = code.ancilla_coords

    flag_circ = memory_experiment_circuit_from_cx_list(
        cx_list=cx_list,
        ancilla_type=ancilla_type,
        data_mapping=data_mapping,
        ancilla_mapping=ancilla_mapping,
        flag_mapping=flag_mapping,
        logicals=code.lz,
        logical_type='Z',
        p_cx=0.01,
        p_idle=0,
        number_of_cycles=8,
        flag=True
    )

    # write test for distance

    print(flag_circ)
