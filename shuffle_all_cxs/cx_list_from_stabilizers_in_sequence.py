#!/usr/bin/env python3
import numpy as np


def build_initial_cx_list(x_stabilizers, z_stabilizers):
    """
    Build an initial cx_list ordering by processing stabilizers one at a time.

    For each stabilizer in x_stabilizers and z_stabilizers, assigns a unique ancilla:
      - X stabilizers get ancilla IDs "aX0", "aX1", ... and are assumed to be of type "X".
      - Z stabilizers get ancilla IDs "aZ0", "aZ1", ... and are assumed to be of type "Z".

    Each CX gate is represented as a tuple (q, a) where:
      - q is a data qubit (an integer).
      - a is the ancilla ID (a string).

    The ordering is such that all CX gates for one stabilizer are contiguous.
    """
    cx_list = []

    # Process X stabilizers one at a time.
    for i, qubits in enumerate(x_stabilizers):
        ancilla_id = f"X{i}"
        for q in qubits:
            cx_list.append((q, ancilla_id))

    # Process Z stabilizers one at a time.
    for i, qubits in enumerate(z_stabilizers):
        ancilla_id = f"Z{i}"
        for q in qubits:
            cx_list.append((q, ancilla_id))

    return cx_list


def build_mappings(x_stabilizers, z_stabilizers, n):
    """
    Build the ancilla_type, data_mapping, and ancilla_mapping dictionaries.

    Parameters:
      x_stabilizers: List of lists, each sub-list containing data qubit indices for an X stabilizer.
      z_stabilizers: List of lists, each sub-list containing data qubit indices for a Z stabilizer.
      n: Total number of data qubits.

    Returns:
      A tuple (ancilla_type, data_mapping, ancilla_mapping).

      - ancilla_type maps each ancilla ID (e.g. "aX0", "aZ0") to its type ("X" or "Z").
      - data_mapping maps each data qubit id (assumed to be an integer) to its physical index.
      - ancilla_mapping assigns physical indices to each ancilla starting after the data qubits.
    """
    ancilla_type = {}
    ancilla_mapping = {}

    # For X stabilizers.
    for i in range(len(x_stabilizers)):
        ancilla_id = f"X{i}"
        ancilla_type[ancilla_id] = "X"
    # For Z stabilizers.
    for i in range(len(z_stabilizers)):
        ancilla_id = f"Z{i}"
        ancilla_type[ancilla_id] = "Z"

    # Data mapping: here we simply map each data qubit id to itself.
    data_mapping = {q: q for q in range(n)}

    # Ancilla mapping: assign physical indices starting after the data qubits.
    # For example, if n = 9, we start ancilla indices at 9.
    next_index = n
    for ancilla_id in sorted(ancilla_type.keys()):
        ancilla_mapping[ancilla_id] = next_index
        next_index += 1

    return ancilla_type, data_mapping, ancilla_mapping


def setup_code(z_stabilizers, x_stabilizers, n, lz):
    # Build the initial cx_list ordering.
    cx_list = build_initial_cx_list(x_stabilizers, z_stabilizers)

    # Build the ancilla_type, data_mapping, and ancilla_mapping dictionaries.
    ancilla_type, data_mapping, ancilla_mapping = build_mappings(x_stabilizers, z_stabilizers, n)

    return cx_list, ancilla_type, data_mapping, ancilla_mapping, lz

def setup_3_by_3_surface_code():
    # Define the 3x3 rotated surface code stabilizers.
    # Z stabilizers:
    z_stabilizers = [
        [0, 1],
        [1, 2, 4, 5],
        [3, 4, 6, 7],
        [7, 8]
    ]

    # X stabilizers:
    x_stabilizers = [
        [0, 1, 3, 4],
        [2, 5],
        [3, 6],
        [4, 5, 7, 8]
    ]

    # Total number of data qubits
    n = 9

    lz = np.zeros((1, n), dtype=int)
    lz[0, [0, 3, 6]] = 1

    return setup_code(z_stabilizers, x_stabilizers, n, lz)