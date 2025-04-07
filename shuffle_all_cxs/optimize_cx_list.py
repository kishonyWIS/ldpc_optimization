from circuit_from_cx_list import memory_experiment_circuit_from_cx_list
from shuffle_full_cx_list import random_legal_local_change_inplace
from cx_list_from_stabilizers_in_sequence import setup_3_by_3_surface_code
from copy import deepcopy
from typing import List, Dict, Tuple
from draw_ordered_tanner_graph import draw_cx_list
import stim
import numpy as np

# A CX gate is represented as a tuple (q, a)
CXGate = Tuple[str, str]

def objective(cx_list: List[CXGate],
              ancilla_type: Dict[str, str],
              data_mapping: Dict[str, int],
              ancilla_mapping: Dict[str, int],
              lz: np.ndarray,
              p: float) -> Tuple[int, stim.Circuit]:
    """
    Build a circuit from the given cx_list and return an objective value computed as the number of
    undetectable logical errors found by Stim's search_for_undetectable_logical_errors function.
    """
    circ, _ = memory_experiment_circuit_from_cx_list(
        cx_list = cx_list,
        ancilla_type = ancilla_type,
        data_mapping = data_mapping,
        ancilla_mapping = ancilla_mapping,
        flag_mapping = dict(),
        lz = lz,
        p = p,
        x_detectors = False,
        z_detectors = True,
        cycles_before_noise = 1,
        cycles_with_noise = 2,
        cycles_after_noise = 1,
        flag = False
    )

    # try:
    errors = circ.search_for_undetectable_logical_errors(
        dont_explore_edges_increasing_symptom_degree=False,
        dont_explore_detection_event_sets_with_size_above=9999,
        dont_explore_edges_with_degree_above=9999,
        canonicalize_circuit_errors=True,
    )
    # except:

    return len(errors), circ

def optimize_cx_list(
        initial_cx_list: List[CXGate],
        ancilla_type: Dict[str, str],
        data_mapping: Dict[str, int],
        ancilla_mapping: Dict[str, int],
        p: float,
        iterations: int = 1000
) -> Tuple[List[CXGate], int]:
    """
    Optimize the cx_list by repeatedly applying a random legal change and evaluating the resulting circuit.

    The objective is to minimize the number of undetectable logical errors (as given by the length of the
    error list returned by Stim).

    Returns the best cx_list found and its objective value.
    """
    best_cx_list = deepcopy(initial_cx_list)
    best_obj, best_circ = objective(best_cx_list, ancilla_type, data_mapping, ancilla_mapping, lz, p)
    print(f"Initial objective value: {best_obj}")

    for i in range(iterations):
        candidate = deepcopy(best_cx_list)
        changed = random_legal_local_change_inplace(candidate, ancilla_type)
        if not changed:
            continue
        obj_val, _ = objective(candidate, ancilla_type, data_mapping, ancilla_mapping, lz, p)
        if obj_val >= best_obj:
            best_cx_list = candidate
        if obj_val > best_obj:
            best_obj = obj_val
            print(f"Iteration {i}: improved objective to {best_obj}")
        if i % 100 == 0:
            draw_cx_list(best_cx_list, ancilla_type)
    return best_cx_list, best_obj


if __name__ == '__main__':
    cx_list, ancilla_type, data_mapping, ancilla_mapping, lz = setup_3_by_3_surface_code()
    optimize_cx_list(initial_cx_list = cx_list,
                     ancilla_type = ancilla_type,
                     data_mapping = data_mapping,
                     ancilla_mapping = ancilla_mapping,
                     p = 0.01,
                     iterations = 1000
)