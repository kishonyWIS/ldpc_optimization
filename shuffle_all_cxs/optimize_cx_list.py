from circuit_from_cx_list import memory_experiment_circuit_from_cx_list
from shuffle_full_cx_list import random_legal_local_change_inplace
from cx_list_from_stabilizers_in_sequence import RotatedSurfaceCode
from copy import deepcopy
from typing import List, Dict, Tuple
from draw_ordered_tanner_graph import draw_cx_list
import stim
import sinter
import numpy as np
from stimbposd import SinterDecoder_BPOSD, sinter_decoders
import matplotlib.pyplot as plt

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
    _, circ = memory_experiment_circuit_from_cx_list(
        cx_list=cx_list,
        ancilla_type=ancilla_type,
        data_mapping=data_mapping,
        ancilla_mapping=ancilla_mapping,
        flag_mapping=dict(),
        lz=lz,
        p=p,
        x_detectors=False,
        z_detectors=True,
        cycles_before_noise=1,
        cycles_with_noise=2,
        cycles_after_noise=1,
        flag=False
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


def objective_logical_error_rate(cx_list: List[CXGate],
                                 ancilla_type: Dict[str, str],
                                 data_mapping: Dict[str, int],
                                 ancilla_mapping: Dict[str, int],
                                 lz: np.ndarray,
                                 p_cx: float,
                                 p_idle: float,
                                 num_shots: int = 10000) -> float:
    """
    Build a circuit from the given cx_list and return an objective value computed as the logical error rate.
    """
    _, circ = memory_experiment_circuit_from_cx_list(
        cx_list=cx_list,
        ancilla_type=ancilla_type,
        data_mapping=data_mapping,
        ancilla_mapping=ancilla_mapping,
        flag_mapping=dict(),
        lz=lz,
        p_cx=p_cx,
        p_idle=p_idle,
        x_detectors=False,
        z_detectors=True,
        cycles_before_noise=1,
        cycles_with_noise=2,
        cycles_after_noise=1,
        flag=False
    )

    task = sinter.Task(
        circuit=circ,

    )

    stats = sinter.collect(tasks=[task], num_workers=10,
                           max_shots=10_000,
                           max_errors=100,
                           decoders=['bposd'],
                           custom_decoders=sinter_decoders(),
                           )
    for stat in stats:
        logical_error_rate = stat.errors / stat.shots
        break
    else:
        logical_error_rate = 0.0

    return logical_error_rate, circ


def optimize_cx_list(
        initial_cx_list: List[CXGate],
        ancilla_type: Dict[str, str],
        data_mapping: Dict[str, int],
        ancilla_mapping: Dict[str, int],
        p_cx: float,
        p_idle: float,
        iterations: int = 100,
        data_coords: Dict[str, Tuple[int, int]] = None,
        ancilla_coords: Dict[str, Tuple[int, int]] = None,
        num_shots: int = 10000,
) -> Tuple[List[CXGate], int]:
    """
    Optimize the cx_list by repeatedly applying a random legal change and evaluating the resulting circuit.

    The objective is to minimize the number of undetectable logical errors (as given by the length of the
    error list returned by Stim).

    Returns the best cx_list found and its objective value.
    """
    objectives_list = []

    best_cx_list = deepcopy(initial_cx_list)
    best_obj, best_circ = objective_logical_error_rate(
        best_cx_list, ancilla_type, data_mapping, ancilla_mapping, lz, p_cx, p_idle, num_shots=num_shots)
    print(f"Initial objective value: {best_obj}")
    objectives_list.append(best_obj)

    for i in range(iterations):
        candidate = deepcopy(best_cx_list)
        changed = random_legal_local_change_inplace(candidate, ancilla_type)
        if not changed:
            continue
        obj_val, _ = objective_logical_error_rate(
            candidate, ancilla_type, data_mapping, ancilla_mapping, lz, p_cx, p_idle, num_shots=num_shots)
        objectives_list.append(obj_val)
        if obj_val <= best_obj:
            best_cx_list = candidate
        if obj_val < best_obj:
            best_obj = obj_val
            print(f"Iteration {i}: improved objective to {best_obj}")
            draw_cx_list(best_cx_list, ancilla_type,
                         data_coords=data_coords, ancilla_coords=ancilla_coords)

    plt.figure()
    plt.plot(objectives_list)
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')
    plt.show()

    return best_cx_list, best_obj


if __name__ == '__main__':
    code = RotatedSurfaceCode(L=5)

    cx_list = code.generate_cx_list()
    ancilla_type, data_mapping, ancilla_mapping = code.build_mappings()
    lz = code.lz
    # Optional: custom coordinates.
    data_coords = code.data_coords
    ancilla_coords = code.ancilla_coords

    optimize_cx_list(initial_cx_list=cx_list,
                     ancilla_type=ancilla_type,
                     data_mapping=data_mapping,
                     ancilla_mapping=ancilla_mapping,
                     p_cx=0.01,
                     p_idle=0.001,
                     iterations=1000,
                     data_coords=data_coords,
                     ancilla_coords=ancilla_coords)
