from typing import Dict, List, Tuple
import sinter
from copy import deepcopy
import numpy as np
from draw_ordered_tanner_graph import draw_cx_list
from optimize_cx_list import CXGate
from circuit_from_cx_list import memory_experiment_circuit_from_cx_list
from ldpc.sinter_decoders import SinterBpOsdDecoder
from shuffle_full_cx_list import random_legal_local_change_inplace
from permute_single_stabilizer import permute_single_stabilizer_inplace
import matplotlib.pyplot as plt


class OptimizerStatus:
    def __init__(self, cx_list: List, objective_value: float, objective_error: float):
        self.cx_list = cx_list
        self.objective_value = objective_value
        self.objective_error = objective_error


class InteractiveCxListOptimizer:
    def __init__(self,
                 initial_cx_list: List[CXGate],
                 ancilla_type: Dict[str, str],
                 data_mapping: Dict[str, int],
                 ancilla_mapping: Dict[str, int],
                 lx: np.ndarray,
                 lz: np.ndarray,
                 experiment_type: str,  # "X", "Z", or "both"
                 p_cx: float,
                 p_idle: float,
                 cycles_with_noise: int):
        """
        Initialize the optimizer.

        Parameters:
          initial_cx_list : list of CXGate
              The starting CX list.
          ancilla_type : dict
              Mapping ancilla id -> "X" or "Z".
          data_mapping, ancilla_mapping : dict
              Mappings from abstract qubit ids to physical indices.
          lx, lz : np.ndarray
              Logical operator matrices for X and Z logicals.
          experiment_type : str
              Which logical error rate to optimize: "X", "Z", or "both".
          p_cx, p_idle : float
              Noise parameters for CX and idling noise.
          cycles_with_noise : int
              Number of cycles with noise.
        """
        self.optimizer_history: List[OptimizerStatus] = []
        self.initial_cx_list = initial_cx_list
        self.ancilla_type = ancilla_type
        self.data_mapping = data_mapping
        self.ancilla_mapping = ancilla_mapping
        self.lx = lx
        self.lz = lz
        self.experiment_type = experiment_type.upper()
        self.p_cx = p_cx
        self.p_idle = p_idle
        self.cycles_with_noise = cycles_with_noise

    def measure_logical_error_rate(self, cx_list: List[CXGate],
                                   max_num_shots: int,
                                   max_num_errors: int) -> OptimizerStatus:
        """
        Build a circuit from the given cx_list and simulate it to obtain a logical error rate.
        The circuit is built by calling memory_experiment_circuit_from_cx_list with arguments:
          - logicals: set to lx if optimizing for logical X, lz if for logical Z.
          - logical_type: "X" or "Z" accordingly.
        If experiment_type is "both", the function builds and simulates two circuits
        (one for each type) and combines the error rates as:
            E_both = 1 - (1-E_X) (1-E_Z).

        Returns:
          An OptimizerStatus containing the cx_list and the (combined) objective value.
        """
        if self.experiment_type in ["X", "Z"]:
            # Use the corresponding logical operator.
            logicals = self.lx if self.experiment_type == "X" else self.lz
            status = self._simulate_circuit(cx_list, logicals, self.experiment_type,
                                            max_num_shots, max_num_errors)
            return status
        elif self.experiment_type == "BOTH":
            # Simulate for X and Z separately.
            status_x = self._simulate_circuit(cx_list, self.lx, "X",
                                              max_num_shots, max_num_errors)
            status_z = self._simulate_circuit(cx_list, self.lz, "Z",
                                              max_num_shots, max_num_errors)
            E_x = status_x.objective_value
            E_z = status_z.objective_value
            # Combine: overall logical error rate.
            E_both = 1 - (1 - E_x) * (1 - E_z)
            # Propagate errors approximately (assuming independence).
            err_x = status_x.objective_error
            err_z = status_z.objective_error
            E_both_error = np.sqrt(( (1-E_z)*err_x )**2 + ((1-E_x)*err_z)**2)
            return OptimizerStatus(cx_list, E_both, E_both_error)
        else:
            raise ValueError("experiment_type must be 'X', 'Z', or 'both'.")

    def _simulate_circuit(self, cx_list: List[CXGate],
                          logicals: np.ndarray,
                          logical_type: str,
                          max_num_shots: int,
                          max_num_errors: int) -> OptimizerStatus:
        """
        Helper to build and simulate a circuit using memory_experiment_circuit_from_cx_list.
        """
        _, circ, idle_time = memory_experiment_circuit_from_cx_list(
            cx_list=cx_list,
            ancilla_type=self.ancilla_type,
            data_mapping=self.data_mapping,
            ancilla_mapping=self.ancilla_mapping,
            flag_mapping=dict(),  # No flag mapping used here
            logicals=logicals,
            logical_type=logical_type,
            p_cx=self.p_cx,
            p_idle=self.p_idle,
            cycles_before_noise=1,
            cycles_with_noise=self.cycles_with_noise,
            cycles_after_noise=1,
            flag=False
        )
        task = sinter.Task(circuit=circ)
        stats = sinter.collect(tasks=[task],
                               num_workers=10,
                               max_shots=max_num_shots,
                               max_errors=max_num_errors,
                               decoders=['bposd'],
                               custom_decoders=self.custom_decoders)
        for stat in stats:
            logical_error_rate = stat.errors / stat.shots
            logical_error_rate_error = np.sqrt(
                logical_error_rate * (1 - logical_error_rate) / stat.shots)
            break
        else:
            logical_error_rate = 0.0
            logical_error_rate_error = 0.0

        return OptimizerStatus(deepcopy(cx_list), logical_error_rate, logical_error_rate_error)

    def start_optimization(self, num_shots, num_errors):
        self.best_cx_list = self.initial_cx_list
        status = self.measure_logical_error_rate(self.best_cx_list,
                                                 max_num_shots=num_shots,
                                                 max_num_errors=num_errors)
        self.optimizer_history.append(status)
        self.best_obj = status.objective_value
        self.best_obj_error = status.objective_error

    def run_optimization(self,
                         max_bp_iterations: int,
                         osd_order: int,
                         iterations: int,
                         max_num_shots: int,
                         max_num_errors: int,
                         draw: bool = False,
                         step_type = 'edge_pair'):
        self.custom_decoders = {
            "bposd": SinterBpOsdDecoder(
                max_iter=max_bp_iterations,
                osd_order=osd_order,
                osd_method='OSD_E'
            )
        }

        if not self.optimizer_history:
            self.start_optimization(num_shots=max_num_shots,
                                    num_errors=max_num_errors)

        for i in range(iterations):
            candidate = deepcopy(self.best_cx_list)
            if step_type == 'edge_pair':
                changed = random_legal_local_change_inplace(candidate, self.ancilla_type)
            elif step_type == 'single_stabilizer':
                permute_single_stabilizer_inplace(candidate)
                changed = True
            else:
                raise ValueError("step_type must be 'edge_pair' or 'single_stabilizer'")

            if not changed:
                continue
            status = self.measure_logical_error_rate(candidate,
                                                     max_num_shots=max_num_shots,
                                                     max_num_errors=max_num_errors)
            self.optimizer_history.append(status)
            if status.objective_value <= self.best_obj:
                self.best_cx_list = candidate
            if status.objective_value < self.best_obj:
                self.best_obj = status.objective_value
                print(f"Iteration {i}: improved objective to {self.best_obj}")
                # if draw:
                #     draw_cx_list(self.best_cx_list, self.ancilla_type,
                #                  data_coords=self.data_coords, ancilla_coords=self.ancilla_coords)

    def plot_history(self):
        plt.figure()
        objective_list = [point.objective_value for point in self.optimizer_history]
        objective_error = [point.objective_error for point in self.optimizer_history]
        plt.errorbar(range(len(objective_list)),
                     objective_list,
                     yerr=objective_error,
                     fmt='o',
                     label='Objective Value')
        plt.xlabel('Iteration')
        plt.ylabel('Objective Value')
        plt.show()
