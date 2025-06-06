from typing import Dict, List, Tuple
import sinter
from copy import deepcopy
import numpy as np
# from draw_ordered_tanner_graph import draw_cx_list
from .optimize_cx_list import CXGate
from .circuit_from_cx_list import memory_experiment_circuit_from_cx_list
from ldpc.sinter_decoders import SinterBpOsdDecoder
from .shuffle_full_cx_list import random_legal_local_change_inplace
from .permute_single_stabilizer import permute_single_stabilizer_inplace
import matplotlib.pyplot as plt
from .cx_list_from_stabilizers_in_sequence import StabilizerCode
import collections
import pickle


class OptimizerStatus:
    def __init__(self, cx_list: List,
                 objective_value: float,
                 objective_error: float,
                 worst_flag: int = None):
        self.cx_list = cx_list
        self.objective_value = objective_value
        self.objective_error = objective_error
        self.worst_flag = worst_flag

    def __repr__(self):
        return f"OptimizerStatus(cx_list={self.cx_list}, objective_value={self.objective_value}, objective_error={self.objective_error}, worst_flag={self.worst_flag})"

    def __eq__(self, other):
        if not isinstance(other, OptimizerStatus):
            return False
        return (self.cx_list == other.cx_list and
                self.objective_value == other.objective_value and
                self.objective_error == other.objective_error and
                self.worst_flag == other.worst_flag)


class InteractiveCxListOptimizer:
    def __init__(self,
                 code: StabilizerCode,
                 experiment_type: str,  # "X", "Z", or "both"
                 p_cx: float,
                 p_idle: float,
                 cycles_with_noise: int,
                 decoder: str = 'bposd',):
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
        self.code = code
        self.initial_cx_list = code.generate_cx_list()
        self.ancilla_type, self.data_mapping, self.ancilla_mapping, self.flag_mapping = code.build_mappings()
        self.lx = code.lx
        self.lz = code.lz
        self.experiment_type = experiment_type.upper()
        self.p_cx = p_cx
        self.p_idle = p_idle
        self.cycles_with_noise = cycles_with_noise
        self.decoder = decoder

    def save(self, filename: str):
        """
        Save the optimizer state to a file.
        """
        with open(filename, 'wb') as f:  # Use 'wb' for binary write
            pickle.dump(self, f)

    @staticmethod
    def load_optimizer(filename: str) -> 'InteractiveCxListOptimizer':
        """
        Load the optimizer state from a file.
        """
        with open(filename, 'rb') as f:  # Use 'rb' for binary read
            return pickle.load(f)

    def measure_logical_error_rate(self,
                                   cx_list: List[CXGate],
                                   max_num_shots: int,
                                   max_num_errors: int,
                                   flags: bool,
                                   num_workers: int = 10) -> OptimizerStatus:
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
                                            max_num_shots, max_num_errors, flags)

            return status
        elif self.experiment_type == "BOTH":
            # Simulate for X and Z separately.
            status_x = self._simulate_circuit(cx_list, self.lx, "X",
                                              max_num_shots, max_num_errors, flags, num_workers)
            status_z = self._simulate_circuit(cx_list, self.lz, "Z",
                                              max_num_shots, max_num_errors, flags, num_workers)
            E_x = status_x.objective_value
            E_z = status_z.objective_value
            # Combine: overall logical error rate.
            E_both = 1 - (1 - E_x) * (1 - E_z)
            # Propagate errors approximately (assuming independence).
            err_x = status_x.objective_error
            err_z = status_z.objective_error
            E_both_error = np.sqrt(((1-E_z)*err_x)**2 + ((1-E_x)*err_z)**2)

            if status_x.worst_flag is not None and status_z.worst_flag is not None:
                # Combine the worst flags.
                if err_x > err_z:
                    worst_flag = status_x.worst_flag
                else:
                    worst_flag = status_z.worst_flag
            else:
                worst_flag = None
            return OptimizerStatus(cx_list, E_both, E_both_error, worst_flag=worst_flag)
        else:
            raise ValueError("experiment_type must be 'X', 'Z', or 'both'.")

    def _simulate_circuit_with_flags(self, cx_list: List[CXGate],
                                     logicals: np.ndarray,
                                     logical_type: str,
                                     max_num_shots: int,
                                     max_num_errors: int,
                                     num_workers=10) -> OptimizerStatus:
        """
        Helper to build and simulate a circuit with flags using memory_experiment_circuit_from_cx_list.
        """
        flag_circ = memory_experiment_circuit_from_cx_list(
            cx_list=cx_list,
            ancilla_type=self.ancilla_type,
            data_mapping=self.data_mapping,
            ancilla_mapping=self.ancilla_mapping,
            flag_mapping=self.flag_mapping,
            logicals=logicals,
            logical_type=logical_type,
            p_cx=self.p_cx,
            p_idle=self.p_idle,
            number_of_cycles=self.cycles_with_noise,
            flag=True
        )

        task = sinter.Task(circuit=flag_circ)
        stats = sinter.collect(tasks=[task],
                               num_workers=num_workers,
                               max_shots=max_num_shots,
                               max_errors=max_num_errors,
                               decoders=[self.decoder],
                               custom_decoders=self.custom_decoders,
                               count_observable_error_combos=True)

        logical_error_rate, logical_error_rate_error = self.logical_error_rate_with_flags(
            n_shots=stats[0].shots,
            observable_error_combos=stats[0].custom_counts)

        worst_flag = self.find_worst_flag(
            observable_error_combos=stats[0].custom_counts)

        return OptimizerStatus(deepcopy(cx_list), logical_error_rate, logical_error_rate_error, worst_flag=worst_flag)

    def logical_error_rate_with_flags(self, n_shots, observable_error_combos: collections.Counter) -> Tuple[float, float]:
        n_logical_errors = 0
        # Vectorized: filter keys ending with 'E' and sum their counts
        n_logical_errors = sum(
            count for key, count in observable_error_combos.items() if key[-1] == 'E'
        )
        logical_error_rate = n_logical_errors / n_shots
        logical_error_rate_error = np.sqrt(
            logical_error_rate * (1 - logical_error_rate) / n_shots)

        return logical_error_rate, logical_error_rate_error

    def find_worst_flag(self, observable_error_combos: collections.Counter) -> int:

        n_flags = len(self.flag_mapping)
        n_times_flagged = [0] * n_flags

        for key, counts in observable_error_combos.items():
            # Only consider keys where the last entry is the logical error
            if key[-1] == 'E':
                flag_values = key.split('=')[-1][:-1]
                # Vectorized count: for each flag, count 'E' in its positions across cycles
                arr = np.frombuffer(flag_values.encode(),
                                    dtype='S1').reshape(-1, n_flags)
                n_times_flagged += (arr == b'E').sum(axis=0) * counts

        # for i, flag in enumerate(key.split('=')[-1][:-1]):
        #     if flag == 'E':
        #         n_times_flagged[i //
        #                         n_flags] += counts

        max_flag_index = np.argmax(n_times_flagged)

        flag_mapping = list(self.flag_mapping.keys())
        # Find the flag corresponding to the index.
        flag = flag_mapping[max_flag_index]
        return flag

    def _simulate_circuit(self, cx_list: List[CXGate],
                          logicals: np.ndarray,
                          logical_type: str,
                          max_num_shots: int,
                          max_num_errors: int,
                          flags: bool,
                          num_workers: int) -> OptimizerStatus:
        """
        Helper to build and simulate a circuit using memory_experiment_circuit_from_cx_list.
        """
        if flags == True:
            # Use the circuit with flags.
            return self._simulate_circuit_with_flags(cx_list, logicals, logical_type,
                                                     max_num_shots, max_num_errors, num_workers)
        else:
            # Use the circuit without flags.
            return self._simulate_circuit_without_flags(cx_list, logicals, logical_type,
                                                        max_num_shots, max_num_errors, num_workers)

    def _simulate_circuit_without_flags(self, cx_list: List[CXGate],
                                        logicals: np.ndarray,
                                        logical_type: str,
                                        max_num_shots: int,
                                        max_num_errors: int):

        circ = memory_experiment_circuit_from_cx_list(
            cx_list=cx_list,
            ancilla_type=self.ancilla_type,
            data_mapping=self.data_mapping,
            ancilla_mapping=self.ancilla_mapping,
            flag_mapping=self.flag_mapping,
            logicals=logicals,
            logical_type=logical_type,
            p_cx=self.p_cx,
            p_idle=self.p_idle,
            number_of_cycles=self.cycles_with_noise,
            flag=False
        )
        task = sinter.Task(circuit=circ)

        stats = sinter.collect(tasks=[task,],
                               num_workers=10,
                               max_shots=max_num_shots,
                               max_errors=max_num_errors,
                               decoders=[self.decoder],
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

    def start_optimization(self, num_shots: int, num_errors: int, flags: bool, num_workers: int = 10):

        self.best_cx_list = self.initial_cx_list
        status = self.measure_logical_error_rate(self.best_cx_list,
                                                 max_num_shots=num_shots,
                                                 max_num_errors=num_errors,
                                                 flags=flags,
                                                 num_workers=num_workers)

        self.optimizer_history.append(status)
        self.best_obj = status.objective_value
        self.best_obj_error = status.objective_error

    def run_optimization(self,
                         max_bp_iterations: int,
                         osd_order: int,
                         iterations: int,
                         max_num_shots: int,
                         max_num_errors: int,
                         flags: bool,
                         step_type: str = 'edge_pair',
                         num_workers: int = 10
                         ):
        self.custom_decoders = {
            "bposd": SinterBpOsdDecoder(
                max_iter=max_bp_iterations,
                osd_order=osd_order,
                osd_method='OSD_E'
            )
        }

        if not self.optimizer_history:
            self.start_optimization(num_shots=max_num_shots,
                                    num_errors=max_num_errors,
                                    flags=flags,
                                    num_workers=num_workers)
        for i in range(iterations):
            print(f'iteration {i}')
            candidate = deepcopy(self.best_cx_list)
            if step_type == 'edge_pair':
                changed = random_legal_local_change_inplace(
                    candidate, self.ancilla_type)
            elif step_type == 'single_stabilizer':
                if flags:
                    # If using flags, we need to permute the stabilizer block
                    # corresponding to the worst flag.

                    permute_single_stabilizer_inplace(
                        candidate, self.optimizer_history[-1].worst_flag)
                else:
                    permute_single_stabilizer_inplace(candidate)
                changed = True

            else:
                raise ValueError(
                    "step_type must be 'edge_pair' or 'single_stabilizer'")

            if not changed:
                continue
            status = self.measure_logical_error_rate(candidate,
                                                     max_num_shots=max_num_shots,
                                                     max_num_errors=max_num_errors,
                                                     flags=flags,
                                                     num_workers=num_workers)

            self.optimizer_history.append(status)
            if status.objective_value <= self.best_obj:
                self.best_cx_list = candidate

            if status.objective_value < self.best_obj:
                self.best_obj = status.objective_value
                print(f"Iteration {i}: improved objective to {self.best_obj}")
                # if draw:
                #     draw_cx_list(self.best_cx_list, self.ancilla_type,
                #                  data_coords=self.data_coords, ancilla_coords=self.ancilla_coords)

    def plot_history(self, ax: plt.Axes = None, label: str = None):
        if ax is None:
            fig, ax = plt.subplots()
        objective_list = [
            point.objective_value for point in self.optimizer_history]
        objective_error = [
            point.objective_error for point in self.optimizer_history]
        ax.errorbar(range(len(objective_list)),
                    objective_list,
                    yerr=objective_error,
                    fmt='o',
                    label=label)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Objective Value')
        if ax.get_legend() is None:
            ax.legend()
        if ax.figure is not plt.gcf():
            plt.show()
