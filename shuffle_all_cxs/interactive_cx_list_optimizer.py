
from typing import Dict, List
import sinter
from copy import deepcopy
import numpy as np
from draw_ordered_tanner_graph import draw_cx_list
from optimize_cx_list import CXGate
from circuit_from_cx_list import memory_experiment_circuit_from_cx_list
from ldpc.sinter_decoders import SinterBpOsdDecoder
from shuffle_full_cx_list import random_legal_local_change_inplace
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
                 lz: np.ndarray,
                 p_cx: float,
                 p_idle: float,
                 cycles_with_noise: int):

        self.optimizer_history: List[OptimizerStatus] = []
        self.initial_cx_list = initial_cx_list
        self.ancilla_type = ancilla_type
        self.data_mapping = data_mapping
        self.ancilla_mapping = ancilla_mapping
        self.lz = lz

        # setting these here because this should not change while optimizing
        self.p_cx = p_cx
        self.p_idle = p_idle
        self.cycles_with_noise = cycles_with_noise

    def measure_logical_error_rate(self, cx_list: List[CXGate],
                                   max_num_shots: int,
                                   max_num_errors: int):
        """
        Build a circuit from the given cx_list and return an objective value computed as the logical error rate.
        """
        _, circ = memory_experiment_circuit_from_cx_list(
            cx_list=cx_list,
            ancilla_type=self.ancilla_type,
            data_mapping=self.data_mapping,
            ancilla_mapping=self.ancilla_mapping,
            flag_mapping=dict(),
            lz=self.lz,
            p_cx=self.p_cx,
            p_idle=self.p_idle,
            x_detectors=False,
            z_detectors=True,
            cycles_before_noise=1,
            cycles_with_noise=self.cycles_with_noise,
            cycles_after_noise=1,
            flag=False
        )
        # print(len(circ.search_for_undetectable_logical_errors(
        #     dont_explore_detection_event_sets_with_size_above=4,
        #     dont_explore_edges_with_degree_above=4,
        #     dont_explore_edges_increasing_symptom_degree=True,
        # )), 'distance')
        task = sinter.Task(
            circuit=circ,
        )

        stats = sinter.collect(tasks=[task], num_workers=10,
                               max_shots=max_num_shots,
                               max_errors=max_num_errors,
                               decoders=['bposd'],
                               custom_decoders=self.custom_decoders,
                               )
        for stat in stats:
            logical_error_rate = stat.errors / stat.shots
            logical_error_rate_error = np.sqrt(
                logical_error_rate * (1 - logical_error_rate) / stat.shots)
            break
        else:
            logical_error_rate = 0.0
        return OptimizerStatus(
            cx_list=cx_list,
            objective_value=logical_error_rate,
            objective_error=logical_error_rate_error
        )

    def start_optimization(self, num_shots, num_errors):
        self.best_cx_list = self.initial_cx_list
        self.optimizer_history.append(self.measure_logical_error_rate(self.best_cx_list,
                                                                      max_num_shots=num_shots,
                                                                      max_num_errors=num_errors))
        self.best_obj = self.optimizer_history[-1].objective_value
        self.best_obj_error = self.optimizer_history[-1].objective_error

    def run_optimization(self,
                         max_bp_iterations: int,
                         osd_order: int,
                         iterations: int,
                         max_num_shots: int,
                         max_num_errors: int,
                         draw: bool = False):
        self.custom_decoders = {
            "bposd": SinterBpOsdDecoder(
                max_iter=max_bp_iterations,
                osd_order=osd_order,
                osd_method='OSD_E'
            )}

        if not self.optimizer_history:
            self.start_optimization(num_shots=max_num_shots,
                                    num_errors=max_num_errors)

        for i in range(iterations):
            candidate = deepcopy(self.best_cx_list)
            changed = random_legal_local_change_inplace(
                candidate, self.ancilla_type)
            if not changed:
                continue
            self.optimizer_history.append(self.measure_logical_error_rate(
                candidate, max_num_shots=max_num_shots, max_num_errors=max_num_errors))
            if self.optimizer_history[-1].objective_value <= self.best_obj:
                self.best_cx_list = candidate
            if self.optimizer_history[-1].objective_value < self.best_obj:
                self.best_obj = self.optimizer_history[-1].objective_value
                print(f"Iteration {i}: improved objective to {self.best_obj}")
#                if draw == True:

 #                   draw_cx_list(self.best_cx_list, self.ancilla_type,
  #                               data_coords=self.data_coords, ancilla_coords=self.ancilla_coords)

            # keep the option to run it as a script directly

    def plot_history(self):
        plt.figure()
        objective_list = [
            point.objective_value for point in self.optimizer_history]
        objective_error = [
            point.objective_error for point in self.optimizer_history]
        plt.errorbar(range(len(objective_list)),
                     objective_list,
                     yerr=objective_error,
                     fmt='o',
                     label='Objective Value')

        plt.xlabel('Iteration')
        plt.ylabel('Objective Value')
        plt.show()
