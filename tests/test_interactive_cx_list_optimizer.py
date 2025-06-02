# create surface code

from src.interactive_cx_list_optimizer import InteractiveCxListOptimizer
from src.cx_list_from_stabilizers_in_sequence import RotatedSurfaceCode
# import draw_ordered_tanner_graph

if __name__ == "__main__":
    code = RotatedSurfaceCode(L=5)
    ancilla_type, data_mapping, ancilla_mapping, flag_mapping = code.build_mappings()

    optimizer = InteractiveCxListOptimizer(
        code,
        experiment_type="both",
        p_cx=0.01,
        p_idle=0,
        cycles_with_noise=5,
    )

    optimizer.run_optimization(max_bp_iterations=10,
                               osd_order=5,
                               iterations=100,
                               max_num_shots=10000,
                               max_num_errors=1000,
                               flags=True,
                               draw=False,
                               step_type='single_stabilizer')


# write a test where each CNOT order is optimal except for one.
