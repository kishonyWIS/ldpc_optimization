# create surface code

from src.interactive_cx_list_optimizer import InteractiveCxListOptimizer
from src.cx_list_from_stabilizers_in_sequence import RotatedSurfaceCode
import cProfile
import pstats


def test_run_optimization():
    code = RotatedSurfaceCode(L=3)

    optimizer = InteractiveCxListOptimizer(
        code,
        experiment_type="both",
        p_cx=0.01,
        p_idle=0,
        cycles_with_noise=3,
        decoder='pymatching'
    )

    optimizer.run_optimization(max_bp_iterations=10,
                               osd_order=5,
                               iterations=2,
                               max_num_shots=100000,
                               max_num_errors=1000,
                               flags=True,
                               step_type='single_stabilizer',
                               num_workers=1)

    assert len(optimizer.optimizer_history) == 3


def test_save_and_open():
    code = RotatedSurfaceCode(L=3)

    optimizer_to_save = InteractiveCxListOptimizer(
        code,
        experiment_type="both",
        p_cx=0.01,
        p_idle=0,
        cycles_with_noise=3,
        decoder='pymatching'
    )

    optimizer_to_save.run_optimization(max_bp_iterations=10,
                                       osd_order=5,
                                       iterations=1,
                                       max_num_shots=100000,
                                       max_num_errors=1000,
                                       flags=True,
                                       step_type='single_stabilizer',
                                       num_workers=1)
    optimizer_to_save.save('test_save_output.json')

    loaded_optimizer = InteractiveCxListOptimizer.load_optimizer(
        'test_save_output.json')

    assert loaded_optimizer.optimizer_history == optimizer_to_save.optimizer_history
