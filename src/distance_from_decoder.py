import stim
import numpy as np
import random
from itertools import product
from ldpc.ckt_noise.dem_matrices import detector_error_model_to_check_matrices
from ldpc.bposd_decoder import BpOsdDecoder


LOGICAL_TAG = "Logical"
def find_distance(circuit: stim.Circuit, max_shots = 2**10, max_bp_iterations=10, osd_order=5):

    new_circuit = stim.Circuit()

    for op in circuit:
        if op.name == "OBSERVABLE_INCLUDE":
            new_circuit.append(stim.CircuitInstruction("DETECTOR", op.targets_copy(), tag = LOGICAL_TAG))
        else:
            new_circuit.append(op)

    logical_locations = []
    dem = new_circuit.detector_error_model()
    k = 0
    for line in str(dem).split("\n"):
        if LOGICAL_TAG in line:
            logical_locations.append(int(line.split('D')[-1]))
            k += 1
    assert k == circuit.num_observables

    matrices = detector_error_model_to_check_matrices(
        dem, allow_undecomposed_hyperedges=True
    )
    decoder = BpOsdDecoder(
        matrices.check_matrix,
        error_channel=list(matrices.priors),
        max_iter=max_bp_iterations,
        bp_method="ms",
        ms_scaling_factor=0.625,
        schedule="parallel",
        omp_thread_count=1,
        serial_schedule_order=None,
        osd_order=osd_order,
        osd_method='OSD_E'
    )

    if 2**k > max_shots:
        logical_flipped_configurations = (random.choices((0, 1), k=k) for _ in range(max_shots))
    else:
        logical_flipped_configurations = product((0, 1), repeat=k)

    distance = np.inf
    minimal_error = None
    for logicals_flipped in logical_flipped_configurations:
        if sum(logicals_flipped) == 0: continue
        syndromes = np.zeros(dem.num_detectors)
        syndromes[logical_locations] = logicals_flipped
        error = decoder.decode(syndromes)
        error_weight = sum(error)
        if not all((matrices.check_matrix @ error) % 2 == syndromes):
            raise Exception
            continue
        if error_weight < distance:
            distance = error_weight
            minimal_error = error

    error_tags = [dem[i].tag for i in np.where(minimal_error)[0]]

    error_explanations = []
    for i in np.where(minimal_error)[0]:
        new_dem = stim.DetectorErrorModel()
        new_dem.append(dem[i])
        explanation = new_circuit.explain_detector_error_model_errors(dem_filter=new_dem)[0]
        error_explanations.append(explanation)

    return distance, minimal_error, error_tags, error_explanations

if __name__ == '__main__':
    d = 5
    p = 1e-3
    circuit = stim.Circuit.generated(
        rounds=d,
        distance=d,
        # after_clifford_depolarization=p,
        # after_reset_flip_probability=p,
        # before_measure_flip_probability=p,
        before_round_data_depolarization=p,
        code_task=f'surface_code:rotated_memory_x',
    )

    print(find_distance(circuit))