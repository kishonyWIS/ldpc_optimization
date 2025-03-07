import numpy as np
from bposd.css import css_code
import stim
from stimbposd import BPOSD
from tqdm import tqdm

# Takes as input a binary square matrix A
# Returns the rank of A over the binary field F_2
def rank2(A):
    rows,n = A.shape
    X = np.identity(n,dtype=int)

    for i in range(rows):
        y = np.dot(A[i,:], X) % 2
        not_y = (y + 1) % 2
        good = X[:,np.nonzero(not_y)]
        good = good[:,0,:]
        bad = X[:, np.nonzero(y)]
        bad = bad[:,0,:]
        if bad.shape[1]>0 :
            bad = np.add(bad,  np.roll(bad, 1, axis=1) )
            bad = bad % 2
            bad = np.delete(bad, 0, axis=1)
            X = np.concatenate((good, bad), axis=1)
    # now columns of X span the binary null-space of A
    return n - X.shape[1]


def get_BB_code():
    # Parameters of a Bivariate Bicycle (BB) code
    # see Section 4 of https://arxiv.org/pdf/2308.07915.pdf for notations
    # The code is defined by a pair of polynomials
    # A and B that depends on two variables x and y such that
    # x^ell = 1
    # y^m = 1
    # A = x^{a_1} + y^{a_2} + y^{a_3}
    # B = y^{b_1} + x^{b_2} + x^{b_3}

    # # [[144,12,12]]
    # ell,m = 12,6
    # a1,a2,a3 = 3,1,2
    # b1,b2,b3 = 3,1,2

    # [[784,24,24]]
    #ell,m = 28,14
    #a1,a2,a3=26,6,8
    #b1,b2,b3=7,9,20

    # [[72,12,6]]
    # ell,m = 6,6
    # a1,a2,a3=3,1,2
    # b1,b2,b3=3,1,2


    # # Ted's code [[90,8,10]]
    # ell,m = 15,3
    # a1,a2,a3 = 9,1,2
    # b1,b2,b3 = 0,2,7

    # [[108,8,10]]
    ell,m = 9,6
    a1,a2,a3 = 3,1,2
    b1,b2,b3 = 3,1,2

    # [[288,12,18]]
    # ell,m = 12,12
    # a1,a2,a3 = 3,2,7
    # b1,b2,b3 = 3,1,2

    # code length
    n = 2*m*ell


    n2 = m*ell

    # Compute check matrices of X- and Z-checks


    # cyclic shift matrices
    I_ell = np.identity(ell,dtype=int)
    I_m = np.identity(m,dtype=int)
    I = np.identity(ell*m,dtype=int)
    x = {}
    y = {}
    for i in range(ell):
        x[i] = np.kron(np.roll(I_ell,i,axis=1),I_m)
    for i in range(m):
        y[i] = np.kron(I_ell,np.roll(I_m,i,axis=1))


    A = (x[a1] + y[a2] + y[a3]) % 2
    B = (y[b1] + x[b2] + x[b3]) % 2

    A1 = x[a1]
    A2 = y[a2]
    A3 = y[a3]
    B1 = y[b1]
    B2 = x[b2]
    B3 = x[b3]

    AT = np.transpose(A)
    BT = np.transpose(B)

    hx = np.hstack((A,B))
    hz = np.hstack((BT,AT))

    # # number of logical qubits
    # k = n - rank2(hx) - rank2(hz)
    # qcode=css_code(hx,hz)
    # print('Testing CSS code...')
    # qcode.test()
    # print('Done')
    # lz = qcode.lz
    # lx = qcode.lx
    # with open(f'lz_ell_{ell}_m_{m}_a1_{a1}_a2_{a2}_a3_{a3}_b1_{b1}_b2_{b2}_b3_{b3}.npy', 'wb') as f:
    #     np.save(f, lz.toarray())

    lz = np.load(f'lz_ell_{ell}_m_{m}_a1_{a1}_a2_{a2}_a3_{a3}_b1_{b1}_b2_{b2}_b3_{b3}.npy')
    return hx,hz,lz


def memory_experiment_circuit(n,x_stabilizers,z_stabilizers,lz,p,x_detectors=False,z_detectors=True,cycles_before_noise=1,cycles_with_noise=1,cycles_after_noise=1):
    circ = stim.Circuit()
    measurement_counter = 0
    measurement_indexes = {'X_syndromes':{i:[] for i in range(len(x_stabilizers))},
                           'Z_syndromes':{i:[] for i in range(len(z_stabilizers))},
                           'X_flags':{i:[] for i in range(len(x_stabilizers))},
                           'Z_flags':{i:[] for i in range(len(z_stabilizers))}}
    for i_cycle, noisy in enumerate([False]*cycles_before_noise + [True]*cycles_with_noise + [False]*cycles_after_noise):
        x_measurement_circ, measurement_counter = measure_x_stabilizers(x_stabilizers,n,
                                                                        p if noisy else 0,
                                                                        flag = noisy,
                                                                        measurement_indexes=measurement_indexes,
                                                                        measurement_counter=measurement_counter)
        z_measurement_circ, measurement_counter = measure_z_stabilizers(z_stabilizers,n,
                                                                        p if noisy else 0,
                                                                        flag = noisy,
                                                                        measurement_indexes=measurement_indexes,
                                                                        measurement_counter=measurement_counter)
        circ += x_measurement_circ
        circ += z_measurement_circ
    # add detectors
    if x_detectors:
        for i_stabilizer, indexes_of_measurements in measurement_indexes['X_syndromes'].items():
            for i_cycle in range(len(indexes_of_measurements) - 1):
                indexes = [ii - measurement_counter for ii in indexes_of_measurements[i_cycle:i_cycle+2]]
                circ.append_operation("DETECTOR",
                                      list(map(stim.target_rec,indexes)),
                                      [i_cycle, i_stabilizer, 0])
    if z_detectors:
        for i_stabilizer, indexes_of_measurements in measurement_indexes['Z_syndromes'].items():
            for i_cycle in range(len(indexes_of_measurements) - 1):
                indexes = [ii - measurement_counter for ii in indexes_of_measurements[i_cycle:i_cycle+2]]
                circ.append_operation("DETECTOR",
                                      list(map(stim.target_rec,indexes)),
                                      [i_cycle, i_stabilizer, 1])

    # add final measurements and logical operators
    circ.append_operation("M", [i for i in range(n)])
    measurement_counter += n
    for i_logical, logical in enumerate(lz):
        qubits_in_logical = [i for i in range(n) if logical[i] == 1]
        circ.append_operation("OBSERVABLE_INCLUDE",
                              [stim.target_rec(i - n) for i in qubits_in_logical],
                              i_logical)

    circ_without_flag_observables = circ.copy()

    # add flag observables
    observable_index = lz.shape[0]
    for i_flag, indexes_of_measurements in measurement_indexes['X_flags'].items():
        for i_cycle in range(len(indexes_of_measurements)):
            indexes = [ii - measurement_counter for ii in [indexes_of_measurements[i_cycle]]]
            circ.append_operation("OBSERVABLE_INCLUDE",
                                  list(map(stim.target_rec, indexes)),
                                  observable_index)
            observable_index += 1
    for i_flag, indexes_of_measurements in measurement_indexes['Z_flags'].items():
        for i_cycle in range(len(indexes_of_measurements)):
            indexes = [ii - measurement_counter for ii in [indexes_of_measurements[i_cycle]]]
            circ.append_operation("OBSERVABLE_INCLUDE",
                                  list(map(stim.target_rec, indexes)),
                                  observable_index)
            observable_index += 1

    return circ, circ_without_flag_observables

def measure_z_stabilizers(z_stabilizers,n,p=0,flag=False,measurement_indexes=None,measurement_counter=0):
    circ = stim.Circuit()
    for i_stab, qubits in enumerate(z_stabilizers):
        # reset the ancilla
        circ.append_operation("R", [n])
        # flag
        if flag and p>0:
            circ.append_operation("RX", [n+1])
        # apply cx gates
        for iq, q in enumerate(qubits):
            # flag
            if flag and p>0 and iq == len(qubits) - 1: # before the last
                circ.append_operation("CX", [n + 1, n])
            if p > 0:
                circ.append_operation("DEPOLARIZE2", [q, n], p)
            circ.append_operation("CX", [q,n])
            # flag
            if flag and p>0 and iq == 0: # after the first
                circ.append_operation("CX", [n + 1, n])
        # flag
        if flag and p>0:
            circ.append_operation("MX", [n+1])
            measurement_indexes['Z_flags'][i_stab].append(measurement_counter)
            measurement_counter += 1
        # measure the ancilla
        circ.append_operation("M", [n])
        measurement_indexes['Z_syndromes'][i_stab].append(measurement_counter)
        measurement_counter += 1
    return circ, measurement_counter

def measure_x_stabilizers(x_stabilizers,n,p=0,flag=False,measurement_indexes=None,measurement_counter=0):
    circ = stim.Circuit()
    for i_stab, qubits in enumerate(x_stabilizers):
        # reset the ancilla
        circ.append_operation("RX", [n])
        # flag
        if flag and p>0:
            circ.append_operation("R", [n+1])
        # apply cx gates
        for iq, q in enumerate(qubits):
            # flag
            if flag and p>0 and iq == len(qubits)-1: # before the last
                circ.append_operation("CX", [n, n + 1])
            if p > 0:
                circ.append_operation("DEPOLARIZE2", [n, q], p)
            circ.append_operation("CX", [n,q])
            # flag
            if flag and p>0 and iq == 0: # after the first
                circ.append_operation("CX", [n, n + 1])
        # flag
        if flag and p>0:
            circ.append_operation("M", [n+1])
            measurement_indexes['X_flags'][i_stab].append(measurement_counter)
            measurement_counter += 1
        # measure the ancilla
        circ.append_operation("MX", [n])
        measurement_indexes['X_syndromes'][i_stab].append(measurement_counter)
        measurement_counter += 1
    return circ, measurement_counter

def get_logical_error_rate(n,x_stabilizers,z_stabilizers,lz,p,num_shots):
    circuit, circuit_without_flag_observables = memory_experiment_circuit(n,x_stabilizers,z_stabilizers,lz,p,cycles_before_noise=1,cycles_with_noise=1,cycles_after_noise=1)

    sampler = circuit.compile_detector_sampler()
    syndromes, observables = sampler.sample(num_shots, separate_observables=True)

    flags = observables[:, lz.shape[0]:]
    observables = observables[:, :lz.shape[0]]

    decoder = BPOSD(circuit_without_flag_observables.detector_error_model(), max_bp_iters=20)

    predicted_observables = decoder.decode_batch(syndromes)
    mistakes = np.any(predicted_observables != observables, axis=1)
    num_mistakes = np.sum(mistakes)

    print(f"{num_mistakes}/{num_shots}")

    worst_flag = np.argmax(np.sum(flags[mistakes], axis=0))
    return num_mistakes/num_shots, worst_flag

import random
import matplotlib.pyplot as plt
import numpy as np

# Simulated annealing to optimize stabilizer order
def simulated_annealing(x_stabilizers, z_stabilizers, n, lz, p, num_shots, initial_temp=1.0, cooling_rate=0.95, min_temp=0.00, max_iters=100, use_flag=False):
    current_x, current_z = [list(row) for row in x_stabilizers], [list(row) for row in z_stabilizers]
    current_energy, worst_flag = get_logical_error_rate(n, current_x, current_z, lz, p, num_shots)
    best_x, best_z, best_energy = current_x, current_z, current_energy

    energies, temperature = [current_energy], initial_temp
    temperatures = [initial_temp]

    for iteration in tqdm(range(max_iters)):
        if not use_flag:
            # Choose stabilizer type and shuffle one row at random
            current, candidate = (current_x, [row[:] for row in current_x]) if random.random() < 0.5 else (current_z, [row[:] for row in current_z])
            random.shuffle(candidate[random.randint(0, len(candidate) - 1)])

        else:
            # shuffle according to the worst flag
            if worst_flag >= len(current_x):
                current, candidate = current_z, [row[:] for row in current_z]
                random.shuffle(candidate[worst_flag - len(current_x)])
            else:
                current, candidate = current_x, [row[:] for row in current_x]
                random.shuffle(candidate[worst_flag])

        # Evaluate new energy
        new_energy, worst_flag = get_logical_error_rate(n, candidate if current is current_x else current_x,
                                            candidate if current is current_z else current_z,
                                            lz, p, num_shots)

        print('worst flag:', worst_flag)

        # standard error
        sigma_current = np.sqrt(current_energy * (1 - current_energy) / num_shots)

        delta_energy = new_energy - current_energy

        # Accept based on energy difference
        if delta_energy < 0 or random.random() < np.exp(-delta_energy / (sigma_current * temperature)):
            current_energy = new_energy
            if current is current_x:
                current_x = candidate
            else:
                current_z = candidate

            if new_energy < best_energy:
                best_x, best_z, best_energy = current_x, current_z, new_energy

        energies.append(new_energy)
        temperatures.append(temperature)

        temperature *= cooling_rate
        if temperature < min_temp:
            break

    fig, ax = plt.subplots(figsize=(10, 6))
    line, = ax.plot([], label='Logical Error Rate')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Logical Error Rate')
    ax.set_title('Simulated Annealing Optimization Progress')
    ax.legend()
    ax.grid(True)
    line.set_xdata(range(len(energies)))
    line.set_ydata(energies)
    # # plot the temperature
    # plt.plot(np.array(temperatures)/10, label='Temperature/10')
    ax.relim()
    ax.autoscale_view()
    # plt.pause(0.01)
    plt.show()

    return best_x, best_z, best_energy


if __name__ == '__main__':
    hx,hz,lz = get_BB_code()
    n = hx.shape[1]
    p = 0.01#0.02
    num_shots = 30000
    x_stabilizers = [list(np.where(row)[0]) for row in hx]
    z_stabilizers = [list(np.where(row)[0]) for row in hz]


    # # surface code (rotated):
    # z_stabilizers = [
    #     [0,1],
    #     [2,3],
    #     [1,2,5,6],
    #     [4,5,8,9],
    #     [6,7,10,11],
    #     [9,10,13,14],
    #     [12,13],
    #     [14,15]]
    #
    # x_stabilizers = [
    #     [0,1,4,5],
    #     [2,3,6,7],
    #     [4,8],
    #     [5,6,9,10],
    #     [7,11],
    #     [8,9,12,13],
    #     [10,11,14,15]]
    #
    # n = 16
    # lz = np.zeros((1, n), dtype=int)
    # lz[0, [0,4,8,12]] = 1

    # 3 by 3 surface code (rotated)
    # z_stabilizers = [
    #     [0,1],
    #     [1,2,4,5],
    #     [3,4,6,7],
    #     [7,8]]
    #
    # x_stabilizers = [
    #     [0,1,3,4],
    #     [2,5],
    #     [3,6],
    #     [4,5,7,8]]
    #
    # n = 9
    # lz = np.zeros((1, n), dtype=int)
    # lz[0, [0,3,6]] = 1

    # #steane code
    # z_stabilizers = [
    #     [0, 3, 5, 6],
    #     [1, 3, 4, 6],
    #     [2, 4, 5, 6],
    # ]
    # x_stabilizers = [
    #     [0, 3, 5, 6],
    #     [1, 3, 4, 6],
    #     [2, 4, 5, 6],
    # ]
    # n=7
    # lz = np.zeros((1, n), dtype=int)
    # lz[0, :] = 1


    # # toric code (rotated):
    # z_stabilizers = [
    #     [12,13,0,1],
    #     [14,15,2,3],
    #     [1,2,5,6],
    #     [3,0,7,4],
    #     [4,5,8,9],
    #     [6,7,10,11],
    #     [9,10,13,14],
    #     [11,8,15,12]]
    #
    # x_stabilizers = [
    #     [13,14,1,2],
    #     [15,12,3,0],
    #     [0,1,4,5],
    #     [2,3,6,7],
    #     [5,6,9,10],
    #     [7,4,11,8],
    #     [8,9,12,13],
    #     [10,11,14,15]]
    # n = 16
    # lz = np.zeros((2, n), dtype=int)
    # lz[0, [0,4,8,12]] = 1
    # lz[1, [0,1,2,3]] = 1



    # logical_error_rate = get_logical_error_rate(n,x_stabilizers,z_stabilizers,lz,noise_model,num_shots)

    best_x, best_z, best_logical_error_rate = simulated_annealing(x_stabilizers, z_stabilizers, n, lz, p,
                                                                  num_shots,max_iters=50, use_flag=True,
                                                                  initial_temp=1., cooling_rate=0.95)
    print(f'Best logical error rate: {best_logical_error_rate}')
    print(f'Best X stabilizers: {best_x}')
    print(f'Best Z stabilizers: {best_z}')