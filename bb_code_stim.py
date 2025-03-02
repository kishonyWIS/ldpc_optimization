import numpy as np
from bposd.css import css_code
import stim
from noise import NoiseModel
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

    # [[144,12,12]]
    ell,m = 12,6
    a1,a2,a3 = 3,1,2
    b1,b2,b3 = 3,1,2

    # [[784,24,24]]
    #ell,m = 28,14
    #a1,a2,a3=26,6,8
    #b1,b2,b3=7,9,20

    # [[72,12,6]]
    #ell,m = 6,6
    #a1,a2,a3=3,1,2
    #b1,b2,b3=3,1,2


    # Ted's code [[90,8,10]]
    #ell,m = 15,3
    #a1,a2,a3 = 9,1,2
    #b1,b2,b3 = 0,2,7

    # [[108,8,10]]
    #ell,m = 9,6
    #a1,a2,a3 = 3,1,2
    #b1,b2,b3 = 3,1,2

    # [[288,12,18]]
    #ell,m = 12,12
    #a1,a2,a3 = 3,2,7
    #b1,b2,b3 = 3,1,2

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

    # number of logical qubits
    # k = n - rank2(hx) - rank2(hz)
    # qcode=css_code(hx,hz)
    # print('Testing CSS code...')
    # qcode.test()
    # print('Done')
    #
    # lz = qcode.lz
    # lx = qcode.lx

    lz = np.load('lz.npy')
    return hx,hz,lz

def memory_experiment_circuit(n,x_stabilizers,z_stabilizers,lz,noise_model:NoiseModel,x_detectors=False,z_detectors=True,cycles_before_noise=1,cycles_with_noise=1,cycles_after_noise=1):
    circ = stim.Circuit()
    num_stabilizers = len(x_stabilizers) + len(z_stabilizers)
    x_measurement_circ = measure_x_stabilizers(x_stabilizers,n)
    z_measurement_circ = measure_z_stabilizers(z_stabilizers,n)
    noisy_x_measurement_circ = noise_model.noisy_circuit(x_measurement_circ)
    noisy_z_measurement_circ = noise_model.noisy_circuit(z_measurement_circ)
    for i_cycle, noisy in enumerate([False]*cycles_before_noise + [True]*cycles_with_noise + [False]*cycles_after_noise):
        circ += x_measurement_circ if not noisy else noisy_x_measurement_circ
        circ += z_measurement_circ if not noisy else noisy_z_measurement_circ
        # add detectors
        if i_cycle > 0:
            if x_detectors:
                for i_x in range(len(x_stabilizers)):
                    circ.append_operation("DETECTOR", list(map(stim.target_rec,
                                                               [i_x - 2*num_stabilizers, i_x - num_stabilizers])),
                                          [i_cycle, i_x, 0])
            if z_detectors:
                for i_z in range(len(z_stabilizers)):
                    circ.append_operation("DETECTOR", list(map(stim.target_rec,
                                                               [i_z - 2*num_stabilizers + len(x_stabilizers), i_z - num_stabilizers + len(x_stabilizers)])),
                                          [i_cycle, i_z, 1])
    # add final measurements and logical operators
    circ.append_operation("M", [i for i in range(n)])
    for i_logical, logical in enumerate(lz):
        qubits_in_logical = [i for i in range(n) if logical[i] == 1]
        circ.append_operation("OBSERVABLE_INCLUDE",
                              [stim.target_rec(i - n) for i in qubits_in_logical],
                              i_logical)
    return circ

def measure_z_stabilizers(z_stabilizers,n):
    circ = stim.Circuit()
    for qubits in z_stabilizers:
        # reset the ancilla
        circ.append_operation("R", [n])
        circ.append_operation("TICK")
        # apply cx gates
        for j in qubits:
            circ.append_operation("CX", [j,n])
            circ.append_operation("TICK")
        # measure the ancilla
        circ.append_operation("M", [n])
        circ.append_operation("TICK")
    return circ

def measure_x_stabilizers(x_stabilizers,n):
    circ = stim.Circuit()
    for qubits in x_stabilizers:
        # reset the ancilla
        circ.append_operation("RX", [n])
        circ.append_operation("TICK")
        # apply cx gates
        for j in qubits:
            circ.append_operation("CX", [n,j])
            circ.append_operation("TICK")
        # measure the ancilla
        circ.append_operation("MX", [n])
        circ.append_operation("TICK")
    return circ

def get_logical_error_rate(n,x_stabilizers,z_stabilizers,lz,noise_model,num_shots):
    circuit = memory_experiment_circuit(n,x_stabilizers,z_stabilizers,lz,noise_model,cycles_before_noise=1,cycles_with_noise=1,cycles_after_noise=1)

    sampler = circuit.compile_detector_sampler()
    shots, observables = sampler.sample(num_shots, separate_observables=True)

    decoder = BPOSD(circuit.detector_error_model(), max_bp_iters=20)

    predicted_observables = decoder.decode_batch(shots)
    num_mistakes = np.sum(np.any(predicted_observables != observables, axis=1))

    print(f"{num_mistakes}/{num_shots}")
    return num_mistakes/num_shots

import random
import matplotlib.pyplot as plt
import numpy as np

# Simulated annealing to optimize stabilizer order
def simulated_annealing(x_stabilizers, z_stabilizers, n, lz, noise_model, num_shots, initial_temp=1.0, cooling_rate=0.97, min_temp=0.00, max_iters=100):
    current_x, current_z = [list(row) for row in x_stabilizers], [list(row) for row in z_stabilizers]
    current_energy = get_logical_error_rate(n, current_x, current_z, lz, noise_model, num_shots)
    best_x, best_z, best_energy = current_x, current_z, current_energy

    energies, temperature = [current_energy], initial_temp
    temperatures = [initial_temp]

    for iteration in tqdm(range(max_iters)):
        # Choose stabilizer type and shuffle one row
        current, candidate = (current_x, [row[:] for row in current_x]) if random.random() < 0.5 else (current_z, [row[:] for row in current_z])
        random.shuffle(candidate[random.randint(0, len(candidate) - 1)])

        # Evaluate new energy
        new_energy = get_logical_error_rate(n, candidate if current is current_x else current_x,
                                            candidate if current is current_z else current_z,
                                            lz, noise_model, num_shots)

        # Accept based on energy difference
        if new_energy < current_energy or random.random() < np.exp((current_energy - new_energy) / temperature):
            current_energy = new_energy
            if current is current_x:
                current_x = candidate
            else:
                current_z = candidate

            if new_energy < best_energy:
                best_x, best_z, best_energy = current_x, current_z, new_energy

        energies.append(current_energy)
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
    # plot the temperature
    plt.plot(np.array(temperatures)/10, label='Temperature/10')
    ax.relim()
    ax.autoscale_view()
    plt.pause(0.01)
    plt.show()

    return best_x, best_z, best_energy


if __name__ == '__main__':
    hx,hz,lz = get_BB_code()
    n = hx.shape[1]
    p = 0.02
    noise_model = NoiseModel.SD6(p=p)
    num_shots = 10000
    x_stabilizers = [list(np.where(row)[0]) for row in hx]
    z_stabilizers = [list(np.where(row)[0]) for row in hz]


    # # surface code:
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

    # # 3 by 3 surface code
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



    # logical_error_rate = get_logical_error_rate(n,x_stabilizers,z_stabilizers,lz,noise_model,num_shots)

    best_x, best_z, best_logical_error_rate = simulated_annealing(x_stabilizers, z_stabilizers, n, lz, noise_model,
                                                                  num_shots,max_iters=500)
    print(f'Best logical error rate: {best_logical_error_rate}')
    print(f'Best X stabilizers: {best_x}')
    print(f'Best Z stabilizers: {best_z}')