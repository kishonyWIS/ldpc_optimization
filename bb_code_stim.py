import numpy as np
from bposd.css import css_code
import stim
from noise import NoiseModel
from stimbposd import BPOSD

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
k = n - rank2(hx) - rank2(hz)
qcode=css_code(hx,hz)
print('Testing CSS code...')
qcode.test()
print('Done')

lz = qcode.lz
lx = qcode.lx

def memory_experiment_circuit(hx,hz,lz,noise_model:NoiseModel,x_detectors=False,z_detectors=True,cycles_before_noise=1,cycles_with_noise=1,cycles_after_noise=1):
    circ = stim.Circuit()
    num_stabilizers = hx.shape[0] + hz.shape[0]
    x_measurement_circ = measure_x_stabilizers(hx)
    z_measurement_circ = measure_z_stabilizers(hz)
    noisy_x_measurement_circ = noise_model.noisy_circuit(x_measurement_circ)
    noisy_z_measurement_circ = noise_model.noisy_circuit(z_measurement_circ)
    for i_cycle, noisy in enumerate([False]*cycles_before_noise + [True]*cycles_with_noise + [False]*cycles_after_noise):
        circ += x_measurement_circ if not noisy else noisy_x_measurement_circ
        circ += z_measurement_circ if not noisy else noisy_z_measurement_circ
        # add detectors
        if i_cycle > 0:
            if x_detectors:
                for i_x in range(hx.shape[0]):
                    circ.append_operation("DETECTOR", list(map(stim.target_rec,
                                                               [i_x - 2*num_stabilizers, i_x - num_stabilizers])),
                                          [i_cycle, i_x, 0])
            if z_detectors:
                for i_z in range(hz.shape[0]):
                    circ.append_operation("DETECTOR", list(map(stim.target_rec,
                                                               [i_z - 2*num_stabilizers + hx.shape[0], i_z - num_stabilizers + hx.shape[0]])),
                                          [i_cycle, i_z, 1])
    # add final measurements and logical operators
    circ.append_operation("M", [i for i in range(n)])
    for i_logical, logical in enumerate([lz]):
        qubits_in_logical = [i for i in range(n) if logical[i] == 1]
        circ.append_operation("OBSERVABLE_INCLUDE",
                              [stim.target_rec(i - n) for i in qubits_in_logical],
                              i_logical)
    return circ

def measure_z_stabilizers(hz):
    circ = stim.Circuit()
    n = hx.shape[1]
    for i in range(hz.shape[0]):
        # reset the ancilla
        circ.append_operation("R", [n])
        # apply cx gates
        for j in range(n):
            if hz[i,j] == 1:
                circ.append_operation("CX", [j,n])
        # measure the ancilla
        circ.append_operation("M", [n])
    return circ

def measure_x_stabilizers(hx):
    circ = stim.Circuit()
    n = hx.shape[1]
    for i in range(hx.shape[0]):
        # reset the ancilla
        circ.append_operation("RX", [n])
        # apply cx gates
        for j in range(n):
            if hx[i,j] == 1:
                circ.append_operation("CX", [n,j])
        # measure the ancilla
        circ.append_operation("MX", [n])
    return circ


noise_model = NoiseModel.SD6(p=0.001)
num_shots = 1000
circuit = memory_experiment_circuit(hx,hz,lz,noise_model,cycles_before_noise=1,cycles_with_noise=1,cycles_after_noise=1)

sampler = circuit.compile_detector_sampler()
shots, observables = sampler.sample(num_shots, separate_observables=True)

decoder = BPOSD(circuit.detector_error_model(), max_bp_iters=20)

predicted_observables = decoder.decode_batch(shots)
num_mistakes = np.sum(np.any(predicted_observables != observables, axis=1))

print(f"{num_mistakes}/{num_shots}")