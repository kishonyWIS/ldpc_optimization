import stim


class MemoryExperiment:
    def __init__(self,
                 n_qubits: int,
                 x_stabilizers: list,
                 z_stabilizers: list,
                 lz: list,
                 p: float,
                 x_detectors: bool = False,
                 z_detectors: bool = True,
                 cycles_before_noise=1,
                 cycles_with_noise=1,
                 cycles_after_noise=1,
                 qubit_coords: dict = None):
        self.n_qubits = n_qubits
        self.x_stabilizers = x_stabilizers
        self.z_stabilizers = z_stabilizers
        self.lz = lz
        self.p = p
        self.x_detectors = x_detectors
        self.z_detectors = z_detectors
        self.cycles_before_noise = cycles_before_noise
        self.cycles_with_noise = cycles_with_noise
        self.cycles_after_noise = cycles_after_noise
        self.qubit_coords = qubit_coords
        self.circuit = stim.Circuit()
        if qubit_coords is not None:
            self.init_qubit_coords()
        self.generate_circuit()

    def init_qubit_coords(self):
        for q_coords, q_index in self.qubit_coords.items():
            self.circuit.append_from_stim_program_text(
                f"QUBIT_COORDS{q_coords} {q_index}")

    def generate_circuit(self):
        measurement_counter = 0
        measurement_indexes = {'X_syndromes': {i: [] for i in range(len(self.x_stabilizers))},
                               'Z_syndromes': {i: [] for i in range(len(self.z_stabilizers))},
                               'X_flags': {i: [] for i in range(len(self.x_stabilizers))},
                               'Z_flags': {i: [] for i in range(len(self.z_stabilizers))}}
        for i_cycle, noisy in enumerate([False]*self.cycles_before_noise + [True]*self.cycles_with_noise + [False]*self.cycles_after_noise):
            x_measurement_circ, measurement_counter = self.measure_x_stabilizers(
                flag=noisy,
                measurement_indexes=measurement_indexes,
                measurement_counter=measurement_counter)
            z_measurement_circ, measurement_counter = self.measure_z_stabilizers(
                flag=noisy,
                measurement_indexes=measurement_indexes,
                measurement_counter=measurement_counter)
            self.circuit += x_measurement_circ
            self.circuit += z_measurement_circ
        # add detectors
        if self.x_detectors:
            for i_stabilizer, indexes_of_measurements in measurement_indexes['X_syndromes'].items():
                for i_cycle in range(len(indexes_of_measurements) - 1):
                    indexes = [
                        ii - measurement_counter for ii in indexes_of_measurements[i_cycle:i_cycle+2]]
                    self.circuit.append_operation("DETECTOR",
                                                  list(
                                                      map(stim.target_rec, indexes)),
                                                  [i_cycle, i_stabilizer, 0])
        if self.z_detectors:
            for i_stabilizer, indexes_of_measurements in measurement_indexes['Z_syndromes'].items():
                for i_cycle in range(len(indexes_of_measurements) - 1):
                    indexes = [
                        ii - measurement_counter for ii in indexes_of_measurements[i_cycle:i_cycle+2]]
                    self.circuit.append_operation("DETECTOR",
                                                  list(
                                                      map(stim.target_rec, indexes)),
                                                  [i_cycle, i_stabilizer, 1])

        # add final measurements and logical operators
        self.circuit.append_operation("M", [i for i in range(self.n_qubits)])
        measurement_counter += self.n_qubits
        """
        for i_logical, logical in enumerate(self.lz):
            qubits_in_logical = [i for i in range(self.n_qubits) if logical[i] == 1]
            self.circuit.append_operation("OBSERVABLE_INCLUDE",
                                          [stim.target_rec(i - self.n_qubits)
                                           for i in qubits_in_logical],
                                          i_logical)

        circ_without_flag_observables = self.circuit.copy()

        # add flag observables
        observable_index = self.lz.shape[0]
        for i_flag, indexes_of_measurements in measurement_indexes['X_flags'].items():
            for i_cycle in range(len(indexes_of_measurements)):
                indexes = [
                    ii - measurement_counter for ii in [indexes_of_measurements[i_cycle]]]
                self.circuit.append_operation("OBSERVABLE_INCLUDE",
                                              list(
                                                  map(stim.target_rec, indexes)),
                                              observable_index)
                observable_index += 1
        for i_flag, indexes_of_measurements in measurement_indexes['Z_flags'].items():
            for i_cycle in range(len(indexes_of_measurements)):
                indexes = [
                    ii - measurement_counter for ii in [indexes_of_measurements[i_cycle]]]
                self.circuit.append_operation("OBSERVABLE_INCLUDE",
                                              list(
                                                  map(stim.target_rec, indexes)),
                                              observable_index)
                observable_index += 1
        
        return circ_without_flag_observables
        """

    def measure_z_stabilizers(self, flag, measurement_indexes=None, measurement_counter=0):
        circ = stim.Circuit()
        for i_stab, qubits in enumerate(self.z_stabilizers):
            # reset the ancilla
            circ.append_operation("R", [self.n_qubits])
            # flag
            if flag and self.p > 0:
                circ.append_operation("RX", [self.n_qubits+1])
            # apply cx gates
            for iq, q in enumerate(qubits):
                # flag
                # before the last
                if flag and self.p > 0 and iq == len(qubits) - 1:
                    circ.append_operation(
                        "CX", [self.n_qubits + 1, self.n_qubits])
                if self.p > 0:
                    circ.append_operation(
                        "DEPOLARIZE2", [q, self.n_qubits], self.p)
                circ.append_operation("CX", [q, self.n_qubits])
                # flag
                if flag and self.p > 0 and iq == 0:  # after the first
                    circ.append_operation(
                        "CX", [self.n_qubits + 1, self.n_qubits])
            # flag
            if flag and self.p > 0:
                circ.append_operation("MX", [self.n_qubits+1])
                measurement_indexes['Z_flags'][i_stab].append(
                    measurement_counter)
                measurement_counter += 1
            # measure the ancilla
            circ.append_operation("M", [self.n_qubits])
            measurement_indexes['Z_syndromes'][i_stab].append(
                measurement_counter)
            measurement_counter += 1
        return circ, measurement_counter

    def measure_x_stabilizers(self, flag=False, measurement_indexes=None, measurement_counter=0):
        circ = stim.Circuit()
        for i_stab, qubits in enumerate(self.x_stabilizers):
            # reset the ancilla
            circ.append_operation("RX", [self.n_qubits])
            # flag
            if flag and self.p > 0:
                circ.append_operation("R", [self.n_qubits+1])
        # apply cx gates
        for iq, q in enumerate(qubits):
            # flag
            if flag and self.p > 0 and iq == len(qubits)-1:  # before the last
                circ.append_operation("CX", [self.n_qubits, self.n_qubits + 1])
            if self.p > 0:
                circ.append_operation(
                    "DEPOLARIZE2", [self.n_qubits, q], self.p)
            circ.append_operation("CX", [self.n_qubits, q])
            # flag
            if flag and self.p > 0 and iq == 0:  # after the first
                circ.append_operation("CX", [self.n_qubits, self.n_qubits + 1])
        # flag
        if flag and self.p > 0:
            circ.append_operation("M", [self.n_qubits+1])
            measurement_indexes['X_flags'][i_stab].append(measurement_counter)
            measurement_counter += 1
        # measure the ancilla
        circ.append_operation("MX", [self.n_qubits])
        measurement_indexes['X_syndromes'][i_stab].append(measurement_counter)
        measurement_counter += 1
        return circ, measurement_counter
