from typing import Dict
import stim
from src.cx_list_from_stabilizers_in_sequence import StabilizerCode
from collections import defaultdict


class InterwovenCircuitCompiler():
    def __init__(self,
                 code: StabilizerCode,
                 cx_list: list,
                 compile_strategy: str = 'x_z_in_sequence',
                 basis: str = 'Z',
                 both_detectors: bool = True,
                 num_iterations: int = 1
                 ):
        """
        compile_strategy: 'x_z_in_sequence' or 'x_z_in_parallel'
        ancilla_type: Mapping ancilla id -> "X" or "Z".
        data_mapping: Mapping data qubit id -> physical index.
        ancilla_mapping: Mapping ancilla id -> physical index.
        """
        self.code = code
        self.compile_strategy = compile_strategy
        self.interwoven_cx_list = self.interweave_cxs(cx_list)
        self.basis = basis  # 'X' or 'Z'
        self.both_detectors = both_detectors
        self.num_iterations = num_iterations
        self.circ = self.generate_stim_circuit()

    def interweave_cxs(self, cx_list: list) -> list:
        x_stabilizers_queue, z_stabilizers_queue = self.build_stabilizer_queues(
            cx_list)
        interwoven_cx_list = self.order_x_stabilizers(x_stabilizers_queue)
        if self.compile_strategy == 'x_z_in_sequence':
            z_cx_list = self.order_x_stabilizers(z_stabilizers_queue)
            interwoven_cx_list.extend(z_cx_list)
        elif self.compile_strategy == 'x_z_in_parallel':
            # TODO should I flip the interwoven cx list? I think so.

            for z_stabilizer_queue in z_stabilizers_queue.values():
                self.add_z_stabilizer(
                    z_stabilizer_queue, interwoven_cx_list)
        return (interwoven_cx_list)

    def build_stabilizer_queues(self, cx_list: list) -> tuple:
        """
        Build the x and z stabilizer queues from the list of CNOTs.
        Each queue is a dictionary where the keys are auxiliary qubits (stabilizers)
        and the values are lists of data qubits that are connected to that auxiliary qubit.
        """
        x_stabilizer_queue = {}
        z_stabilizer_queue = {}

        for data_qubit, auxiliary_qubit in cx_list:
            if auxiliary_qubit.startswith('X'):
                if auxiliary_qubit not in x_stabilizer_queue:
                    x_stabilizer_queue[auxiliary_qubit] = [data_qubit]
                else:
                    x_stabilizer_queue[auxiliary_qubit].append(data_qubit)
            elif auxiliary_qubit.startswith('Z'):
                if auxiliary_qubit not in z_stabilizer_queue:
                    z_stabilizer_queue[auxiliary_qubit] = [data_qubit]
                else:
                    z_stabilizer_queue[auxiliary_qubit].append(data_qubit)

        return (x_stabilizer_queue, z_stabilizer_queue)

    def order_x_stabilizers(self, x_stabilizer_queue: dict) -> list:
        """Order the x stabilizers in a way that minimizes the number of idle cycles.

        The ordering is done by interleaving the CNOTs in such a way that
        each auxiliary qubit is used as soon as it is available, while also
        ensuring that data qubits are not reused until they are available again.
        This is done by maintaining a list of sets, where each set corresponds to a cycle
        and contains the CNOTs that can be executed in that cycle
        """

        self.qubits_used = list({},)
        interwoven_cx_list = []
        self.timestep_of_gates = defaultdict(int)

        while x_stabilizer_queue:
            for auxiliary_qubit in list(x_stabilizer_queue.keys()):
                # If the auxiliary qubit has no more data qubits, remove it from the queue
                if not x_stabilizer_queue[auxiliary_qubit]:
                    del x_stabilizer_queue[auxiliary_qubit]
                    continue

                next_data_qubit = x_stabilizer_queue[auxiliary_qubit][0]

                # last_used is one timestep after the last time the auxiliary qubit was used
                last_used = self.find_when_last_used(auxiliary_qubit)

                # Find the earliest timestep after last_used when the next data qubit is available
                gate_timestep = self.find_earliest_availability(
                    [next_data_qubit], last_used)

                # Ensure interwoven_cx_list is long enough
                while len(interwoven_cx_list) <= gate_timestep:
                    interwoven_cx_list.append(set())
                    self.qubits_used.append(set())

                # Add the CNOT to the interwoven list
                interwoven_cx_list[gate_timestep].add(
                    (auxiliary_qubit, next_data_qubit)
                )
                self.timestep_of_gates[(
                    auxiliary_qubit, next_data_qubit)] = gate_timestep

                # Mark the qubits as used in this timestep
                self.qubits_used[gate_timestep].update(
                    [auxiliary_qubit, next_data_qubit])

                # Remove the data qubit from the queue
                x_stabilizer_queue[auxiliary_qubit].pop(0)
        return interwoven_cx_list

    def find_when_last_used(self, qubit) -> int:
        """
        Find the last cycle when the qubit was used. Useful for auxiliary qubit.
        """

        for i in range(len(self.qubits_used) - 1, -1, -1):
            if qubit in self.qubits_used[i]:

                return i + 1

        return 0

    def find_earliest_availability(self,
                                   qubits: list,
                                   start_looking_from: int = 0) -> int:
        """
        Find the earliest cycle when both the data qubit and the auxiliary qubit are available.
        """

        qubits_earliest_availability = start_looking_from
        for qubit in qubits:
            # Find the earliest cycle where the qubit is not used
            qubit_earliest_availability = None
            for i in range(start_looking_from, len(self.qubits_used)):

                if qubit not in self.qubits_used[i]:
                    qubit_earliest_availability = i

                    if qubit_earliest_availability > qubits_earliest_availability:
                        qubits_earliest_availability = qubit_earliest_availability
                    break

            if qubit_earliest_availability == None:
                qubit_earliest_availability = len(self.qubits_used)
                self.qubits_used.append(set())

        return qubits_earliest_availability

    def find_intersecting_stabilizers(self, z_stab_queue: list) -> list:
        intersecting_stabilizers = dict()
        for stab_index, x_stab in enumerate(self.code.x_stabilizers):
            intersecting_qubits = set(x_stab) & set(z_stab_queue)
            if intersecting_qubits:
                intersecting_stabilizers['X' +
                                         str(stab_index)] = intersecting_qubits
        return intersecting_stabilizers

    def add_gates(self, gates_to_add: dict, cx_list: list) -> list:
        """
        Add gates to the cx_list at the specified timesteps.
        gates_to_add is a dictionary where keys are timesteps and values are tuples of (data_qubit, auxiliary_qubit).
        """
        for timestep, (data_qubit, auxiliary_qubit) in gates_to_add.items():

            # Add the CNOT to the cx_list
            cx_list[timestep].add((data_qubit, auxiliary_qubit))
            self.qubits_used[timestep].update(
                [data_qubit, auxiliary_qubit])
            self.timestep_of_gates[(data_qubit, auxiliary_qubit)] = timestep

        return cx_list

    def attempt_to_add_z_stabilizer(self, z_stab_name: str, z_stab_queue: list, intersecting_stabilizers: dict) -> bool:
        last_used = 0
        gates_to_add = dict()
        for data_qubit in z_stab_queue:
            earliest_availability = self.find_earliest_availability(
                [data_qubit], last_used)

            for x_stab_name, intersecting_qubits in intersecting_stabilizers.items():
                if data_qubit in intersecting_qubits:
                    if not earliest_availability < self.timestep_of_gates[(x_stab_name, data_qubit)]:
                        return (False, gates_to_add)
            else:
                gates_to_add[earliest_availability] = (data_qubit, z_stab_name)
            last_used = earliest_availability + 1
        return (True, gates_to_add)

    def add_z_stabilizer(self, z_stab_name: str, z_stab_queue: list, cx_list: list) -> None:
        """
        Add a Z stabilizer to the circuit, ensuring that it does not lead to a non-deterministic detector.
        This is done by inserting CNOTs as soon as possible. Such that they are performed before X stabilizer CNOTs.
        If there is no timestep available, an empty timestep is added at the beginning of the circuit."""
        intersecting_stabilizers = self.find_intersecting_stabilizers(
            z_stab_queue)
        succesfull = False
        while succesfull is False:
            succesfull, gates_to_add = self.attempt_to_add_z_stabilizer(
                z_stab_name, z_stab_queue, intersecting_stabilizers)
            if not succesfull:
                # add an empty timestep at the beginning
                cx_list.insert(0, set())
                self.qubits_used.insert(0, set())
                # add + 1 to each value in timestep_of_gates
                for key in self.timestep_of_gates:
                    # this is probably a little inefficient.
                    self.timestep_of_gates[key] += 1
        cx_list = self.add_gates(gates_to_add, cx_list)
        return cx_list

    def create_circuit_layer(self,
                             circ: stim.Circuit,
                             first_round: bool,
                             x_detectors: bool,
                             z_detectors: bool) -> list:
        """
        Create a circuit layer from the interwoven CNOT list.
        This is a list of sets, where each set contains the CNOTs that can be executed in that cycle.
        """
        for a_label, a_index in self.code.ancilla_mapping.items():
            circ.append(
                'R' if a_label[0] == 'Z' else 'RX',
                a_index
            )

        circ.append('TICK')
        for timestep, cx_gates in enumerate(self.interwoven_cx_list):
            for gate in cx_gates:
                auxiliary_qubit, data_qubit = gate
                if self.code.ancilla_type[auxiliary_qubit] == 'X':
                    circ.append('CX', [
                        self.code.data_mapping[data_qubit], self.code.ancilla_mapping[auxiliary_qubit]])
                else:
                    circ.append('CX', [
                        self.code.ancilla_mapping[auxiliary_qubit], self.code.data_mapping[data_qubit]])
            circ.append('TICK')

        for a_label, a_index in self.code.ancilla_mapping.items():
            circ.append(
                'M' if a_label[0] == 'Z' else 'MX',
                a_index
            )

        if first_round:

            if self.basis == 'Z':
                for i, (a_label, a_index) in enumerate(self.code.ancilla_mapping.items()):
                    if a_label[0] == 'Z' and z_detectors:
                        rec_targets = [stim.target_rec(
                            i - len(self.code.ancilla_mapping))]
                        circ.append(
                            'DETECTOR', rec_targets)
            else:
                for i, (a_label, a_index) in enumerate(self.code.ancilla_mapping.items()):
                    if a_label[0] == 'X' and x_detectors:
                        rec_targets = [stim.target_rec(
                            i - len(self.code.ancilla_mapping))]
                        circ.append(
                            'DETECTOR', rec_targets)
        else:
            if x_detectors:
                for i, (a_label, a_index) in enumerate(self.code.ancilla_mapping.items()):
                    if a_label[0] == 'X' and z_detectors:
                        rec_targets = [stim.target_rec(
                            i - len(self.code.ancilla_mapping)),  stim.target_rec(i - 2*len(self.code.ancilla_mapping))]
                        circ.append(
                            'DETECTOR', rec_targets)
            if z_detectors:
                for i, (a_label, a_index) in enumerate(self.code.ancilla_mapping.items()):
                    if a_label[0] == 'Z' and x_detectors:
                        rec_targets = [stim.target_rec(
                            i - len(self.code.ancilla_mapping)), stim.target_rec(i - 2*len(self.code.ancilla_mapping))]
                        circ.append(
                            'DETECTOR', rec_targets)
        return (circ)

    def generate_stim_circuit(self) -> stim.Circuit:
        """
        Generate a Stim circuit from the interwoven CNOT list.
        """
        if self.both_detectors == True:
            include_x_detectors = True
            include_z_detectors = True
        else:
            include_x_detectors = self.basis == 'X'
            include_z_detectors = self.basis == 'Z'
        data_indices = [self.code.data_mapping[q]
                        for q in sorted(self.code.data_mapping.keys())]
        head_circ = stim.Circuit()
        head_circ.append(
            'R' if self.basis == 'Z' else 'RX', data_indices)
        self.create_circuit_layer(
            head_circ,
            first_round=True,
            x_detectors=include_x_detectors,
            z_detectors=include_z_detectors
        )

        body_circ = stim.Circuit()

        self.create_circuit_layer(
            body_circ,
            first_round=False,
            x_detectors=include_x_detectors,
            z_detectors=include_z_detectors
        )

        tail_circ = stim.Circuit()
        tail_circ.append(
            f'M{self.basis}', data_indices)
        for i, x_stab in enumerate(self.code.x_stabilizers):

            rec_targets = [stim.target_rec(
                -1*self.code.n + q_i) for q_i in x_stab]
            rec_targets.append(stim.target_rec(
                -1*self.code.n - len(self.code.ancilla_mapping) + i))
            tail_circ.append('DETECTOR', rec_targets)
        for i, z_stab in enumerate(self.code.z_stabilizers):
            rec_targets = [stim.target_rec(
                -1*self.code.n + q_i) for q_i in z_stab]
            rec_targets.append(stim.target_rec(
                -1*self.code.n - len(self.code.ancilla_mapping) + i + len(self.code.x_stabilizers)))
            tail_circ.append('DETECTOR', rec_targets)

        if self.basis == 'Z':
            logicals = self.code.lz
        else:
            logicals = self.code.lx

        for i_logical, logical in enumerate(logicals):
            qubits_in_logical = [i for i in range(
                self.code.n) if logical[i] == 1]

            rec_targets = [stim.target_rec(
                i - self.code.n) for i in qubits_in_logical]

            tail_circ.append(
                'OBSERVABLE_INCLUDE', rec_targets, i_logical)
        final_circuit = head_circ + \
            (self.num_iterations-1)*body_circ + tail_circ
        return final_circuit
