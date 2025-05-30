from src.cx_list_from_stabilizers_in_sequence import StabilizerCode
from collections import defaultdict


class InterwovenCircuitCompiler():
    def __init__(self,
                 code: StabilizerCode,
                 cx_list: list,
                 compile_strategy: str = 'x_z_in_sequence'):
        """
        compile_strategy: 'x_z_in_sequence' or 'x_z_in_parallel'

        """
        self.code = code
        self.compile_strategy = compile_strategy
        self.interwoven_cx_list = self.interweave_cxs(cx_list)

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
