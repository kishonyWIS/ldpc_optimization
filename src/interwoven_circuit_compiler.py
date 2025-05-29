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
        auxiliary_qubits_to_compile = set(x_stabilizer_queue.keys())
        interwoven_cx_list = []

        while auxiliary_qubits_to_compile:
            for auxiliary_qubit in list(auxiliary_qubits_to_compile):
                if not x_stabilizer_queue[auxiliary_qubit]:
                    auxiliary_qubits_to_compile.remove(auxiliary_qubit)
                    continue

                next_data_qubit = x_stabilizer_queue[auxiliary_qubit][0]
                last_used = self.find_when_last_used(auxiliary_qubit)
                print(last_used, 'last used?')
                gate_timestep = self.find_earliest_availability(
                    [next_data_qubit], last_used)
                print(gate_timestep, 'gate_timestep?')
                # Ensure interwoven_cx_list is long enough
                while len(interwoven_cx_list) <= gate_timestep:
                    interwoven_cx_list.append(set())
                    self.qubits_used.append(set())
                interwoven_cx_list[gate_timestep].add(
                    (auxiliary_qubit, next_data_qubit)
                )
                self.qubits_used[gate_timestep].add(auxiliary_qubit)
                self.qubits_used[gate_timestep].add(next_data_qubit)

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

    def add_z_stabilizer(self, z_stab_name: str, z_stab_queue: list, cx_list: list) -> None:
        # TODO
        for data_qubit in z_stab_queue:
            earliest_availability = self.find_earliest_availability(
                data_qubit, self.data_qubits_used)

            cx_list[earliest_availability].add(
                (data_qubit, z_stab_name)
            )
            self.data_qubits_used[earliest_availability].add(
                data_qubit)
            self.ancillary_qubits_used[earliest_availability].add(
                z_stab_name)

    def test_commutativity():
        pass
