class InterwovenCircuitCompiler():
    def __init__(self, code, cx_list):
        self.code = code
        self.interwoven_cx_list = self.interweave_cxs(cx_list)

    def interweave_cxs(self, cx_list):
        x_stabilizer_queue, z_stabilizer_queue = self.build_queues(cx_list)
        interwoven_cx_list = self.order_x_stabilizers(x_stabilizer_queue)

    def build_stabilizer_queues(self, cx_list):
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

    def order_x_stabilizers(self, x_stabilizer_queue):
        """Order the x stabilizers in a way that minimizes the number of idle cycles.

        The ordering is done by interleaving the CNOTs in such a way that
        each auxiliary qubit is used as soon as it is available, while also
        ensuring that data qubits are not reused until they are available again.
        This is done by maintaining a list of sets, where each set corresponds to a cycle
        and contains the CNOTs that can be executed in that cycle
        """
        data_qubits_used = [set()]
        auxiliary_qubits_used = [set()]
        auxiliary_qubits_to_compile = set(x_stabilizer_queue.keys())
        interwoven_cx_list = []

        while auxiliary_qubits_to_compile:
            for auxiliary_qubit in list(auxiliary_qubits_to_compile):
                if not x_stabilizer_queue[auxiliary_qubit]:
                    auxiliary_qubits_to_compile.remove(auxiliary_qubit)
                    continue

                next_data_qubit = x_stabilizer_queue[auxiliary_qubit][0]
                earliest_availability = self.find_earliest_availability(
                    next_data_qubit, auxiliary_qubit, data_qubits_used, auxiliary_qubits_used
                )

                # Ensure interwoven_cx_list is long enough
                while len(interwoven_cx_list) <= earliest_availability:
                    interwoven_cx_list.append(set())
                    data_qubits_used.append(set())
                    auxiliary_qubits_used.append(set())

                interwoven_cx_list[earliest_availability].add(
                    (auxiliary_qubit, next_data_qubit)
                )
                data_qubits_used[earliest_availability].add(next_data_qubit)
                auxiliary_qubits_used[earliest_availability].add(
                    auxiliary_qubit)

                x_stabilizer_queue[auxiliary_qubit].pop(0)
        return interwoven_cx_list

    def find_earliest_availability(self, data_qubit, auxiliary_qubit, data_qubits_used, auxiliary_qubits_used):
        """
        Find the earliest cycle when both the data qubit and the auxiliary qubit are available.
        """
        data_last_used = next((i for i in range(
            len(data_qubits_used)-1, -1, -1) if data_qubit in data_qubits_used[i]), -1)
        aux_last_used = next((i for i in range(len(
            auxiliary_qubits_used)-1, -1, -1) if auxiliary_qubit in auxiliary_qubits_used[i]), -1)
        return (max(data_last_used, aux_last_used)+1)
