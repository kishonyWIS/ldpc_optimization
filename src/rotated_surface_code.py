
class RotatedSurfaceCode():
    def __init__(self, distance):
        self.distance = distance
        self.init_data_qubits()
        self.init_face_checks()
        self.init_boundary_checks()

    def init_data_qubits(self):
        self.data_qubits = dict()
        y_middle = self.distance-1
        q_index = 0
        # diagonal lines of data qubits
        starting_x, starting_y = (0, y_middle)
        for i in range(self.distance):
            for j in range(self.distance):
                self.data_qubits[starting_x+j, starting_y-j] = q_index
                q_index += 1
            starting_x += 1
            starting_y += 1

    def init_face_checks(self):
        self.ancilla_qubits = dict()
        x_middle, y_middle = (self.distance-1, self.distance-1)
        q_index = len(self.data_qubits)
        check_type = ['z', 'x']
        self.z_checks = []
        self.x_checks = []
        starting_x, starting_y = (1, y_middle)

        for i in range(self.distance-1):
            for j in range(self.distance-1):
                center = (starting_x+j, starting_y-j)

                if j % 2 == 0:
                    pauli_op = check_type[i % 2]
                    new_check = [self.data_qubits[center[0]+1, center[1]],
                                 self.data_qubits[center[0], center[1]+1],
                                 self.data_qubits[center[0]-1, center[1]],
                                 self.data_qubits[center[0], center[1]-1]]

                else:
                    pauli_op = check_type[(i+1) % 2]
                    new_check = [self.data_qubits[center[0]+1, center[1]],
                                 self.data_qubits[center[0], center[1]-1],
                                 self.data_qubits[center[0], center[1]+1],
                                 self.data_qubits[center[0]-1, center[1]]]

                if pauli_op == 'z':
                    self.z_checks.append(new_check)
                else:
                    self.x_checks.append(new_check)
                self.ancilla_qubits[center] = q_index
                q_index += 1

            starting_x += 1
            starting_y += 1

    def init_boundary_checks(self):
        q_index = len(self.ancilla_qubits) + len(self.data_qubits)
        x_middle, y_middle = self.distance-1, self.distance-1
        # y_top = d
        for i in range(0, self.distance//2):
            # bottom left boundary
            center = (2*i+1, y_middle-2*(i+1))
            self.ancilla_qubits[center] = q_index
            self.z_checks.append([self.data_qubits[center[0]+1, center[1]],
                                  self.data_qubits[center[0], center[1]+1]])
            q_index += 1

            # top right boundary
            center = (x_middle + 2*i + 1, 2*(self.distance-1) - 2*i)
            self.ancilla_qubits[center] = q_index
            self.z_checks.append([self.data_qubits[center[0], center[1]-1],
                                  self.data_qubits[center[0]-1, center[1]]])
            q_index += 1

            # top left boundary
            center = (2*i, y_middle + 2*i+1)
            self.ancilla_qubits[center] = q_index
            self.x_checks.append([self.data_qubits[center[0]+1, center[1]],
                                  self.data_qubits[center[0], center[1]-1]])
            q_index += 1

            # bottom right boundary
            center = (x_middle + 2*(i+1), 2*i + 1)
            self.ancilla_qubits[center] = q_index
            self.x_checks.append([self.data_qubits[center[0], center[1]+1],
                                  self.data_qubits[center[0]-1, center[1]]])

            q_index += 1
