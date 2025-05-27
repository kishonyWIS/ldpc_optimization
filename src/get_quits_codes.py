import stim
from src.cx_list_from_stabilizers_in_sequence import StabilizerCode
from quits.qldpc_code import *
from quits.circuit import get_qldpc_mem_circuit
from quits.decoder import sliding_window_bposd_circuit_mem
from quits.simulation import get_stim_mem_result


def get_bpc_code(cx_order='lex'):
    lift_size, factor = 15, 3
    p1 = [0, 1, 5]  # e + x + x^5
    p2 = [0, 2, 7]  # e + x^2 + x^7

    code = BpcCode(p1, p2, lift_size, factor)  # Define the BpcCode object
    return init_code(code, cx_order=cx_order)


def get_lpc_code(cx_order='lex'):
    lift_size = 16
    b = np.array([
        [0, 0, 0, 0, 0],  # [e, e, e, e, e]
        [0, 2, 4, 7, 11],  # [e, x^2, x^4, x^7, x^11]
        [0, 3, 10, 14, 15]  # [e, x^3, x^10, x^14, x^15]
    ])
    code = QlpCode(b, b, lift_size)  # Define the QlpCode object
    return init_code(code, cx_order=cx_order)


def init_code(code, cx_order='lex'):
    # Build the Tanner graph and assign directions to its edges.
    code.build_graph(seed=1)

    if cx_order == 'lex':
        stabilizer_x = []
        for i in range(code.hx.shape[0]):
            row = []
            for j in range(code.hx.shape[1]):
                if code.hx[i, j] == 1:
                    row.append(j)
            stabilizer_x.append(row)

        stabilizer_z = []
        for i in range(code.hz.shape[0]):
            row = []
            for j in range(code.hz.shape[1]):
                if code.hz[i, j] == 1:
                    row.append(j)
            stabilizer_z.append(row)

        bpc_code = StabilizerCode(stabilizer_x, stabilizer_z, code.lx, code.lz)
        cx_list = bpc_code.generate_cx_list()
        ancilla_type, data_mapping, ancilla_mapping = bpc_code.build_mappings()

    elif cx_order == 'theirs':
        ancilla_mapping = {
            'Z' + str(i): q for i, q in enumerate(code.zcheck_qubits)}
        ancilla_mapping.update(
            {'X' + str(i): q for i, q in enumerate(code.xcheck_qubits)})
        ancilla_type = {
            name: 'X' if name[0] == 'X' else 'Z' for name in ancilla_mapping.keys()}
        data_mapping = {q: q for q in set(
            code.all_qubits) - set(ancilla_mapping.values())}

        cx_list = []
        directions = list(code.direction_inds.keys())
        for direction_ind in range(len(directions)):
            direction = directions[direction_ind]
            for color in range(code.num_colors[direction]):
                edges = code.colored_edges[direction_ind][color]
                cx_list.extend(list(zip(edges[::2], edges[1::2])))

        all_qubits_ids_to_names = {}
        for name, q in ancilla_mapping.items():
            all_qubits_ids_to_names[q] = name
        for name, q in data_mapping.items():
            all_qubits_ids_to_names[q] = name
        # %%
        # map qubit indices to ancilla names or data names in the cx_list
        for i in range(len(cx_list)):
            q1, q2 = cx_list[i]
            cx_list[i] = (all_qubits_ids_to_names[q1],
                          all_qubits_ids_to_names[q2])
            # change the order so that the data qubit is always the first one
            if cx_list[i][0] in ancilla_mapping:
                assert cx_list[i][1] in data_mapping
                cx_list[i] = (cx_list[i][1], cx_list[i][0])
            else:
                assert cx_list[i][1] in ancilla_mapping

    else:
        raise ValueError('Unknown cx_order')

    return cx_list, ancilla_type, data_mapping, ancilla_mapping, code.lz, code.lx
