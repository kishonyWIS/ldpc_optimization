from src.cx_list_from_stabilizers_in_sequence import RotatedSurfaceCode
from src.interwoven_circuit_compiler import InterwovenCircuitCompiler
import stim

code = RotatedSurfaceCode(L=3, ordering='optimal')
cx_list = code.generate_cx_list()

# this doesn't call __init__
compiler = InterwovenCircuitCompiler.__new__(InterwovenCircuitCompiler)
compiler.code = code
compiler.cx_list = cx_list


def test_order_x_stabilizers():
    x_stabilizer_queue = {'X0': [3, 0]}

    ordered_x_stabilizers = compiler.order_x_stabilizers(x_stabilizer_queue)
    assert ordered_x_stabilizers == [{('X0', 3)}, {('X0', 0)}]

    x_stabilizer_queue = {'X0': [3, 0, 4, 1], 'X1': [
        7, 4, 8, 5], 'X2': [6, 3], 'X3': [2, 5]}
    ordered_x_stabilizers = compiler.order_x_stabilizers(x_stabilizer_queue)

    assert ordered_x_stabilizers == [{('X3', 2), ('X0', 3), ('X1', 7), ('X2', 6)}, {(
        'X1', 4), ('X0', 0), ('X2', 3), ('X3', 5)}, {('X1', 8), ('X0', 4)}, {('X0', 1), ('X1', 5)}]

    x_stabilizer_queue = {'X0': [0, 2, 1], 'X1': [0, 2, 1]}
    ordered_x_stabilizers = compiler.order_x_stabilizers(x_stabilizer_queue)
    assert ordered_x_stabilizers == [
        {('X0', 0)}, {('X1', 0), ('X0', 2)}, {('X1', 2), ('X0', 1)}, {('X1', 1)}]


def test_build_stabilizer_queues():
    x_queue, z_queue = compiler.build_stabilizer_queues(
        code.generate_cx_list())
    assert x_queue == {'X0': [3, 0, 4, 1], 'X1': [
        7, 4, 8,  5], 'X2': [6, 3], 'X3': [2, 5]}
    assert z_queue == {'Z0': [2, 1, 5, 4], 'Z1': [
        4, 3, 7, 6], 'Z2': [1, 0], 'Z3': [8, 7]}


def test_find_intersecting_stabilizers():
    intersecting_stabs = compiler.find_intersecting_stabilizers([1, 2, 4, 5])
    assert intersecting_stabs == {'X0': {1, 4}, 'X1': {4, 5}, 'X3': {2, 5}}


def test_add_z_stabilizer():
    # This is adding a Z stabilizer that doesn't lead
    # to a non-deterministic detector
    cx_list = [{('X0', 0), }, {('X0', 2), ('X1', 0)},
               {('X0', 1), ('X1', 2)}, {('X1', 1)}]
    compiler.code.x_stabilizers = [[0, 2, 1], [0, 2, 1]]
    z_stab_name = 'Z0'
    z_stab_queue = [2, 1]
    compiler.qubits_used = [{'X0', 0}, {
        'X0', 2, 'X1', 0}, {'X0', 1, 'X1', 2, 'X1', 1}]
    compiler.timestep_of_gates = {('X0', 0): 0, ('X0', 2): 1, ('X1', 0): 1,
                                  ('X0', 1): 2, ('X1', 2): 2, ('X1', 1): 3}
    cx_list_with_z_stab = compiler.add_z_stabilizer(
        z_stab_name, z_stab_queue, cx_list)

    expected_cx_list = [{('X0', 0), (2, 'Z0')},
                        {('X0', 2), ('X1', 0), (1, 'Z0')},
                        {('X0', 1), ('X1', 2)},
                        {('X1', 1)}]
    assert cx_list_with_z_stab == expected_cx_list

    # This should trigger a non-deterministic detector
    cx_list = [{('X0', 0), ('X1', 3)},
               {('X0', 1), ('X1', 2)},
               {('X0', 2), ('X1', 1)}]

    z_stab_name = 'Z0'
    z_stab_queue = [1, 2]

    compiler.qubits_used = [{'X0', 0, 'X1', 3},
                            {'X0', 1, 'X1', 2}, {'X0', 2, 'X1', 1}]
    cx_list_with_z_stab = compiler.add_z_stabilizer(
        z_stab_name, z_stab_queue, cx_list)

    expected_cx_list = [{(1, 'Z0')}, {('X0', 0), (2, 'Z0'), ('X1', 3)}, {
        ('X0', 1), ('X1', 2)}, {('X0', 2), ('X1', 1)}]

    assert expected_cx_list == cx_list_with_z_stab


# TODO fill surface code example.

def test_interweave_cxs():
    compiler.compile_strategy = 'x_z_in_sequence'
    interwoven_cx_list = compiler.interweave_cxs(cx_list)
    print(interwoven_cx_list)

    assert len(interwoven_cx_list) == 8


# test_interweave_cxs()

# def test_order_CNOTS_rsc():
#     code = RotatedSurfaceCode(L=3, ordering='optimal')

#     cx_list = code.generate_cx_list()
#     compiler = InterwovenCircuitCompiler(code, cx_list)

#     # write a test to compare to.
# # test_order_CNOTS_rsc()


# def test_order_CNOTS_commuting():
#     cx_list = [
#         (0, 'X0'),
#         (1, 'X0'),
#         (2, 'X0'),
#         (3, 'X0'),
#         (0, 'Z0'),
#         (1, 'Z0'),
#         (2, 'Z1'),
#         (3, 'Z1')
#     ]
#     compiler = InterwovenCircuitCompiler(None, cx_list)

#     reordered_cx_list = {0: {(0, 'X0')},
#                          1: {(1, 'X0'), (0, 'Z0')},
#                          2: {(2, 'X0'), (1, 'Z0')},
#                          3: {(3, 'X0'), (2, 'Z0')},
#                          4: {(3, 'Z1')}}


# def test_8_gate_circuit():
#     cx_list = [

#         (0, 'X0'),
#         (1, 'X0'),
#         (2, 'X0'),
#         (3, 'X0'),
#         (0, 'Z0'),
#         (1, 'Z0'),
#         (2, 'Z1'),
#         (3, 'Z1')
#     ]
#     """
#     QUBIT_COORDS(0, 0) 0
#     QUBIT_COORDS(1, 0) 1
#     QUBIT_COORDS(2, 0) 2
#     QUBIT_COORDS(3, 0) 3
#     QUBIT_COORDS(4, 0) 4
#     QUBIT_COORDS(5, 0) 5
#     R 0 1 2 3 5
#     RX 4
#     TICK
#     CX 4 0
#     TICK
#     CX 4 1 0 5
#     TICK
#     CX 4 2 1 5
#     TICK
#     CX 4 3 2 5
#     TICK
#     CX 3 5
#     TICK
#     M 0 1 2 3 5
#     MX 4
#     """


# test_8_gate_circuit()
# def test_d3_surface_code()
