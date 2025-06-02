from src.circuit_from_cx_list import memory_experiment_circuit_from_cx_list

import numpy as np
import stim
from src.cx_list_from_stabilizers_in_sequence import StabilizerCode, RotatedSurfaceCode

from quits.qldpc_code import *
from quits.circuit import get_qldpc_mem_circuit
from quits.decoder import sliding_window_bposd_circuit_mem
from quits.simulation import get_stim_mem_result

code = RotatedSurfaceCode(L=3)

cx_list = code.generate_cx_list()
ancilla_type, data_mapping, ancilla_mapping, flag_mapping = code.build_mappings()
lz = code.lz
lx = code.lx
data_coords = code.data_coords
ancilla_coords = code.ancilla_coords

flag_circ = memory_experiment_circuit_from_cx_list(
    cx_list=cx_list,
    ancilla_type=ancilla_type,
    data_mapping=data_mapping,
    ancilla_mapping=ancilla_mapping,
    flag_mapping=flag_mapping,
    logicals=code.lz,
    logical_type='Z',
    p_cx=0.01,
    p_idle=0,
    number_of_cycles=2,
    flag=True
)

# write test for distance

print(flag_circ)
print(flag_mapping)
print(code.x_stabilizers)
print(code.z_stabilizers)
