import unittest
import numpy as np
# Unit tests using unittest framework
#Import everything from the /src directory:
import sys
import os
from numba.core.errors import NumbaValueError
from numba.types import int8, float16
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from core import toZX, toString, concatenate_ZX, right_pad, left_pad, symplectic_inner_product, commutes, GLOBAL_INTEGER, bsip_array, commute_array_fast, symplectic_inner_product_int
from group import row_reduce, inner_product, null_space, radical
from ptgalois.converter import toZX as toZX_pt
from ptgalois.converter import toString as toString_pt

from ptgalois.group import inner_product as inner_product_pt
from ptgalois.group import radical as radical_pt
from galois import GF2
#from paulitools.group import 
#from paulitools import toZX, toString, generator  as toZX_old, toString_old, generator
import ptgalois as pt


pt1 = toZX(["XX", "YY", "ZZ"]) #ALL COMMUTING


ptX = toZX('X')
ptZZ = toZX('ZZ')
print(ptX)

print(symplectic_inner_product(ptX, ptZZ))


paulis = ['XXI', 'YIZ', 'ZZZ', 'IYX']
sym_form = toZX(paulis)
print(type(sym_form))
print(sym_form)
sym_form_list = []
length = sym_form[0]
for i in range(1, len(sym_form)):
    sym_form_list.append((length, np.array([sym_form[i]])))


comm_matrix = bsip_array(sym_form)
comm_matrix_fast = commute_array_fast(sym_form)

print(comm_matrix)
print(comm_matrix_fast)


#comm_matrix_fast = commute_array_fast(sym_form_list)
# n = len(sym_form)-1
# length = sym_form[0]
# commutation_matrix = np.zeros((n,n), dtype=GLOBAL_INTEGER)
# for i in range(n):
#     for j in range(i,n):
#         p1 = sym_form[i+1]
#         p2 = sym_form[j+1]
#         #print(toString(np.array([length,p1])), toString(np.array([length,p2])))
#         #print(commutes(np.array([p1]), np.array([p2]), length=length))