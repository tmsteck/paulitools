import unittest
import numpy as np
# Unit tests using unittest framework
#Import everything from the /src directory:
import sys
import os
from numba.core.errors import NumbaValueError
from numba.types import int8, float16
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from core import toZX, toString, concatenate_ZX, right_pad, left_pad, symplectic_inner_product, commutes, GLOBAL_INTEGER
from group import row_reduce, inner_product, null_space, radical
from ptgalois.converter import toZX as toZX_pt
from ptgalois.group import inner_product as inner_product_pt
from ptgalois.group import radical as radical_pt
from galois import GF2
from util import toBinary, getParity
#from paulitools.group import 
#from paulitools import toZX, toString, generator  as toZX_old, toString_old, generator
import ptgalois as pt
# Unit tests usin

class TestToBinary(unittest.TestCase):
    def setUp(self):
        self.set_1 = toZX(["XX", "YY", "ZZ"]) #ALL COMMUTING
        self.set_2 = toZX(["XX", "XI", "ZZ"]) #L = 1
        self.set_3 = toZX(["ZI", "XI", "IX","IZ"]) #L = 2
    
    def test_set_1(self):
        expected = np.array([
            [0, 0, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 0, 0]
        ])
        output = toBinary(self.set_1)
        np.testing.assert_array_equal(output, expected)
        

class TestGetParity(unittest.TestCase):
    def setUp(self):
        # Create test Pauli strings
        self.no_y = toZX(["XXZZII"])  # No Y operators
        self.even_y = toZX(["XYYZXI"])  # 2 Y operators (even)
        self.odd_y = toZX(["XYZI"])  # 1 Y operator (odd)
        
        self.no_x = toZX(["ZZZIII"])  # No X operators
        self.even_x = toZX(["XXZIZI"])  # 2 X operators (even)
        self.odd_x = toZX(["XZZII"])  # 1 X operator (odd)
        
        self.no_z = toZX(["XXXIII"])  # No Z operators
        self.even_z = toZX(["XZZXZI"])  # 3 Z operators (odd)
        self.odd_z = toZX(["XZIII"])  # 1 Z operator (odd)

    def test_y_parity(self):
        # Test Y parity calculations
        self.assertEqual(getParity(self.no_y, basis='Y'), 0, "Pauli string with no Y should have even Y parity")
        self.assertEqual(getParity(self.even_y, basis='Y'), 0, "Pauli string with even Y count should have even Y parity")
        self.assertEqual(getParity(self.odd_y, basis='Y'), 1, "Pauli string with odd Y count should have odd Y parity")

    def test_x_parity(self):
        # Test X parity calculations
        self.assertEqual(getParity(self.no_x, basis='X'), 0, "Pauli string with no X should have even X parity")
        self.assertEqual(getParity(self.even_x, basis='X'), 0, "Pauli string with even X count should have even X parity")
        self.assertEqual(getParity(self.odd_x, basis='X'), 1, "Pauli string with odd X count should have odd X parity")

    def test_z_parity(self):
        # Test Z parity calculations
        self.assertEqual(getParity(self.no_z, basis='Z'), 0, "Pauli string with no Z should have even Z parity")
        self.assertEqual(getParity(self.even_z, basis='Z'), 1, "Pauli string with odd Z count should have odd Z parity")
        self.assertEqual(getParity(self.odd_z, basis='Z'), 1, "Pauli string with odd Z count should have odd Z parity")

    def test_mixed_paulis(self):
        # Test with mixed Pauli operators
        mixed = toZX(["XYZI"])  # Contains X, Y, Z and I
        self.assertEqual(getParity(mixed, basis='X'), 1, "Should count one X operator")
        self.assertEqual(getParity(mixed, basis='Y'), 1, "Should count one Y operator")
        self.assertEqual(getParity(mixed, basis='Z'), 1, "Should count one Z operator")


if __name__ == '__main__':
    unittest.main()