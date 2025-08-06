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
#from paulitools.group import 
#from paulitools import toZX, toString, generator  as toZX_old, toString_old, generator
import ptgalois as pt
# Unit tests usin

class TestRowReduce(unittest.TestCase):
    def test_single_qubit(self):
        input_data = np.array([1, 2, 4, 7], dtype=GLOBAL_INTEGER)
        expected_output = np.array([1, 4, 2], dtype=GLOBAL_INTEGER)
        output = row_reduce(input_data)
        np.testing.assert_array_equal(output, expected_output)
    def test_single_matrix_single_qubit(self):
        input = np.array([1, 2], dtype=GLOBAL_INTEGER)
        expected_output = np.array([1, 2], dtype=GLOBAL_INTEGER)
        output = row_reduce(input)
        np.testing.assert_array_equal(output, expected_output)
    def test_negative_input(self):
        input = np.array([1, 7, 5], dtype=GLOBAL_INTEGER)
        expected_output = np.array([1, 4, 2], dtype=GLOBAL_INTEGER)
        output = row_reduce(input)
        np.testing.assert_array_equal(output, expected_output)

    def test_multi_qubit_do_nothing(self):
        input = toZX(["XX", "ZZ"])
        expected_output = np.array([2, 24, 6], dtype=GLOBAL_INTEGER)
        output = row_reduce(input)
        np.testing.assert_array_equal(output, expected_output)
    def test_multi_degenerate(self):
        input = toZX(["XX", "XX"])
        expected_output = np.array([2, 24], dtype=GLOBAL_INTEGER)
        output = row_reduce(input)
        np.testing.assert_array_equal(output, expected_output)
    def test_multi_dependent(self):
        input = toZX(["XX", "XI", "IX"])
        expected_output = np.array([2, 16, 8], dtype=GLOBAL_INTEGER)
        output = row_reduce(input)
        np.testing.assert_array_equal(output, expected_output)

class TestInnerProduct(unittest.TestCase):
    #Test information:
    def setUp(self):
        self.set_1_pt = toZX_pt(["XX", "YY", "ZZ"]) #ALL COMMUTING
        self.set_2_pt = toZX_pt(["XX", "XI", "ZZ"]) #L = 1
        self.set_3_pt = toZX_pt(["ZI", "XI", "IX","IZ"]) #L = 2
        
        self.set_1 = toZX(["XX", "YY", "ZZ"]) #ALL COMMUTING
        self.set_2 = toZX(["XX", "XI", "ZZ"]) #L = 1
        self.set_3 = toZX(["ZI", "XI", "IX","IZ"]) #L = 2
        
    def test_commuting(self):
        #All zeros
        expected_output = np.zeros((3,3), dtype=GLOBAL_INTEGER)
        pt_output = inner_product_pt(self.set_1_pt, self.set_1_pt)
        output = inner_product(self.set_1)
        np.testing.assert_array_equal(output, expected_output)
        np.testing.assert_array_equal(pt_output, expected_output)
    
    def test_L1(self):
        pt_output = inner_product_pt(self.set_2_pt, self.set_2_pt)
        output = inner_product(self.set_2)
        np.testing.assert_array_equal(output, pt_output)
    
    def test_L2(self):
        pt_output = inner_product_pt(self.set_3_pt, self.set_3_pt)
        output = inner_product(self.set_3)
        np.testing.assert_array_equal(output, pt_output)
    #TODO: Add mismatch handling, etc

class TestNullSpaceMod2(unittest.TestCase):
    def test_null_space(self):
        # Generate random matrices and compare the null spaces
        for _ in range(10):
            A = np.random.randint(0, 2, (5, 5))#, dtype=int8)
            expected_output = GF2(A).null_space()
            output = null_space(A)
            np.testing.assert_array_equal(output, expected_output)

class TestRadical(unittest.TestCase):
    def setUp(self):
        self.set_1_pt = toZX_pt(["XX", "YY", "ZZ"]) #ALL COMMUTING
        self.set_2_pt = toZX_pt(["XX", "XI", "ZZ"]) #L = 1
        self.set_3_pt = toZX_pt(["ZI", "XI", "IX","IZ"]) #L = 2
        
        self.set_1 = toZX(["XX", "YY", "ZZ"]) #ALL COMMUTING
        self.set_2 = toZX(["XX", "XI", "ZZ"]) #L = 1
        self.set_3 = toZX(["ZI", "XI", "IX","IZ"]) #L = 2
        
    def test_commuting(self):
        pt_output = radical_pt(self.set_1_pt)
        output = radical(self.set_1)
        np.testing.assert_array_equal(output, pt_output)
        #np.testing.assert_array_equal(pt_output, expected_output)
    def test_L1(self):
        pt_output = radical_pt(self.set_2_pt)
        output = radical(self.set_2)
        np.testing.assert_array_equal(output, pt_output)
    def test_L2(self):
        pt_output = radical_pt(self.set_3_pt)
        output = radical(self.set_3)
        np.testing.assert_array_equal(output, pt_output)

if __name__ == '__main__':
    unittest.main()