import unittest
import numpy as np
# Unit tests using unittest framework
#Import everything from the /src directory:
import sys
import os
from numba.core.errors import NumbaValueError
from numba.types import int8, float16
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from core import (
    toZX,
    toString,
    right_pad,
    left_pad,
    symplectic_inner_product,
    commutes,
    GLOBAL_INTEGER,
    toZX_extended,
    to_standard_if_possible,
)
from group import row_reduce, inner_product, null_space, radical, differences
from ptgalois.converter import toZX as toZX_pt
from ptgalois.group import inner_product as inner_product_pt
from ptgalois.group import radical as radical_pt
from galois import GF2
from util import toBinary, getParity,filtered_purity, filtered_purity_reference
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
class TestUtilFunctionsWithExtension(unittest.TestCase):
    def setUp(self):
        self.pauli_strings = ["XYZI", "ZZXX"]
        self.legacy = toZX(self.pauli_strings)
        self.large_collection = toZX_extended(self.pauli_strings, force_large=True)
        self.large_standard = to_standard_if_possible(self.large_collection)

    def test_toBinary_matches_legacy(self):
        binary_legacy = toBinary(self.legacy)
        binary_large = toBinary(self.large_standard)
        np.testing.assert_array_equal(binary_large, binary_legacy)

    def test_getParity_matches_legacy(self):
        k = int(self.legacy[0])
        for idx in range(1, len(self.legacy)):
            pauli_legacy = np.array([k, self.legacy[idx]], dtype=self.legacy.dtype)
            pauli_large = np.array([k, self.large_standard[idx]], dtype=self.large_standard.dtype)
            for basis in ("X", "Y", "Z"):
                parity_legacy = getParity(pauli_legacy, basis=basis)
                parity_large = getParity(pauli_large, basis=basis)
                self.assertEqual(parity_large, parity_legacy)

class TestStabilizerPurity(unittest.TestCase):
    def setUp(self):
        #load in the pickled array qiskit_test_data.pkl from the same folder:
        import pickle
        with open(os.path.join(os.path.dirname(__file__), 'qiskit_test_data.pkl'), 'rb') as f:
            self.test_data = pickle.load(f)
        self.differences = differences(self.test_data)
        self.actual_center = toZX(['XIII', 'IIIX'])
        #now load: qiskit_test_data_noisy.pkl
        with open(os.path.join(os.path.dirname(__file__), 'qiskit_test_data_noisy.pkl'), 'rb') as f:
            self.noisy_data = pickle.load(f)
        self.noisy_differences = differences(self.noisy_data)
        
        
    def test_centralizer(self):
        # Check that the centralizer of the differences is the original stabilizer group
        from group import centralizer
        computed_centralizer = toZX(centralizer(self.differences))
        # Convert both to sets of strings for easier comparison
        print(computed_centralizer)
        self.computed_center = computed_centralizer
        computed_center_string = toString(computed_centralizer)
        original_center_string = toString(self.actual_center)
        print("Computed Centralizer:", computed_center_string)
        print("Original Stabilizer Group:", original_center_string)
        self.assertEqual(set(computed_center_string), set(original_center_string), "Centralizer does not match original stabilizer group")
        #self.assertEqual(original_set, centralizer_set, "Centralizer does not match original stabilizer group")
        
    def test_stabilizer_purity(self):
        purity = filtered_purity(self.actual_center, self.test_data)
        self.assertGreaterEqual(purity, 0, "Purity should be non-negative")
        #The purity should be strictly 1 for noise free data
        self.assertAlmostEqual(purity, 1.0, "Purity should be 1 for a pure stabilizer state")
    
    #def test_purity_noisy(self):
    #    stabilizer_purity_noisy = filtered_purity(self.actual_center, self.noisy_data)
    #    print(stabilizer_purity_noisy)
    #    normal_purity = getParity(self.noisy_data)
    #    print(normal_purity)
    #    self.assertGreaterEqual(stabilizer_purity_noisy, 0, "Purity should be non-negative")
    #    self.assertLess(stabilizer_purity_noisy, 1.0, "Purity should be less than 1 for noisy data")
       
    # def test_rought_purity(self):
    #     # Compare filtered_purity to a reference implementation
    #     reference_purity = filtered_purity_reference(self.actual_center, self.test_data)
    #     test_purity = filtered_purity(self.actual_center, self.test_data)
    #     self.assertAlmostEqual(test_purity, reference_purity, places=6, msg="Filtered purity does not match reference implementation")
        
        


if __name__ == '__main__':
    unittest.main()