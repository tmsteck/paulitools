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
from group import row_reduce, inner_product, null_space, radical, differences
from ptgalois.converter import toZX as toZX_pt
from ptgalois.group import inner_product as inner_product_pt
from ptgalois.group import radical as radical_pt
from galois import GF2
#from paulitools.group import 
#from paulitools import toZX, toString, generator  as toZX_old, toString_old, generator
import ptgalois as pt
# Unit tests using unittest framework

class TestDifferences(unittest.TestCase):
    def test_differences_single(self):
        input = toZX(["XX", "YY", "ZZ"])
        expected_output = toZX(["ZZ", "XX", "YY"])
        output = differences(input)
        np.testing.assert_array_equal(output, expected_output)
        
    def test_differences_double(self):
        input1 = toZX(["XX", "YY", "ZZ"])
        input2 = toZX(["XZ", "YX", "ZY"])
        expected_output = toZX(["IY", "IZ", "IX"])
        output = differences(input1, input2)
        np.testing.assert_array_equal(output, expected_output)
    
    def test_assert_double(self):
        input1 = toZX(["XX", "YY", "ZZ"])
        input2 = toZX(["XZ", "YX"])
        with self.assertRaises(AssertionError) as e:
            output = differences(input1, input2)
            # Check that the error message is as expected
            self.assertEqual(str(e), "paulis and paulis2 must have the same shape")
        input3 = toZX(["XXX", "YYY", "ZZZ"])
        with self.assertRaises(AssertionError) as e:
            output = differences(input2, input3)
            self.assertEqual(str(e), "paulis and paulis2 must have the same k value")
        
    def test_pad_double(self):
        input1 = toZX(["XX", "YY", "ZZ"])
        input2 = toZX(["X", "Y", "Z"])
        expected_output = toZX(["IX", "IY", "IZ"])
        output = differences(input1, right_pad(input2, 2))
        np.testing.assert_array_equal(output, expected_output)


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

    def test_L1(self):
        pt_output = radical_pt(self.set_2_pt)
        output = radical(self.set_2)
        np.testing.assert_array_equal(output, pt_output)

    def test_L2(self):
        pt_output = radical_pt(self.set_3_pt)
        output = radical(self.set_3)
        np.testing.assert_array_equal(output, pt_output)

    def test_random_equivalence(self):
        # Compare ptgalois and numba radical for random 16-qubit, 8-string sets
        n_qubits = 16
        n_paulis = 8
        n_trials = 20
        for _ in range(n_trials):
            # Each Pauli string is a random binary string of length 2*n_qubits (ZX form)
            paulis_bin = np.random.randint(0, 2, (n_paulis, 2*n_qubits), dtype=np.uint8)
            # Convert to integer representation (sign bit = 0)
            paulis_int = np.zeros(n_paulis, dtype=GLOBAL_INTEGER)
            for i in range(n_paulis):
                val = 0
                for j in range(2*n_qubits):
                    if paulis_bin[i, j]:
                        val |= (1 << (2*n_qubits - j))
                paulis_int[i] = val
            zx_numba = np.zeros(n_paulis+1, dtype=GLOBAL_INTEGER)
            zx_numba[0] = n_qubits
            zx_numba[1:] = paulis_int
            zx_ptgalois = pt.converter.toZX([pt.converter.toString(val) for val in paulis_int])
            # Compare radical
            rad_numba = radical(zx_numba)
            rad_ptgalois = radical_pt(zx_ptgalois)
            # Compare as sets of rows (order may differ)
            self.assertEqual(rad_numba.shape, rad_ptgalois.shape)
            if rad_numba.shape[0] > 0:
                self.assertTrue(np.all(np.sort(rad_numba, axis=0) == np.sort(rad_ptgalois, axis=0)))

    def test_radical_speed(self):
        import time
        n_qubits = 16
        n_paulis = 8
        n_trials = 1001
        zx_numba_list = []
        zx_ptgalois_list = []
        for _ in range(n_trials):
            paulis_bin = np.random.randint(0, 2, (n_paulis, 2*n_qubits), dtype=np.uint8)
            paulis_int = np.zeros(n_paulis, dtype=GLOBAL_INTEGER)
            for i in range(n_paulis):
                val = 0
                for j in range(2*n_qubits):
                    if paulis_bin[i, j]:
                        val |= (1 << (2*n_qubits - j))
                paulis_int[i] = val
            zx_numba = np.zeros(n_paulis+1, dtype=GLOBAL_INTEGER)
            zx_numba[0] = n_qubits
            zx_numba[1:] = paulis_int
            zx_numba_list.append(zx_numba)
            zx_ptgalois = pt.converter.toZX([pt.converter.toString(val) for val in paulis_int])
            zx_ptgalois_list.append(zx_ptgalois)
        # Warm up Numba
        radical(zx_numba_list[0])
        # Time Numba radical
        t0 = time.time()
        for zx in zx_numba_list:
            radical(zx)
        t1 = time.time()
        # Time ptgalois radical
        for zx in zx_ptgalois_list:
            radical_pt(zx)
        t2 = time.time()
        print(f"\nRadical speed test (n_trials={n_trials}, n_qubits={n_qubits}, n_paulis={n_paulis}):")
        print(f"Numba radical:   {t1-t0:.4f} s")
        print(f"ptgalois radical: {t2-t1:.4f} s")

if __name__ == '__main__':
    unittest.main()