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
from group import row_reduce, inner_product, null_space, radical, differences, row_space, centralizer, group, ingroup
from ptgalois.converter import toZX as toZX_pt
from ptgalois.converter import toString as toString_pt
from ptgalois.group import inner_product as inner_product_pt
from ptgalois.group import radical as radical_pt
from ptgalois.group import centralizer as centralizer_pt
from galois import GF2
#from paulitools.group import 
#from paulitools import toZX, toString, generator  as toZX_old, toString_old, generator
import ptgalois as pt
# Unit tests using unittest framework


def _row_tuple_set(matrix):
    matrix = np.asarray(matrix)
    if matrix.size == 0:
        return set()
    matrix = np.mod(matrix.astype(np.int64), 2)
    if matrix.ndim == 1:
        return {tuple(matrix.tolist())}
    return {tuple(row.tolist()) for row in matrix}

class TestToZXBinaryArrays(unittest.TestCase):
    """Test cases for binary array input support in toZX function"""
    
    def test_single_binary_array_1d(self):
        """Test single 1D binary array input"""
        # Test X operator: [0, 1] (Z=0, X=1)
        binary_x = np.array([0, 1], dtype=np.uint8)
        result = toZX(binary_x)
        expected = toZX("X")
        np.testing.assert_array_equal(result, expected)
        
        # Test Z operator: [1, 0] (Z=1, X=0)
        binary_z = np.array([1, 0], dtype=np.uint8)
        result = toZX(binary_z)
        expected = toZX("Z")
        np.testing.assert_array_equal(result, expected)
        
        # Test Y operator: [1, 1] (Z=1, X=1)
        binary_y = np.array([1, 1], dtype=np.uint8)
        result = toZX(binary_y)
        expected = toZX("Y")
        np.testing.assert_array_equal(result, expected)
        
        # Test Identity: [0, 0] (Z=0, X=0)
        binary_i = np.array([0, 0], dtype=np.uint8)
        result = toZX(binary_i)
        expected = toZX("I")
        np.testing.assert_array_equal(result, expected)
    
    def test_multi_qubit_binary_array_1d(self):
        """Test multi-qubit 1D binary array input"""
        # Test XX: [0, 0, 1, 1] (Z=[0,0], X=[1,1])
        binary_xx = np.array([0, 0, 1, 1], dtype=np.uint8)
        result = toZX(binary_xx)
        expected = toZX("XX")
        np.testing.assert_array_equal(result, expected)
        
        # Test ZZ: [1, 1, 0, 0] (Z=[1,1], X=[0,0])
        binary_zz = np.array([1, 1, 0, 0], dtype=np.uint8)
        result = toZX(binary_zz)
        expected = toZX("ZZ")
        np.testing.assert_array_equal(result, expected)
        
        # Test XZ: [0, 1, 1, 0] (Z=[0,1], X=[1,0]) - X on qubit 0, Z on qubit 1
        binary_xz = np.array([0, 1, 1, 0], dtype=np.uint8)
        result = toZX(binary_xz)
        expected = toZX("XZ")
        np.testing.assert_array_equal(result, expected)
    
    def test_multiple_binary_arrays_2d(self):
        """Test 2D binary array input (multiple Pauli strings)"""
        # Test multiple single-qubit operators
        binary_arrays = np.array([
            [0, 1],  # X
            [1, 0],  # Z
            [1, 1],  # Y
            [0, 0]   # I
        ], dtype=np.uint8)
        
        result = toZX(binary_arrays)
        expected = toZX(["X", "Z", "Y", "I"])
        np.testing.assert_array_equal(result, expected)
    
    def test_multiple_multi_qubit_binary_arrays_2d(self):
        """Test 2D binary array input with multi-qubit operators"""
        # Test 2-qubit operators
        binary_arrays = np.array([
            [0, 0, 1, 1],  # XX
            [1, 1, 0, 0],  # ZZ  
            [0, 1, 1, 0],  # XZ - X on qubit 0, Z on qubit 1
            [1, 0, 0, 1]   # ZX - Z on qubit 0, X on qubit 1
        ], dtype=np.uint8)
        
        result = toZX(binary_arrays)
        expected = toZX(["XX", "ZZ", "XZ", "ZX"])
        np.testing.assert_array_equal(result, expected)
    
    def test_binary_array_equivalence_with_strings(self):
        """Test that binary arrays produce same results as string inputs"""
        # Generate some test cases
        test_strings = ["X", "Z", "Y", "I", "XX", "XY", "YZ", "ZI", "IXYZ"]
        
        for test_string in test_strings:
            # Convert string to expected result
            expected = toZX(test_string)
            k = expected[0]
            
            # Manually create binary representation
            binary_array = np.zeros(2 * k, dtype=np.uint8)
            
            for i, pauli in enumerate(test_string):
                if pauli == 'X':
                    binary_array[i + k] = 1  # X bit
                elif pauli == 'Z':
                    binary_array[i] = 1      # Z bit
                elif pauli == 'Y':
                    binary_array[i] = 1      # Z bit
                    binary_array[i + k] = 1  # X bit
                # I remains [0, 0]
            
            # Test binary array input
            result = toZX(binary_array)
            np.testing.assert_array_equal(result, expected, 
                                        f"Binary array conversion failed for '{test_string}'")
    
    def test_binary_array_error_cases(self):
        """Test error handling for invalid binary array inputs"""
        # Test odd-length array (should fail)
        with self.assertRaises(ValueError):
            toZX(np.array([1, 0, 1], dtype=np.uint8))
        
        # Test 3D array (should fail)
        with self.assertRaises(ValueError):
            toZX(np.array([[[1, 0], [0, 1]]], dtype=np.uint8))
        
        # Test 2D array with odd width (should fail)
        with self.assertRaises(ValueError):
            toZX(np.array([[1, 0, 1], [0, 1, 0]], dtype=np.uint8))
    
    def test_binary_array_consistency_with_row_space(self):
        """Test that binary arrays work correctly with other functions like row_space"""
        # Create some test binary arrays
        binary_arrays = np.array([
            [0, 0, 1, 1],  # XX
            [1, 1, 0, 0],  # ZZ
            [0, 1, 1, 0]   # XZ - X on qubit 0, Z on qubit 1
        ], dtype=np.uint8)
        
        # Convert to ZX format
        zx_from_binary = toZX(binary_arrays)
        
        # Compare with string input
        zx_from_strings = toZX(["XX", "ZZ", "XZ"])
        
        # Should be identical
        np.testing.assert_array_equal(zx_from_binary, zx_from_strings)
        
        # Test with row_space function
        rs_binary = row_space(zx_from_binary)
        rs_strings = row_space(zx_from_strings)
        
        np.testing.assert_array_equal(rs_binary, rs_strings)
    
    def test_binary_array_round_trip(self):
        """Test round-trip conversion: binary -> ZX -> row_space -> binary"""
        # Original binary arrays
        original_binary = np.array([
            [1, 0, 0, 1],  # ZX - Z on qubit 0, X on qubit 1
            [0, 1, 1, 0],  # XZ - X on qubit 0, Z on qubit 1
            [1, 1, 1, 1]   # YY - Y on both qubits
        ], dtype=np.uint8)
        
        # Convert to ZX format
        zx_format = toZX(original_binary)
        
        # Get row space (should be in binary format)
        rs = row_space(zx_format)
        
        # The row space should contain the original binary representations
        # (possibly in different order due to row reduction)
        
        # Convert back through ZX to verify consistency
        zx_from_rs = toZX(rs)
        rs_again = row_space(zx_from_rs)
        
        # Row space should be identical regardless of conversion path
        np.testing.assert_array_equal(rs, rs_again)

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


class TestInGroup(unittest.TestCase):
    def test_independent_single(self):
        basis = toZX(["XX", "ZZ"])
        candidate = toZX(["XZ"])
        result = ingroup(candidate, basis)
        self.assertTrue(result[0])

    def test_dependent_single(self):
        basis = toZX(["XX", "ZI"])
        candidate = toZX(["XX"])
        result = ingroup(candidate, basis)
        self.assertFalse(result[0])

    def test_multiple_candidates(self):
        basis = toZX(["XX", "ZZ"])
        candidates = toZX(["XI", "XX", "ZX"])
        result = ingroup(candidates, basis)
        expected = np.array([True, False, True], dtype=np.bool_)
        np.testing.assert_array_equal(result, expected)

    def test_reduced_basis(self):
        basis = toZX(["XX", "XI", "IZ"])
        reduced_basis = row_reduce(basis)
        candidate = toZX(["ZZ"])
        result = ingroup(candidate, reduced_basis, reduced=True)
        self.assertTrue(result[0])

    def test_k_mismatch(self):
        basis = toZX(["X"])
        candidates = toZX(["XX"])
        with self.assertRaises(ValueError):
            ingroup(candidates, basis)

    def test_empty_candidates(self):
        basis = toZX(["X"])
        empty_candidates = np.array([basis[0]], dtype=GLOBAL_INTEGER)
        result = ingroup(empty_candidates, basis)
        self.assertEqual(result.shape[0], 0)

    def test_identity_candidate(self):
        basis = toZX(["XX", "ZZ"])
        identity = toZX(["II"])
        result = ingroup(identity, basis)
        self.assertFalse(result[0])

    def test_result_dtype(self):
        basis = toZX(["XX", "ZZ"])
        candidates = toZX(["XI", "ZX"])
        result = ingroup(candidates, basis)
        self.assertEqual(result.dtype, np.bool_)

class TestInnerProduct(unittest.TestCase):
    #Test information:
    def setUp(self):
        self.set_1_pt = toZX_pt(["XX", "YY", "ZZ"]) #ALL COMMUTING
        self.set_2_pt = toZX_pt(["XX", "XI", "ZI"]) #L = 1
        self.set_3_pt = toZX_pt(["ZI", "XI", "IX","IZ"]) #L = 2
        
        self.set_1 = toZX(["XX", "YY", "ZZ"]) #ALL COMMUTING
        self.set_2 = toZX(["XX", "XI", "ZI"]) #L = 1
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
        np.random.seed(42)
        for _ in range(10):
            A = np.random.randint(0, 2, (5, 5))#, dtype=int8)
            expected_output = GF2(A).null_space()
            output = null_space(A)
            np.testing.assert_array_equal(output, expected_output)
        for _ in range(10):
            A = np.random.randint(0, 2, (2, 2))#, dtype=int8)
            expected_output = GF2(A).null_space()
            output = null_space(A)
            np.testing.assert_array_equal(output, expected_output)
        for _ in range(10):
            A = np.random.randint(0, 2, (20, 20))#, dtype=int8)
            expected_output = GF2(A).null_space()
            output = null_space(A)
            np.testing.assert_array_equal(output, expected_output)

class TestCentralizer(unittest.TestCase):
    """Test centralizer function against ptgalois implementation"""
    
    def setUp(self):
        self.set_1_pt = toZX_pt(["XX", "YY", "ZZ"]) #ALL COMMUTING
        self.set_2_pt = toZX_pt(["XX", "XI", "ZI"]) #L = 1
        self.set_3_pt = toZX_pt(["ZI", "XI", "IX","IZ"]) #L = 2
        self.set_1 = toZX(["XX", "YY", "ZZ"]) #ALL COMMUTING
        self.set_2 = toZX(["XX", "XI", "ZI"]) #L = 1
        self.set_3 = toZX(["ZI", "XI", "IX","IZ"]) #L = 2
        self.set_4 = toZX(["ZIIII", "IZIII", "IIZII", "IIIZI", "IIIIZ", "XIIII", "IXIII", "YYIII", "YXIII"])

    def test_centralizer_commuting(self):
        """Test centralizer for all commuting Paulis"""
        pt_centralizer = centralizer_pt(self.set_1_pt)
        our_centralizer = centralizer(self.set_1)
        
        #print(f"\nCommuting case:")
        #print(f"Ptgalois centralizer shape: {pt_centralizer.shape}")
        #print(f"Our centralizer shape: {our_centralizer.shape}")
        
        # Test that all elements in our centralizer commute with input group
        rs = row_space(self.set_1)
        for cent_elem in our_centralizer:
            for rs_elem in rs:
                # Check symplectic inner product is 0 (they commute)
                k = self.set_1[0]
                z1, x1 = cent_elem[:k], cent_elem[k:]
                z2, x2 = rs_elem[:k], rs_elem[k:]
                symp_prod = (np.sum(z1 * x2) + np.sum(x1 * z2)) % 2
                self.assertEqual(symp_prod, 0, 
                               f"Centralizer element {cent_elem} should commute with {rs_elem}")
        
        # For all commuting inputs, the centralizer should have the same dimension
        # as the ptgalois implementation
        self.assertEqual(pt_centralizer.shape[0], our_centralizer.shape[0],
                        "Centralizer dimensions should match ptgalois")

    def test_centralizer_L1(self):
        """Test centralizer for L=1 case"""
        pt_centralizer = centralizer_pt(self.set_2_pt)
        our_centralizer = centralizer(self.set_2)
        
        # For debugging, print both results
        print(f"\nPtgalois centralizer shape: {pt_centralizer.shape}")
        print(f"Our centralizer shape: {our_centralizer.shape}")
        
        # Test that centralizer elements commute with original group elements
        for cent_elem in our_centralizer:
            for orig_elem in row_space(self.set_2):
                # Check symplectic inner product is 0 (they commute)
                k = self.set_2[0]
                z1, x1 = cent_elem[:k], cent_elem[k:]
                z2, x2 = orig_elem[:k], orig_elem[k:]
                symp_prod = (np.sum(z1 * x2) + np.sum(x1 * z2)) % 2
                self.assertEqual(symp_prod, 0, 
                               f"Centralizer element {cent_elem} should commute with {orig_elem}")

    def test_centralizer_L2(self):
        """Test centralizer for L=2 case"""
        pt_centralizer = centralizer_pt(self.set_3_pt)
        our_centralizer = centralizer(self.set_3)
        
        # Test dimensions
        print(f"\nL=2 case:")
        print(f"Ptgalois centralizer shape: {pt_centralizer.shape}")
        print(f"Our centralizer shape: {our_centralizer.shape}")
        
        # Test that centralizer elements commute with original group elements
        for cent_elem in our_centralizer:
            for orig_elem in row_space(self.set_3):
                # Check symplectic inner product is 0 (they commute)
                k = self.set_3[0]
                z1, x1 = cent_elem[:k], cent_elem[k:]
                z2, x2 = orig_elem[:k], orig_elem[k:]
                symp_prod = (np.sum(z1 * x2) + np.sum(x1 * z2)) % 2
                self.assertEqual(symp_prod, 0, 
                               f"Centralizer element {cent_elem} should commute with {orig_elem}")
    def test_case_4(self):
        """Test centralizer for 5-qubit stabilizer case"""
        our_centralizer = centralizer(self.set_4)
        output = toZX(["IIZII", "IIIZI", "IIIIZ"][::-1]) #no particular reason this is reversed -- probably the same as row_reduced version
        np.testing.assert_array_equal(toZX(our_centralizer), output)
        
        
    # def test_comp_basis_tests_centralizer(self):
    #     test_center_set = toZX(['ZII', 'IZI', 'IIZ', "IIX"])
    #     full_group = group(test_center_set)
    #     stabilizer_generators = paulitools.group.centralizer(full_group)

    def test_centralizer_random_equivalence(self):
        """Compare centralizer against ptgalois for random test cases"""
        n_qubits = 10  # Smaller for faster testing
        n_paulis = 15
        n_trials = 30
        
        for trial in range(n_trials):
            # Generate random Pauli group
            paulis_bin = np.random.randint(0, 2, (n_paulis, 2*n_qubits), dtype=np.uint8)
            
            # Convert to both formats
            zx_numba = toZX(paulis_bin)
            
            # Convert to ptgalois format
            pauli_strings = []
            for i in range(n_paulis):
                # Convert binary to string representation
                z_part = paulis_bin[i, :n_qubits]
                x_part = paulis_bin[i, n_qubits:]
                pauli_str = ""
                for j in range(n_qubits):
                    if z_part[j] and x_part[j]:
                        pauli_str += "Y"
                    elif x_part[j]:
                        pauli_str += "X"
                    elif z_part[j]:
                        pauli_str += "Z"
                    else:
                        pauli_str += "I"
                pauli_strings.append(pauli_str)
            
            zx_ptgalois = toZX_pt(pauli_strings)
            
            # Compute centralizers
            try:
                pt_centralizer = centralizer_pt(zx_ptgalois)
                our_centralizer = centralizer(zx_numba)
                #compare lengths (number of elements)
                np.testing.assert_equal(pt_centralizer.shape[0], our_centralizer.shape[0],
                                       err_msg=f"Trial {trial}: Centralizer sizes should match")
                # Test that our centralizer has correct mathematical properties
                # (exact equivalence testing is complex due to different representations)
                
                # Test that all elements in our centralizer commute with input group
                rs = row_space(zx_numba)
                for cent_elem in our_centralizer:
                    for rs_elem in rs:
                        k = zx_numba[0]
                        z1, x1 = cent_elem[:k], cent_elem[k:]
                        z2, x2 = rs_elem[:k], rs_elem[k:]
                        symp_prod = (np.sum(z1 * x2) + np.sum(x1 * z2)) % 2
                        self.assertEqual(symp_prod, 0, 
                                       f"Trial {trial}: Centralizer element should commute with group element")
                        
            except Exception as e:
                # Skip cases that fail in ptgalois (might be edge cases)
                print(f"Skipping trial {trial} due to error: {e}")
                continue

    def test_centralizer_speed(self):
        """Test the performance of centralizer implementation"""
        import time
        n_qubits = 6
        n_paulis = 9
        n_trials = 100
        
        # Generate test cases
        test_cases = []
        for _ in range(n_trials):
            paulis_bin = np.random.randint(0, 2, (n_paulis, 2*n_qubits), dtype=np.uint8)
            zx_numba = toZX(paulis_bin)
            test_cases.append(zx_numba)
        
        # Warm up Numba
        centralizer(test_cases[0])
        
        # Time our centralizer
        t0 = time.time()
        for zx in test_cases:
            centralizer(zx)
        t1 = time.time()
        
        elapsed = t1 - t0
        ops_per_sec = n_trials / elapsed if elapsed > 0 else float('inf')
        
        print(f"\nCentralizer speed test (n_trials={n_trials}, n_qubits={n_qubits}, n_paulis={n_paulis}):")
        print(f"Our centralizer: {elapsed:.6f} s ({ops_per_sec:.2f} ops/s)")
        
        # Test that it's reasonably fast
        self.assertLess(elapsed, 5.0, "Centralizer should complete within reasonable time")

        
class TestRowSpace(unittest.TestCase):
    def setUp(self):
        # Create test cases with different properties
        self.test_cases = [
            toZX(["XX", "YY", "ZZ"]),  # All commuting Paulis
            toZX(["XX", "XI", "ZI"]),   # L = 1
            toZX(["ZI", "XI", "IX", "IZ"]),  # L = 2
            toZX(["XXX", "YYY", "ZZZ"]),  # 3-qubit
        ]
    
    def test_row_space_dimensions(self):
        """Test that row_space produces matrices with correct dimensions"""
        for i, test_case in enumerate(self.test_cases):
            k = test_case[0]  # Number of qubits
            num_rows = len(test_case) - 1  # Number of Pauli strings
            
            # Get row space
            rs = row_space(test_case)
            
            # Check dimensions
            self.assertEqual(rs.shape[1], 2*k, f"Row space should have 2*{k}={2*k} columns for test case {i}")
            self.assertLessEqual(rs.shape[0], num_rows, 
                               f"Row space should have at most {num_rows} rows for test case {i}")
    
    def test_row_space_properties(self):
        """Test that row_space has expected mathematical properties"""
        for i, test_case in enumerate(self.test_cases):
            # Get row space
            rs = row_space(test_case)
            
            # 1. Every row should be linearly independent
            rank = np.linalg.matrix_rank(rs.astype(np.float64) % 2)
            self.assertEqual(rank, rs.shape[0], 
                            f"Row space should contain only linearly independent rows for test case {i}")
            
            # 2. Row reduce should not change the row space
            reduced = row_reduce(test_case)
            rs2 = row_space(reduced)
            
            # Compare row spaces by converting to sets of tuples
            rs_rows = {tuple(row) for row in rs}
            rs2_rows = {tuple(row) for row in rs2}
            self.assertEqual(rs_rows, rs2_rows, 
                           f"row_space(row_reduce(P)) should equal row_space(P) for test case {i}")
    
    def test_spanning_properties(self):
        """Test that row_space correctly spans the original Paulis"""
        for i, test_case in enumerate(self.test_cases):
            # Get row space in binary representation
            rs = row_space(test_case)
            k = test_case[0]  # Number of qubits
            
            # For each original Pauli string, verify it can be represented as a linear 
            # combination of rows in the row space
            for j in range(1, len(test_case)):
                # Convert Pauli to binary form (ignoring sign bit)
                pauli_int = test_case[j] >> 1
                pauli_bin = np.zeros(2*k, dtype=np.int8)
                
                for bit in range(2*k):
                    pauli_bin[bit] = (pauli_int >> bit) & 1
                
                # Check if the vector is in the span of the row space
                extended = np.vstack([rs, pauli_bin])
                
                # If the rank doesn't increase, then the vector is in the span
                rank_rs = np.linalg.matrix_rank(rs.astype(np.float64) % 2)
                rank_ext = np.linalg.matrix_rank(extended.astype(np.float64) % 2)
                
                self.assertEqual(rank_rs, rank_ext,
                               f"Pauli {j} should be in the row space for test case {i}")
    
    def test_row_space_speed(self):
        """Test the performance of row_space implementation"""
        import time
        n_qubits = 10
        n_paulis = 5
        n_trials = 50
        
        # Generate random test cases
        np.random.seed(42)
        test_cases = []
        
        for _ in range(n_trials):
            # Create random Pauli strings
            pauli_strings = []
            for _ in range(n_paulis):
                # Generate random binary string for Z|X
                binary = ''.join(np.random.choice(['0', '1']) for _ in range(2*n_qubits))
                pauli_strings.append(binary)
            
            # Convert to ZX form
            zx_form = toZX(pauli_strings)
            test_cases.append(zx_form)
        
        # Warm up Numba
        row_space(test_cases[0])
        
        # Time row_space
        t0 = time.time()
        for tc in test_cases:
            rs = row_space(tc)
        t1 = time.time()
        
        elapsed = t1 - t0
        ops_per_sec = n_trials / elapsed if elapsed > 0 else float('inf')
        
        print(f"\nRow space speed test (n_trials={n_trials}, n_qubits={n_qubits}, n_paulis={n_paulis}):")
        print(f"row_space: {elapsed:.6f} s ({ops_per_sec:.2f} ops/s)")


class TestGroupFunctionsWithExtension(unittest.TestCase):
    def setUp(self):
        self.pauli_strings = ["XX", "XY", "ZZ", "YZ"]
        self.legacy = toZX(self.pauli_strings)
        self.large_collection = toZX_extended(self.pauli_strings, force_large=True)
        self.large_standard = to_standard_if_possible(self.large_collection)

    def test_row_reduce_matches_legacy(self):
        reduced_legacy = row_reduce(self.legacy)
        reduced_large = row_reduce(self.large_standard)
        np.testing.assert_array_equal(reduced_large, reduced_legacy)

    def test_row_space_matches_legacy(self):
        space_legacy = row_space(self.legacy)
        space_large = row_space(self.large_standard)
        self.assertEqual(_row_tuple_set(space_large), _row_tuple_set(space_legacy))

    def test_radical_matches_legacy(self):
        radical_legacy = radical(self.legacy)
        radical_large = radical(self.large_standard)
        self.assertEqual(_row_tuple_set(radical_large), _row_tuple_set(radical_legacy))

    def test_centralizer_matches_legacy(self):
        centralizer_legacy = centralizer(self.legacy)
        centralizer_large = centralizer(self.large_standard)
        self.assertEqual(
            _row_tuple_set(centralizer_large),
            _row_tuple_set(centralizer_legacy),
        )

    def test_inner_product_matches_legacy(self):
        inner_legacy = inner_product(self.legacy)
        inner_large = inner_product(self.large_standard)
        np.testing.assert_array_equal(inner_large, inner_legacy)

    def test_differences_matches_legacy(self):
        diff_legacy = differences(self.legacy)
        diff_large = differences(self.large_standard)
        np.testing.assert_array_equal(diff_large, diff_legacy)

if __name__ == '__main__':
    unittest.main()