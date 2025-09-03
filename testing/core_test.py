import unittest
import numpy as np
# Unit tests using unittest framework
#Import everything from the /src directory:
import sys
import os
from numba.core.errors import NumbaValueError
import time

#sys.path.append('../src')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from core import toZX, toString, concatenate_ZX, right_pad, left_pad, symplectic_inner_product, commutes, GLOBAL_INTEGER, bsip_array, commute_array_fast, symplectic_inner_product_int



# Unit tests using unittest framework
class TestConvertToIntegerRepresentation(unittest.TestCase):
    def test_pauli_string_input(self):
        input_data = "XYZI"
        expected_output = np.array([4, 108], dtype=GLOBAL_INTEGER)
        output = toZX(input_data)
        np.testing.assert_array_equal(output, expected_output)

    def test_list_of_tuples_input(self):
        input_data = [('X', 0), ('Y', 1), ('Z', 2)]
        expected_output = np.array([3, 60], dtype=GLOBAL_INTEGER)
        output = toZX(input_data)
        np.testing.assert_array_equal(output, expected_output)

    def test_single_paulis(self):
        inputs_and_expected = [
            ('X', np.array([1, 4], dtype=GLOBAL_INTEGER)),
            ('Y', np.array([1, 6], dtype=GLOBAL_INTEGER)),
            ('Z', np.array([1, 2], dtype=GLOBAL_INTEGER))
        ]
        for input_data, expected_output in inputs_and_expected:
            output = toZX(input_data)
            np.testing.assert_array_equal(output, expected_output)

    def test_list_of_pauli_strings(self):
        inputs_and_expected = [
            (['X', 'Z'], np.array([1, 4, 2], dtype=GLOBAL_INTEGER)),
            (['XX'], np.array([2, 24], dtype=GLOBAL_INTEGER)),
            (['XX', 'YY'], np.array([2, 24, 30], dtype=GLOBAL_INTEGER)),
            (['IX', 'YI'], np.array([2, 16, 10], dtype=GLOBAL_INTEGER))
        ]
        for input_data, expected_output in inputs_and_expected:
            output = toZX(input_data)
            np.testing.assert_array_equal(output, expected_output)

    def test_identity_only_pauli_string(self):
        input_data = "IIII"
        expected_output = np.array([4, 0], dtype=GLOBAL_INTEGER)
        output = toZX(input_data)
        np.testing.assert_array_equal(output, expected_output)

    def test_invalid_input_data_type(self):
        with self.assertRaises(ValueError):
            toZX(123)
        with self.assertRaises(ValueError):
            toZX({'X': 0})

    def test_invalid_pauli_character(self):
        with self.assertRaises(ValueError):
            toZX("ABCD")
        with self.assertRaises(ValueError):
            toZX([('A', 0), ('B', 1)])
    
    def test_multi_length_input(self):
        input1 = ['X', 'YY', 'ZZZ']
        input2 = ['X', 'IY']
        input3 = ['Y','XIYZ' ]
        ZX1 = toZX(input1)
        ZX2 = toZX(input2)
        ZX3 = toZX(input3)
        self.assertEqual(ZX1[0], 3)
        self.assertEqual(ZX2[0], 2)
        self.assertEqual(ZX3[0], 4)

    def test_binary_string_single_input(self):
        """Test binary string format: Z|X representation"""
        # Test case: IXYZ -> Z bits: 0011, X bits: 0110 -> "00110110"
        input_data = "00110110"
        expected_output = np.array([4, 216], dtype=GLOBAL_INTEGER)  # IXYZ
        output = toZX(input_data)
        np.testing.assert_array_equal(output, expected_output)

    def test_binary_string_basic_paulis(self):
        """Test basic Pauli operators in binary format"""
        test_cases = [
            ("01", np.array([1, 4], dtype=GLOBAL_INTEGER)),    # X: Z=0, X=1
            ("11", np.array([1, 6], dtype=GLOBAL_INTEGER)),    # Y: Z=1, X=1  
            ("10", np.array([1, 2], dtype=GLOBAL_INTEGER)),    # Z: Z=1, X=0
            ("00", np.array([1, 0], dtype=GLOBAL_INTEGER)),    # I: Z=0, X=0
        ]
        for binary_input, expected in test_cases:
            with self.subTest(binary_input=binary_input):
                output = toZX(binary_input)
                np.testing.assert_array_equal(output, expected)

    def test_binary_string_multi_qubit(self):
        """Test multi-qubit binary strings"""
        test_cases = [
            ("0011", np.array([2, 24], dtype=GLOBAL_INTEGER)),   # XX: Z=00, X=11
            ("1111", np.array([2, 30], dtype=GLOBAL_INTEGER)),   # YY: Z=11, X=11
            ("1100", np.array([2, 6], dtype=GLOBAL_INTEGER)),    # ZZ: Z=11, X=00
            ("0000", np.array([2, 0], dtype=GLOBAL_INTEGER)),    # II: Z=00, X=00
        ]
        for binary_input, expected in test_cases:
            with self.subTest(binary_input=binary_input):
                output = toZX(binary_input)
                np.testing.assert_array_equal(output, expected)

    def test_binary_string_list_input(self):
        """Test list of binary strings"""
        input_data = ["01", "11", "10", "00"]  # X, Y, Z, I
        expected_output = np.array([1, 4, 6, 2, 0], dtype=GLOBAL_INTEGER)
        output = toZX(input_data)
        np.testing.assert_array_equal(output, expected_output)

    def test_binary_string_ptgalois_example(self):
        """Test the specific example from ptgalois: '11000110' = IXYZ"""
        # IXYZ: I(00), X(01), Y(11), Z(10)
        # Z bits: 0011 (reading right to left: I=0, X=0, Y=1, Z=1)
        # X bits: 0110 (reading right to left: I=0, X=1, Y=1, Z=0)
        # Combined: "00110110"
        input_data = "00110110"
        
        # Verify it matches the expected IXYZ representation
        expected_ixyz = toZX("IXYZ")
        output = toZX(input_data)
        np.testing.assert_array_equal(output, expected_ixyz)

    def test_binary_string_invalid_length(self):
        """Test that invalid binary string lengths raise errors"""
        with self.assertRaises(ValueError):
            toZX("101")  # Odd length - can't be split into Z|X
        with self.assertRaises(ValueError):
            toZX("10101")  # Odd length
            
    def test_binary_string_mixed_with_regular(self):
        """Test that mixing binary strings with regular strings in lists fails appropriately"""
        # This should work - all binary strings
        binary_list = ["01", "11", "10"]
        output = toZX(binary_list)
        self.assertEqual(output[0], 1)  # 1 qubit each
        
        # Regular Pauli strings should also work
        pauli_list = ["X", "Y", "Z"]
        output2 = toZX(pauli_list)
        self.assertEqual(output2[0], 1)  # 1 qubit each

class TestConvertFromIntegerRepresentation(unittest.TestCase):
    def test_individual_paulis(self):
        inputs_and_expected = [
            (np.array([1, 2], dtype=GLOBAL_INTEGER), '+Z'),
            (np.array([1, 4], dtype=GLOBAL_INTEGER), '+X'),
            (np.array([1, 6], dtype=GLOBAL_INTEGER), '+Y')
        ]
        for input_data, expected_output in inputs_and_expected:
            self.assertEqual(toString(input_data), expected_output)

    def test_double_paulis(self):
        inputs_and_expected = [
            (np.array([2, 24], dtype=GLOBAL_INTEGER), '+XX'),
            (np.array([2, 30], dtype=GLOBAL_INTEGER), '+YY'),
            (np.array([2, 6], dtype=GLOBAL_INTEGER), '+ZZ')
        ]
        for input_data, expected_output in inputs_and_expected:
            self.assertEqual(toString(input_data), expected_output)

    def test_negative_double_paulis(self):
        inputs_and_expected = [
            (np.array([2, 25], dtype=GLOBAL_INTEGER), '-XX'),
            (np.array([2, 31], dtype=GLOBAL_INTEGER), '-YY'),
            (np.array([2, 7], dtype=GLOBAL_INTEGER), '-ZZ')
        ]
        for input_data, expected_output in inputs_and_expected:
            self.assertEqual(toString(input_data), expected_output)

    def test_mixed_pauli_strings(self):
        inputs_and_expected = [
            (np.array([4, 216], dtype=GLOBAL_INTEGER), '+IXYZ'),
            (np.array([4, 217], dtype=GLOBAL_INTEGER), '-IXYZ'),
            (np.array([4, 198], dtype=GLOBAL_INTEGER), '+ZYXI'),
            (np.array([4, 199], dtype=GLOBAL_INTEGER), '-ZYXI')
        ]
        for input_data, expected_output in inputs_and_expected:
            self.assertEqual(toString(input_data), expected_output)

    # def test_invalid_integer_representation(self):
    #     with self.assertRaises(ValueError):
    #         toString(np.array([1]))
    #     with self.assertRaises(ValueError):
    #         toString(np.array([1, "4"]))
    #     #with self.assertRaises(ValueError):
    #     #    toString(np.array([1, -1]))

class TestRightPad(unittest.TestCase):
    def test_no_padding(self):
        test_inputs = ['X', 'Y', 'Z', 'I', 'XX', 'IZ', 'YX', 'XYZI']
        pad_length = 4
        expected_outputs = ['XIII', 'YIII', 'ZIII', 'IIII', 'XXII', 'IZII', 'YXII', 'XYZI']
        ZX_form = toZX(test_inputs)
        ZX_ouptuts = right_pad(ZX_form, pad_length)
        expected_outputs = toZX(expected_outputs)
        self.assertEqual(ZX_ouptuts[0], expected_outputs[0])
        for i in range(1, len(ZX_ouptuts)):
            self.assertEqual(ZX_ouptuts[i], expected_outputs[i], f"Input: {test_inputs[i-1]}, Output: {ZX_ouptuts[i]}, Expected: {expected_outputs[i]}")

    def test_basic_right_pad(self):
        test_inputs = ['X', 'Y', 'Z', 'I', 'XX', 'IZ', 'YX']
        pad_length = 4
        expected_outputs = ['XIII', 'YIII', 'ZIII', 'IIII', 'XXII', 'IZII', 'YXII']
        ZX_form = toZX(test_inputs)
        ZX_ouptuts = right_pad(ZX_form, pad_length)
        expected_outputs = toZX(expected_outputs)
        self.assertEqual(ZX_ouptuts[0], expected_outputs[0])
        for i in range(1, len(ZX_ouptuts)):
            self.assertEqual(ZX_ouptuts[i], expected_outputs[i], f"Input: {test_inputs[i-1]}, Output: {ZX_ouptuts[i]}, Expected: {expected_outputs[i]}")

class TestLeftPad(unittest.TestCase):
    def test_left_pad(self):
        sym_form = toZX(['XX'])
        result_size = 4
        expected_output = toZX(['XXII'])
        result = left_pad(sym_form, result_size)
        self.assertEqual(result[0], expected_output[0])
        np.testing.assert_array_equal(result[1:], expected_output[1:])

class TestConcatenateZX(unittest.TestCase):
    #def test_empty_list(self):
    #    with self.assertRaises(NumbaValueError):
    #        concatenate_ZX([])
    
    def test_single_sym_form(self):
        sym_form = np.array([3, 5, 10], dtype=GLOBAL_INTEGER)
        result = concatenate_ZX([sym_form])
        np.testing.assert_array_equal(result, sym_form)
    
    def test_multiple_sym_forms_different_lengths(self):
        String_forms = [
            ['X','Y'],
            ['XX','ZI'],
            ['XZY', 'Z']
        ]
        output12 = ['XI','YI','XX','ZI']
        output13 = ['XII', 'YII', 'XZY', 'ZII']
        cat12 = concatenate_ZX([toZX(String_forms[0]), toZX(String_forms[1])])
        cat13 = concatenate_ZX([toZX(String_forms[0]), toZX(String_forms[2])])
        ZXoutput12 = toZX(output12)
        ZXoutput13 = toZX(output13)
        self.assertEqual(cat12[0], ZXoutput12[0])
        self.assertEqual(cat13[0], ZXoutput13[0])
        np.testing.assert_array_equal(cat12[1:], ZXoutput12[1:])
        np.testing.assert_array_equal(cat13[1:], ZXoutput13[1:])
        
    # def test_multiple_sym_forms_same_length(self):
    #     sym_forms = [
    #         (3, np.array([5, 10])),
    #         (3, np.array([7, 14]))
    #     ]
    #     expected_length = 3
    #     expected_ints = np.array([5, 10, 7, 14])
    #     result = concatenate_ZX(sym_forms)
    #     self.assertEqual(result[0], expected_length)
    #     np.testing.assert_array_equal(result[1], expected_ints)

class TestSymplecticInnerProduct(unittest.TestCase):
    def test_commuting(self):
        p1 = toZX(['XX'])
        p2 = toZX(['YY'])
        self.assertEqual(symplectic_inner_product(p1, p2), 0)
        
    def test_non_commuting(self):
        p1 = toZX(['XX'])
        p2 = toZX(['ZI'])
        self.assertEqual(symplectic_inner_product(p1, p2), 1)

    def test_different_lengths(self):
        p1 = toZX('X')
        p2 = toZX('ZZ')
        # anti-commute.
        self.assertEqual(symplectic_inner_product(p1, p2), 1)
        p3 = toZX('Y')
        # Y = YI, ZZ. Y_1 Z_2 + Z_1 X_2 = 1*0 + 1*1 = 1. Anti-commute.
        self.assertEqual(symplectic_inner_product(p3, p2), 1)

class TestSymplecticInnerProductInt(unittest.TestCase):
    def test_commuting(self):
        p1 = toZX('XX')
        p2 = toZX('YY')
        self.assertEqual(symplectic_inner_product_int(p1[1], p2[1], p1[0]), 0)

    def test_non_commuting(self):
        p1 = toZX('XX')
        p2 = toZX('ZI')
        self.assertEqual(symplectic_inner_product_int(p1[1], p2[1], p1[0]), 1)

class TestCommutes(unittest.TestCase):
    def test_commuting(self):
        p1 = toZX(['XX'])
        p2 = toZX(['YY'])
        self.assertTrue(commutes(p1, p2))

    def test_non_commuting(self):
        p1 = toZX(['XX'])
        p2 = toZX(['ZI'])
        self.assertFalse(commutes(p1, p2))

    def test_commuting_multiple(self):
        p1 = toZX(['XX', 'YY'])
        p2 = toZX(['YY', 'XX'])
        self.assertTrue(commutes(p1, p2))

    def test_non_commuting_multiple(self):
        p1 = toZX(['XX', 'YY'])
        p2 = toZX(['ZI', 'XX'])
        self.assertFalse(commutes(p1, p2))

class TestCommutationArrays(unittest.TestCase):
    def test_commute_array_correctness(self):
        paulis = ['XXI', 'YIZ', 'ZZZ', 'IYX']
        sym_form = toZX(paulis)
        
        # Manually create list of tuples for commute_array_fast
        sym_form_list = []
        length = sym_form[0]
        for i in range(1, len(sym_form)):
            sym_form_list.append((length, np.array([sym_form[i]])))

        comm_matrix = bsip_array(sym_form)
        #comm_matrix_fast = commute_array_fast(sym_form_list)

        ##np.testing.assert_array_equal(comm_matrix, comm_matrix_fast)

        expected = np.array([
            [0, 1, 0, 1],
            [1, 0, 1, 1],
            [0, 1, 0, 0],
            [1, 1, 0, 0]
        ])
        np.testing.assert_array_equal(comm_matrix, expected)

    def test_commute_array_performance(self):
        num_paulis = 100
        num_qubits = 10
        
        # Generate random paulis
        paulis = []
        for _ in range(num_paulis):
            p_str = "".join(np.random.choice(['I', 'X', 'Y', 'Z'], size=num_qubits))
            paulis.append(p_str)
            
        sym_form = toZX(paulis)
        
        sym_form_list = []
        length = sym_form[0]
        for i in range(1, len(sym_form)):
            sym_form_list.append((length, np.array([sym_form[i]])))

        start_time = time.time()
        comm_matrix = bsip_array(sym_form)
        time_slow = time.time() - start_time

        start_time = time.time()
        #comm_matrix_fast = commute_array_fast(sym_form_list)
        #time_fast = time.time() - start_time
        
        #np.testing.assert_array_equal(comm_matrix, comm_matrix_fast)
        
        print(f"\nPerformance test ({num_paulis} paulis, {num_qubits} qubits):")
        print(f"commute_array: {time_slow:.6f}s")
        #print(f"commute_array_fast: {time_fast:.6f}s")
        #self.assertLess(time_fast, time_slow)

if __name__ == "__main__":
    unittest.main()


