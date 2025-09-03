import unittest
import numpy as np
from galois import GF
from ptgalois.converter import toZX, toString, ZXtoMatrix, multiKron

class TestPauliTools(unittest.TestCase):

    def test_toZX_format1(self):
        # Test format 1: List of Pauli Integers
        p_list = [0, 1, 2, 3]
        expected_output = GF(2)([[0, 0, 1, 1], [0, 1, 1, 0]])
        self.assertTrue(np.array_equal(toZX(p_list), expected_output))

    def test_toZX_format2(self):
        # Test format 2: List of Pauli Matrix Letters
        p_list = ['I', 'X', 'Y', 'Z']
        expected_output = GF(2)([[0, 0, 1, 1], [0, 1, 1, 0]])
        self.assertTrue(np.array_equal(toZX(p_list), expected_output))

    def test_toString(self):
        # Test toString function
        paulis = np.array([[0, 1, 1, 0, 0, 1, 1, 0]])
        expected_output = ['IXYZ']
        self.assertEqual(toString(paulis), expected_output)

    def test_ZXtoMatrix(self):
        # Test ZXtoMatrix function
        p_list = np.array([[0, 1, 1, 0, 0, 1, 1, 0]])
        expected_output = np.array([[0, 1], [1, 0]])
        self.assertTrue(np.array_equal(ZXtoMatrix(p_list), expected_output))

    def test_multiKron(self):
        # Test multiKron function
        p_list = [np.array([[1, 0], [0, 1]]), np.array([[0, 1], [1, 0]])]
        expected_output = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
        self.assertTrue(np.array_equal(multiKron(p_list), expected_output))

if __name__ == '__main__':
    unittest.main()
