import unittest
from ptgalois.core import split_binary_number, popcount, inner_product, commutator
import numpy as np

def multiKron(P):
    """ Applies the Kronecker Product to a list of Pauli Matrices

    Parameters
    ----------
    P : list
        A list of Pauli Matrices

    Returns
    -------
    K : numpy array
        The Kronecker Product of the Pauli Matrices
    """
    K = P[0]
    for i in range(1, len(P)):
        K = np.kron(K, P[i])
    return K

class TestPauliTools(unittest.TestCase):

    def test_split_binary_number(self):
        num = 15
        n = 2
        expected_output = (3, 3)
        self.assertEqual(split_binary_number(num, n), expected_output)

    def test_popcount(self):
        v = 15
        expected_output = 4
        self.assertEqual(popcount(v), expected_output)

    def test_inner_product(self):
        a = np.array([0, 1, 1, 0])
        b = np.array([1, 0, 1, 1])
        expected_output = 1
        self.assertEqual(inner_product(a, b), expected_output)

    def test_commutator(self):
        p1 = np.array([0, 1, 1, 0])
        p2 = np.array([1, 0, 1, 1])
        expected_output = p1 + p2
        self.assertTrue(np.array_equal(commutator(p1, p2), expected_output))

    def test_multiKron(self):
        p_list = [np.array([[1, 0], [0, 1]]), np.array([[0, 1], [1, 0]])]
        expected_output = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
        self.assertTrue(np.array_equal(multiKron(p_list), expected_output))

if __name__ == '__main__':
    unittest.main()
