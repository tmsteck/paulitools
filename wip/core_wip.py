import unittest
import numpy as np
from numba import njit

# Pauli Matrices
I = np.array([[1, 0], [0, 1]])
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])

# Conversion Dictionaries
toMatrix = {0: I, 1: X, 2: Y, 3: Z, 'I': I, 'X': X, 'Y': Y, 'Z': Z}
toPauli = {0: 'I', 1: 'X', 2: 'Y', 3: 'Z', 'I': 0, 'X': 1, 'Y': 2, 'Z': 3}

# Basis States
Zero = np.array([[1], [0]])
One = np.array([[0], [1]])
Plus = 1/np.sqrt(2) * np.array([[1], [1]])
Minus = 1/np.sqrt(2) * np.array([[1], [-1]])
PlusY = 1/np.sqrt(2) * np.array([[1], [1j]])
MinusY = 1/np.sqrt(2) * np.array([[1], [-1j]])
toState = {0: Zero, 1: One, '+': Plus, '-': Minus, '0': Zero, '1': One, '+i': PlusY, '-i': MinusY}

@njit
def split_binary_number(num, n):
    """
    Splits a binary number into two parts
    Args:
        num: Integer to split
        n: Number of bits to split into
    Returns:
        Tuple: (upper_bits, lower_bits)
    """
    mask = (1 << n) - 1
    lower_bits = num & mask
    upper_bits = num >> n
    return upper_bits, lower_bits

@njit
def popcount(v):
    """ FROM https://stackoverflow.com/questions/71097470/msb-lsb-popcount-in-numba """
    v = v - ((v >> 1) & 0x55555555)
    v = (v & 0x33333333) + ((v >> 2) & 0x33333333)
    c = np.uint32((v + (v >> 4) & 0xF0F0F0F) * 0x1010101) >> 24
    return c

@njit
def inner_product(a, b):
    """
    Computes the symplectic inner product of the two inputs. Outputs 0 if they commute, 1 if they anticommute.
    Args:
        a: Pauli Matrix in the form of an integer
        b: Pauli Matrix in the form of an integer
    Returns:
        Boolean: False if the two Pauli Matrices commute, True otherwise
    """
    assert len(a) == len(b), 'Pauli Matrices must be the same size'
    n = len(a) // 2
    inner_product_result = 0
    for i in range(n):
        inner_product_result += a[i] * b[n + i] + a[n + i] * b[i]
    return inner_product_result % 2

@njit
def commutator(p1, p2):
    """
    Computes the commutator of two Pauli Matrices, returns None if they commute
    Args:
        p1: GF2 representation of the pauli Matrix
        p2: GF2 representation of the right pauli matrix
    Returns:
        GF2 array: Commutator of the two Pauli Matrices, or 0 if they commute
    """
    if inner_product(p1, p2) == 0:
        return np.zeros_like(p1)
    else:
        return (p1 + p2) % 2

@njit
def multiKron(P):
    """ Applies the Kronecker Product to a list of Pauli Matrices

    Parameters
    ----------
    P : numpy.ndarray
        A numpy array of Pauli Matrices

    Returns
    -------
    K : numpy array
        The Kronecker Product of the Pauli Matrices
    """
    K = P[0]
    for i in range(1, len(P)):
        K = kron_product(K, P[i])
    return K

@njit
def kron_product(A, B):
    """
    Computes the Kronecker product of two matrices A and B.

    Args:
        A: First matrix
        B: Second matrix

    Returns:
        Kronecker product of A and B
    """
    A_rows, A_cols = A.shape
    B_rows, B_cols = B.shape
    result = np.zeros((A_rows * B_rows, A_cols * B_cols), dtype=A.dtype)
    for i in range(A_rows):
        for j in range(A_cols):
            result[i*B_rows:(i+1)*B_rows, j*B_cols:(j+1)*B_cols] = A[i, j] * B
    return result

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
        a = np.array([0, 1, 1, 0], dtype=np.int64)
        b = np.array([1, 0, 1, 1], dtype=np.int64)
        expected_output = 1
        self.assertEqual(inner_product(a, b), expected_output)

    def test_commutator(self):
        p1 = np.array([0, 1, 1, 0], dtype=np.int64)
        p2 = np.array([1, 0, 1, 1], dtype=np.int64)
        expected_output = (p1 + p2) % 2
        self.assertTrue(np.array_equal(commutator(p1, p2), expected_output))

    def test_multiKron(self):
        p_list = np.array([I, X], dtype=object)
        expected_output = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
        self.assertTrue(np.array_equal(multiKron(p_list), expected_output))

if __name__ == '__main__':
    unittest.main()
