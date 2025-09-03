import numpy as np
import numba
from numba import njit
from ptgalois.util import popcount


# Pauli Matricies
I = np.array([[1, 0], [0, 1]])
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
#Conversion Dictionaries
toMatrix = {0: I, 1: X, 2: Y, 3: Z, 'I': I, 'X': X, 'Y': Y, 'Z': Z}
toPauli = {0: 'I', 1: 'X', 2: 'Y', 3: 'Z', 'I': 0, 'X': 1, 'Y': 2, 'Z': 3}
#Basis States
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

@njit#('boolean(uint32,uint32,uint32)')
def sbf_old(a,b,n):
    """
    Compute the Symplectic inner product of the two binary numbers to determine if the two Pauli Matricies commute
    WARNING: WORKS FOR ONLY 32 QUBITS
    Args:
        a: Pauli Matrix in the form of an integer
        b: Pauli Matrix in the form of an integer
        n: Number of qubits
    Returns:
        Boolean: False if the two Pauli Matricies commute, True otherwise
    """
    assert n < 33, 'Only 32 qubits supported'
    #Split:
    upperA, lowerA = split_binary_number(a, n)
    upperB, lowerB = split_binary_number(b, n)
    #Compute:
    left = popcount(upperA & lowerB)
    right = popcount(upperB & lowerA)
    return (left - right) % 2

def inner_product(a,b):
    """Computes the symplectic inner product of the two inputs. Outputs 0 if they commute, 1 if they anticommute.
    
    Args:
        a: Pauli Matrix in the form of an integer
        b: Pauli Matrix in the form of an integer
    Returns:
        Boolean: False if the two Pauli Matricies commute, True otherwise
    """
    #If signed, strip the sign off. Check for both a and b:
    if a.shape[-1] % 2: #If the number of columns is odd, then the first column is the sign
            #c = a.copy()
            a = a[:,:-1]
    if b.shape[-1] % 2: #-1 indexing fixes it so that we don't need an array of dimension larger than 1 -- now works for a single index
        b = b[:,:-1]
    Warning('Verify that a, b are not modified in place')
    assert a.shape[-1] == b.shape[-1], 'Pauli Matricies must be the same size'
    #az, ax = np.split(a, 2,axis=len(a.shape)-1)
    #print(b)
    bz, bx = np.split(b, 2,axis=len(b.shape)-1)
    b = np.concatenate((bx, bz), axis=len(b.shape)-1)
    #print(b)
    return a @ b.T
    
    

#@njit
def commutator(p1, p2, signed = False):
    """
    Computes the commutator of two Pauli Matricies, returns None if they commute
    Args:
        p1: GF2 representation of the pauli Matrix
        p2: GF2 representation of the right pauli matrix
        TODO signed: If true, returns the signed Integer, otherwise returns the absolute value
    Returns:
        GF2 array: Commutator of the two Pauli Matricies, or 0 if they commute
    """
    if not inner_product(p1, p2):
        return 0
    else:
        return p1 + p2
    
def commutes(PauliInt1, PauliInt2):
    """
    Returns True if the two Paulis Matricies Commute, False otherwise
        PauliInt1: Pauli Matrix in the form of an integer
        PauliInt2: Pauli Matrix in the form of an integer
        n: Number of qubits
    Returns:
        Boolean: True if the two Pauli Matricies commute, False otherwise
    """
    return not inner_product(PauliInt1, PauliInt2)