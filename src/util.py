import numpy as np
import unittest
from numba import jit, types, njit, prange
from numba.core.errors import NumbaTypeError, NumbaValueError
from operator import ixor
from numpy import int64

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from core import GLOBAL_INTEGER, symplectic_inner_product, toZX, commutes
import numpy as np
try:
    from joblib import Parallel, delayed
except ImportError:
    # Fallback for parallel execution if joblib is not available
    class Dummy:
        def __call__(self, *args, **kwargs):
            return list(map(args[0], args[1]))
    
    def delayed(func):
        return func
    
    Parallel = Dummy

@njit
def AtoBinary(pauli):
    k = pauli[0]
    length = 2*k+1
    #Convert int to binary, padd to length, remove the 0b prefix
    #binforms = #array of strings of length 2k+1
    binforms = np.empty((len(pauli)-1), dtype=object)
    for i in range(len(pauli)-1):
        binforms[i] = bin(pauli[i+1])[2:].zfill(length)[:-1][::-1]
    output_matrix = np.zeros((2*k, len(pauli)-1))
    for i in range(len(binforms)):
        for j in range(2*k):
            #Only update if it's a 1:
            if binforms[i][j] == '1':
                output_matrix[j,i] = int(binforms[i][j])
    return output_matrix

@njit #Tested ISH
def toBinary(pauli):
    """
    Converts an array of Pauli integers to a binary matrix representation.

    Args:
        pauli (np.ndarray): Input array where pauli[0] = k (number of qubits),
                            and pauli[1:] are integers representing Pauli operators.

    Returns:
        np.ndarray: A binary matrix of shape (2*k, len(pauli)-1).
    """
    k = pauli[0]
    length = 2 * k + 1  # Total bits including the sign bit
    num_paulis = len(pauli) - 1  # Number of Pauli operators

    # Initialize the output matrix
    output_matrix = np.zeros((2 * k, num_paulis), dtype=np.int8)

    for i in range(num_paulis):
        # Get the integer representation of the Pauli operator and remove the sign bit
        integer = pauli[i + 1] >> 1  # Shift right to remove the sign bit

        for j in range(2 * k):
            # Extract bit j and assign to the output matrix
            output_matrix[j, i] = (integer >> j) & 1

    return output_matrix.T

@njit
def convert_array_type(arr, dtype):
    new_arr = np.empty(arr.shape, dtype=dtype)
    new_arr[:] = arr
    return new_arr
from numba import njit

# A helper function to count set bits (popcount), which is very fast in Numba.
@njit
def popcount(n):
    """Counts the number of set bits in an integer (Hamming weight)."""
    count = 0
    while n > 0:
        # This efficiently removes the rightmost set bit
        n &= (n - 1)
        count += 1
    return count

@njit
def getParity(pauli, basis='Y'):
    """
    Calculates the parity of a specific Pauli operator ('X', 'Y', or 'Z')
    in a Pauli string using efficient bitwise operations.
    
    Args:
        pauli (np.ndarray): Pauli operator in ZX format, where pauli[0] is the
                            number of qubits (k) and pauli[1] is the integer
                            representation.
        basis (str): The basis to count ('X', 'Y', or 'Z').
    
    Returns:
        int: 0 if the count of the basis operators is even, 1 if it is odd.
    """
    k = pauli[0]
    int_rep = pauli[1]

    # Isolate the k-bit representations for the Z and X parts of the operator
    # Z part is in bits 1 to k
    z_bits = (int_rep >> 1) & ((1 << k) - 1)
    # X part is in bits k+1 to 2k
    x_bits = (int_rep >> (k + 1)) & ((1 << k) - 1)

    count = 0
    if basis == 'Y':
        # A 'Y' operator exists where both X and Z bits are 1.
        # Count the number of positions where both are set.
        count = popcount(x_bits & z_bits)
    elif basis == 'X':
        # An 'X' operator exists where the X bit is 1 and the Z bit is 0.
        count = popcount(x_bits & (~z_bits))
    elif basis == 'Z':
        # A 'Z' operator exists where the Z bit is 1 and the X bit is 0.
        count = popcount(z_bits & (~x_bits))
    
    return count % 2

def get_pauli_obs(pauli_input, probs, parallel=False):
    """
    Calculate expectation values of Pauli operators.
    
    Parameters:
        pauli_input: a Pauli string or a list of Pauli strings
        probs: a dictionary of probabilities where keys are bit-strings and values are probabilities
        parallel (bool): Whether to use parallel execution
    
    Returns:
        np.ndarray: The expectation values of the Pauli operator(s)
    """
    pauli = pauli_input.copy()
    pauliZX = toZX(pauli)
    expectations = np.zeros(len(pauliZX) - 1)
    
    def get_val(pauli, shot):
        yParity_P = getParity(pauli, basis='Y')
        comm_val = 1 - commutes(pauli, shot)  # 1 if they anti-commute, 0 if they commute
        return np.power(-1, (yParity_P + comm_val) % 2)
    
    for key in probs.keys():
        keyZX = toZX(key)
        if parallel:
            results = Parallel(n_jobs=-1)(delayed(get_val)(pauliZX[i+1:i+2], keyZX) for i in range(len(pauliZX)-1))
            expectations += np.array(results) * probs[key]
        else:
            for i in range(len(pauliZX) - 1):
                val = get_val(pauliZX[i+1:i+2], keyZX)
                expectations[i] += val * probs[key]
    
    return expectations

def get_pauli_pauli_obs(pauli_input, probs, parallel=False):
    """
    Calculate expectation values of Pauli operators, including Y parity of the shot.
    
    Parameters:
        pauli_input: a Pauli string or a list of Pauli strings
        probs: a dictionary of probabilities where keys are bit-strings and values are probabilities
        parallel (bool): Whether to use parallel execution
    
    Returns:
        np.ndarray: The expectation values of the Pauli operator(s)
    """
    pauli = pauli_input.copy()
    pauliZX = toZX(pauli)
    expectations = np.zeros(len(pauliZX) - 1)
    
    def get_val(pauli, shot):
        yParity = getParity(pauli, basis='Y')
        comm_val = 1 - commutes(pauli, shot)  # 1 if they anti-commute, 0 if they commute
        yParity_shot = getParity(shot, basis='Y')  # Y parity of the shot
        return np.power(-1, (yParity + comm_val + yParity_shot) % 2)
    
    for key in probs.keys():
        keyZX = toZX(key)
        if parallel:
            results = Parallel(n_jobs=-1)(delayed(get_val)(pauliZX[i+1:i+2], keyZX) for i in range(len(pauliZX)-1))
            expectations += np.array(results) * probs[key]
        else:
            for i in range(len(pauliZX) - 1):
                val = get_val(pauliZX[i+1:i+2], keyZX)
                expectations[i] += val * probs[key]
    
    return expectations

def getCentralizer(counts, return_generators=False):
    """
    Given Paulis in the ZX form, generates part of the group by adding samples 
    and then computes the center.

    Args:
        counts: a dictionary of probabilities and bit-strings (No Qiskit form)
        return_generators: if true, returns the generators of the group
    
    Returns:
        The centralizer of the group, and the generators if return_generators is True
    """
    try:
        import galois
    except ImportError:
        raise ImportError("The galois package is required for getCentralizer")
    
    try:
        from group import centralizer, row_reduce as generator
    except ImportError:
        raise ImportError("The group module with centralizer and row_reduce functions is required")
    
    # Checks how many unique outputs there are so we can iterate over all of them
    unique_shots_count = len(counts.keys())
    qubits = len(list(counts.keys())[0]) // 2
    group = galois.GF(2).Zeros((unique_shots_count**2, 2*qubits))

    paulis = [key for key in counts.keys()]
    pauliZX = toZX(paulis)
    
    for i in range(unique_shots_count):
        for j in range(unique_shots_count):
            if i > j:
                # Add the binary representations of Paulis i and j
                group[unique_shots_count*i+j] = toBinary(np.array([pauliZX[0], pauliZX[i+1] ^ pauliZX[j+1]]))
    
    # Row reduces the group to get the generators
    generators = generator(group)

    # Computes the center of the generators
    center_symplectic = centralizer(generators, reduced=True)
    
    if return_generators:
        return center_symplectic, generators
    else:
        return center_symplectic

def Pauli_expectation(shots, pauli):
    """
    Shots should be a (N x 2) array, where N is the number of unique shots
    
    Returns the expectation value of the Pauli string given the samples:
    
    Args:
        shots (np.ndarray): An array of shots. shots[i,1] is the probability of Pauli shots[i,0]
        pauli (np.ndarray): A Pauli string in bsf format (array of ints)
    Returns:
        float: The expectation value of the Pauli string
    """
    k = pauli[0]
    Y_check = toZX(['Y'*k])
    
    def get_sign(shot, pauli):
        yParity = symplectic_inner_product(pauli, Y_check)
        comm_value = symplectic_inner_product(shot, pauli)
        return np.power(-1, ((yParity + comm_value) % 2))
    
    expectation = 0.0
    for i in range(len(shots)):
        shot_value = shots[i, 0]
        prob = shots[i, 1]
        sign = get_sign(np.array([k, shot_value]), pauli)
        expectation += sign * prob
    
    return expectation