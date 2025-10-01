import numpy as np
import unittest
from numba import jit, types, njit, prange#, float16
from numba.types import float16, float64, int8
from numba.core.errors import NumbaTypeError, NumbaValueError
from operator import ixor
from numpy import int64

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from core import GLOBAL_INTEGER, symplectic_inner_product, symplectic_inner_product_int
from util import convert_array_type, toBinary, popcount
from galois import GF2

import numpy as np

from numba import njit

@njit(cache=True)
def row_reduce(input_pauli):
    """
    Perform row reduction on integers using bitwise operations,
    and remove zero rows from the result.
    Parameters:
        input_pauli (np.ndarray): The first element is k (an integer),
                                  the rest are integers representing rows.
    Returns:
        np.ndarray: The reduced integers, with k at the zero index,
                   and zero rows removed.
    """
    data = input_pauli.copy()
    k = data[0]
    num_bits = 2 * k + 1  # Total bits including the sign bit
    num_bits_wo_sign = num_bits - 1
    int_rows = data[1:].copy()
    n_rows = len(int_rows)
    n_cols = num_bits_wo_sign
    # Remove the sign bit (rightmost bit) by shifting right
    for i in range(n_rows):
        int_rows[i] = int_rows[i] >> 1
    pivot_row = 0
    for col in range(n_cols - 1, -1, -1):
        found_pivot = False
        for row in range(pivot_row, n_rows):
            if (int_rows[row] >> col) & 1:
                if row != pivot_row:
                    tmp = int_rows[pivot_row]
                    int_rows[pivot_row] = int_rows[row]
                    int_rows[row] = tmp
                found_pivot = True
                break
        if not found_pivot:
            continue
        for row in range(pivot_row + 1, n_rows):
            if (int_rows[row] >> col) & 1:
                int_rows[row] ^= int_rows[pivot_row]
        pivot_row += 1
        if pivot_row >= n_rows:
            break
    # Backward substitution
    for i in range(pivot_row - 1, -1, -1):
        row_val = int_rows[i]
        pivot_col = -1
        for col in range(n_cols - 1, -1, -1):
            if (row_val >> col) & 1:
                pivot_col = col
                break
        if pivot_col == -1:
            continue
        for row in range(i):
            if (int_rows[row] >> pivot_col) & 1:
                int_rows[row] ^= int_rows[i]
    # Remove zero rows and shift left to restore the sign bit position
    reduced_int_rows = []
    for row in int_rows:
        if row != 0:
            reduced_row = row << 1  # Restore sign bit position (set to zero)
            reduced_int_rows.append(reduced_row)
    result = np.zeros(len(reduced_int_rows) + 1, dtype=GLOBAL_INTEGER)
    result[0] = k
    for i in range(len(reduced_int_rows)):
        result[i+1] = reduced_int_rows[i]
    return result

@njit(cache=True)
def generators(input_pauli):
    """Just a wrapper around row_reduce
    """
    return row_reduce(input_pauli)

# NO NUMBA, BAD

# Numba-compatible null space over GF(2) using row_reduce as a subroutine
@njit(cache=True)
def null_space(A):
    """
    Compute the null space of a binary matrix A (mod 2) using row-reduction.
    Returns a matrix whose rows form a basis for the null space.
    """
    m, n = A.shape
    A = A.copy()
    pivots = np.full(m, -1, dtype=np.int32)
    row = 0
    for col in range(n):
        sel = -1
        for r in range(row, m):
            if A[r, col]:
                sel = r
                break
        if sel == -1:
            continue
        if sel != row:
            tmp = A[row].copy()
            A[row] = A[sel]
            A[sel] = tmp
        pivots[row] = col
        for r in range(row + 1, m):
            if A[r, col]:
                A[r] ^= A[row]
        row += 1
        if row == m:
            break
    # Backward elimination
    for i in range(row-1, -1, -1):
        col = pivots[i]
        if col == -1:
            continue
        for r in range(i):
            if A[r, col]:
                A[r] ^= A[i]
    # Identify free variables (columns not used as pivots)
    used = np.zeros(n, dtype=np.bool_)
    for i in range(row):
        if pivots[i] != -1:
            used[pivots[i]] = True
    nullity = 0
    for j in range(n):
        if not used[j]:
            nullity += 1
    if nullity == 0:
        return np.zeros((0, n), dtype=np.uint8)
    N = np.zeros((nullity, n), dtype=np.uint8)
    idx = 0
    for fv in range(n):
        if not used[fv]:
            N[idx, fv] = 1
            for i in range(row):
                col = pivots[i]
                if col == -1:
                    continue
                N[idx, col] = A[i, fv]
            idx += 1
    return N.astype(np.uint8)



@njit(cache=True)
def inner_product(paulis):
    """
    Computes the symplectic inner product matrix of the given paulis.
    
    Given an input [k, sym_1, sym_2, ..., sym_n], the symplectic inner product matrix is computed as
    [[sbf(sym_1, sym_1), sbf(sym_1, sym_2), ..., sbf(sym_1, sym_n)],
     [sbf(sym_2, sym_1), sbf(sym_2, sym_2), ..., sbf(sym_2, sym_n)],
     ...
     [sbf(sym_n, sym_1), sbf(sym_n, sym_2), ..., sbf(sym_n, sym_n)]]
    Conver the resulting binary array to an integer representation, add k at the zero index. here, k is n, the number of sym inputs which should be less than 2n
    Additionally, append a sign bit =0 at the zero index. 
    
    
    Args:
        paulis (list): List containing k at index 0 and the symplectic forms.
        
    Returns:
        numpy.ndarray: Symplectic inner product matrix.
    """
    k = paulis[0]
    sym_forms = paulis[1:]
    n = len(sym_forms)
    ip_matrix = np.zeros((n, n), dtype=np.int8)
    for i in prange(n):
        for j in prange(n):
            ip_matrix[i, j] = symplectic_inner_product_int(sym_forms[i], sym_forms[j], k)
    return ip_matrix


@njit(cache=True)
def radical(paulis, reduced=False):
    """
    Computes the generators of the radical of the group generated by the paulis.
    
    Args:
        paulis (list): List containing k at index 0 and the symplectic forms.
        reduced (bool): If False, reduce the data using the row_reduce function.
        
    Returns:
        GF2 array: Integer array
    """
    if not reduced:
        reduced_pauli = row_reduce(paulis.copy())
        
    else:
        reduced_pauli = paulis.copy()
    return null_space(inner_product(reduced_pauli))


#@njit(cache=True)
def differences(paulis, paulis2 = None):
    """
    Computes the bell difference between samples. Returns paulis[i] AND paulis [i+1] cyclically

    Args:
        paulis (ndarray): ZX form Pauli strings
        paulis2 (ndarray, optional): If provided, computes differences between paulis and paulis2 instead of cyclic differences within paulis.
    Returns:
        ndarray: Array of differences
    """
    if paulis2 is not None:
        assert paulis.shape == paulis2.shape, "paulis and paulis2 must have the same shape"
        assert paulis[0] == paulis2[0], "paulis and paulis2 must have the same k value"    
    n = paulis.shape[0]
    k = paulis[0]
    if paulis2 is not None:
        diffs = np.zeros((n), dtype=GLOBAL_INTEGER)
        for i in range(1,n):
            diffs[i] = paulis[i] ^ paulis2[i]
        diffs[0] = k
        return diffs
    else:
        diffs = np.zeros((n), dtype=GLOBAL_INTEGER)
        for i in range(1,n):
            if i == n-1:
                diffs[i] = paulis[i] ^ paulis[1]
            else:
                # This could be speed up by leaving out the %n and just ending up with n-1 instead of n terms
                diffs[i] = paulis[i] ^ paulis[(i + 1)]
        diffs[0] = k
        return diffs


def matmul_mod2(A, B_cols):
    A_uint = np.asarray(A, dtype=np.uint8)
    B_arr = np.asarray(B_cols, dtype=np.uint8)

    if B_arr.ndim != 2:
        raise ValueError("B_cols must be a 2D array for GF(2) multiplication.")

    m, blocks = A_uint.shape

    if B_arr.shape[0] == blocks:
        B_use = B_arr
    elif B_arr.shape[1] == blocks:
        B_use = B_arr.T
    else:
        raise ValueError("Shape mismatch between A columns and B_cols.")

    n = B_use.shape[1]
    out = np.zeros((m, n), dtype=np.uint8)

    for i in range(m):
        for j in range(n):
            acc = 0
            for t in range(blocks):
                acc ^= (A_uint[i, t] & B_use[t, j])
            out[i, j] = acc

    return out

def centralizer(pauli_input, reduced=False):
    """Returns the centralizer of the input Pauli group. First computes the radical, then takes the kernel of the reduced Pauli input basis ker(P)@P.
    
    
    Args:
        pauli_input (ndarray): List containing k at index 0 and the symplectic forms.
        reduced (bool): If False, reduce the data using the row_reduce function.
    """    
    #if not reduced:
    #    reduced_pauli = row_reduce(pauli_input.copy())
    #else:
    #    reduced_pauli = pauli_input.copy()
    kernel = radical(pauli_input.copy(), reduced=reduced)
    #kernel = convert_array_type(kernel, int8)
    #reduced_pauli = toBinary(reduced_pauli)
    #old: return kernel @ row_space(pauli_input)
    return matmul_mod2(kernel, row_space(pauli_input))



def group(pauli_input):
    """ Returns the entire group generated by the input. May be exponentially large. (Probably will be!)
    Args:
        pauli_input (ndarray): List containing k at index 0 and the symplectic forms.
    Returns:
        ndarray: All group elements
    """
    #recursively check the ^ between all elements, then append and remove duplicates
    n = pauli_input.shape[0]
    k = pauli_input[0]
    paulis = pauli_input[1:].copy().tolist()
    current_len = len(paulis)
    added = True
    while added:
        added = False
        new_elements = []
        for i in range(current_len):
            for j in range(i, current_len):
                new_elem = paulis[i] ^ paulis[j]
                if new_elem not in paulis and new_elem not in new_elements:
                    new_elements.append(new_elem)
                    added = True
        paulis.extend(new_elements)
        current_len = len(paulis)
    result = np.zeros(len(paulis)+1, dtype=GLOBAL_INTEGER)
    result[0] = k
    for i in range(len(paulis)):
        result[i+1] = paulis[i]
    return result



#TODO: Check inGroup function

@njit(cache=True)
def ingroup(candidates, pauli_set, reduced=False):
    """
    Determine whether candidate Pauli operators are linearly independent from a reference set.

    Both ``candidates`` and ``pauli_set`` use the packed ZX integer layout where index 0 stores
    ``k`` (the number of qubits) and subsequent entries store the packed operators.

    Args:
        candidates (np.ndarray): Packed Pauli integers with shape (m + 1,). Index 0 stores ``k``
            and the remaining entries are the operators to test.
        pauli_set (np.ndarray): Packed Pauli integers defining the reference span. The first entry
            must be the same ``k`` value as ``candidates``.
        reduced (bool): If ``True``, ``pauli_set`` is assumed to already be row-reduced. When
            ``False`` the function will compute the reduced basis.

    Returns:
        np.ndarray: Boolean array of length ``len(candidates) - 1`` where ``True`` indicates that
            the corresponding candidate is linearly independent from ``pauli_set``.
    """

    if candidates.shape[0] <= 1:
        return np.zeros(0, dtype=np.bool_)

    if candidates[0] != pauli_set[0]:
        raise ValueError("candidates and pauli_set must have the same k value")

    if reduced:
        basis = pauli_set.copy()
    else:
        basis = row_reduce(pauli_set.copy())

    k = np.int32(basis[0])
    rank = basis.shape[0] - 1
    span_cols = 2 * k

    basis_rows = np.zeros(rank, dtype=GLOBAL_INTEGER)
    pivot_cols = np.full(rank, -1, dtype=np.int32)

    for r in range(rank):
        row_val = basis[r + 1] >> 1
        basis_rows[r] = row_val
        pivot = -1
        for col in range(span_cols - 1, -1, -1):
            if (row_val >> col) & 1:
                pivot = col
                break
        pivot_cols[r] = pivot

    count = candidates.shape[0] - 1
    independent = np.zeros(count, dtype=np.bool_)

    for idx in range(count):
        vec = candidates[idx + 1] >> 1
        remainder = vec
        for r in range(rank):
            pivot = pivot_cols[r]
            if pivot == -1:
                continue
            if (remainder >> pivot) & 1:
                remainder ^= basis_rows[r]
        independent[idx] = remainder != 0

    return independent

@njit(cache=True)
def row_space(pauli_input):
    """
    Computes the row space of a matrix over GF(2).
    First reduces the matrix using row_reduce, then returns the
    binary matrix representation of the reduced form.
    
    Args:
        pauli_input (np.ndarray): List containing k at index 0 and the symplectic forms.
    
    Returns:
        np.ndarray: Binary matrix representation of the row space,
                   with dimensions (num_rows, 2*k) where k is the number of qubits.
    """
    # First reduce the input to get linearly independent generators
    reduced_pauli = row_reduce(pauli_input.copy())
    
    k = reduced_pauli[0]  # Number of qubits
    num_rows = len(reduced_pauli) - 1  # Number of Pauli strings (excluding k)
    
    # Create binary matrix with dimensions (num_rows, 2*k)
    # Each row represents a Pauli string in binary form (without sign bit)
    binary_matrix = np.zeros((num_rows, 2*k), dtype=np.int8)
    
    for i in prange(num_rows):
        pauli_int = reduced_pauli[i+1]  # Skip k at index 0
        # Remove the sign bit by shifting right
        pauli_int = pauli_int >> 1
        
        # Extract the binary representation (Z|X format)
        for j in prange(k):
            # Extract Z bits (positions 0 to k-1)
            binary_matrix[i, j] = (pauli_int >> j) & 1
            
            # Extract X bits (positions k to 2k-1)
            binary_matrix[i, j+k] = (pauli_int >> (j+k)) & 1
    
    return binary_matrix