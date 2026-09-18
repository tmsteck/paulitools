import numpy as np
from numba import njit

from ._numba import NUMBA_CACHE

from .core import GLOBAL_INTEGER, _validate_legacy_array_nb, symplectic_inner_product_int

@njit(cache=NUMBA_CACHE)
def row_reduce(input_pauli):
    """
    Return a basis of the binary Pauli span, discarding all phases.

    This is linear algebra modulo Pauli phases, not stabilizer reduction.
    Use ``stabilizer_reduce`` to preserve signs and detect inconsistent
    commuting generators.
    Parameters:
        input_pauli (np.ndarray): The first element is k (an integer),
                                  the rest are integers representing rows.
    Returns:
        np.ndarray: The reduced integers, with k at the zero index,
                   and zero rows removed.
    """
    if input_pauli.ndim != 1:
        raise ValueError("Packed ZX must be a one-dimensional integer array")
    _validate_legacy_array_nb(input_pauli)
    k = input_pauli[0]
    num_bits = 2 * k + 1  # Total bits including the sign bit
    num_bits_wo_sign = num_bits - 1
    int_rows = input_pauli[1:].copy()
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

@njit(cache=NUMBA_CACHE)
def generators(input_pauli):
    """Return binary-span generators modulo phases; alias for row_reduce."""
    return row_reduce(input_pauli)

@njit(cache=NUMBA_CACHE)
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



@njit(cache=NUMBA_CACHE)
def inner_product(paulis):
    """Return the binary commutation matrix for length-prefixed packed input.

    Entry (i,j) is 1 exactly when the corresponding operators anticommute.
    The symmetric int8 result has one row per operator and zero diagonal.
    Scalar phases are ignored.
    """
    if paulis.ndim != 1:
        raise ValueError("Packed ZX must be a one-dimensional integer array")
    _validate_legacy_array_nb(paulis)
    k = paulis[0]
    sym_forms = paulis[1:]
    n = len(sym_forms)
    ip_matrix = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(i + 1, n):
            value = symplectic_inner_product_int(sym_forms[i], sym_forms[j], k)
            ip_matrix[i, j] = value
            ip_matrix[j, i] = value
    return ip_matrix


@njit(cache=NUMBA_CACHE)
def radical(paulis, reduced=False):
    """
    Return coefficient rows spanning the kernel of the commutation matrix.

    The coefficients refer to ``row_reduce(paulis)[1:]`` by default, or
    exactly ``paulis[1:]`` when ``reduced=True``. They are not physical
    Z|X rows. Multiply by that same binary basis to obtain the center;
    ``center`` performs this operation directly. Phases are discarded.
    ``reduced=True`` assumes the input rows are already independent.
    """
    if not reduced:
        reduced_pauli = row_reduce(paulis)
    else:
        reduced_pauli = paulis
    return null_space(inner_product(reduced_pauli))


@njit(cache=NUMBA_CACHE)
def differences(paulis, paulis2 = None):
    """
    Return packed XOR differences, cyclically or between two matched arrays.

    This is a binary-label operation, not phase-aware Pauli multiplication.

    Args:
        paulis (ndarray): ZX form Pauli strings
        paulis2 (ndarray, optional): If provided, computes differences between paulis and paulis2 instead of cyclic differences within paulis.
    Returns:
        ndarray: Array of differences
    """
    if paulis.ndim != 1:
        raise ValueError("Packed ZX must be a one-dimensional integer array")
    _validate_legacy_array_nb(paulis)
    if paulis2 is not None:
        if paulis2.ndim != 1:
            raise ValueError("Packed ZX must be a one-dimensional integer array")
        _validate_legacy_array_nb(paulis2)
        assert paulis.shape == paulis2.shape, "paulis and paulis2 must have the same shape"
        assert paulis[0] == paulis2.flat[0], "paulis and paulis2 must have the same k value"
    n = paulis.shape[0]
    k = paulis[0]
    if paulis2 is not None:
        diffs = np.zeros((n), dtype=GLOBAL_INTEGER)
        for i in range(1,n):
            diffs[i] = paulis[i] ^ paulis2.flat[i]
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


@njit(cache=NUMBA_CACHE)
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

@njit(cache=NUMBA_CACHE)
def _packed_rows_to_binary(paulis):
    if paulis.ndim != 1:
        raise ValueError("Packed ZX must be a one-dimensional integer array")
    _validate_legacy_array_nb(paulis)
    k = paulis[0]
    binary = np.zeros((len(paulis) - 1, 2 * k), dtype=np.int8)
    for row in range(binary.shape[0]):
        value = paulis[row + 1] >> 1
        for col in range(2 * k):
            binary[row, col] = (value >> col) & 1
    return binary


@njit(cache=NUMBA_CACHE)
def center(pauli_input, reduced=False):
    """Return binary Z|X generators of V intersect V-perp, ignoring phases.

    V is the binary span of the packed input. If ``reduced=True``, the
    supplied rows must already be independent; their ordering is preserved
    when interpreting radical coefficients. Output has shape (dimension,
    2*k), with dtype uint8. This does not validate stabilizer signs.
    """
    basis = pauli_input if reduced else row_reduce(pauli_input)
    kernel = radical(basis, reduced=True)
    return matmul_mod2(kernel, _packed_rows_to_binary(basis))


@njit(cache=NUMBA_CACHE)
def centralizer(pauli_input, reduced=False):
    """Legacy alias for ``center``; returns V intersect V-perp, not V-perp.

    The historical name and packed-input ABI are retained for compatibility.
    Use ``ambient_centralizer`` for all Paulis commuting with the input.
    """
    return center(pauli_input, reduced)


@njit(cache=NUMBA_CACHE)
def ambient_centralizer(pauli_input, reduced=False):
    """Return binary Z|X generators of the ambient commutant V-perp.

    For k qubits and input binary rank r, the output has shape (2*k-r,
    2*k). It is the null space of P J, where P has Z|X rows and J swaps
    the two bit planes. ``reduced=True`` skips input basis reduction.
    Signs have no effect on commutation and are discarded.
    """
    basis = pauli_input if reduced else row_reduce(pauli_input)
    binary = _packed_rows_to_binary(basis)
    k = basis[0]
    constraints = np.empty_like(binary)
    constraints[:, :k] = binary[:, k:]
    constraints[:, k:] = binary[:, :k]
    return null_space(constraints)



def group(pauli_input):
    """Return the XOR closure of packed words (potentially exponential).

    This historical helper XORs the sign bit as well as the binary labels.
    It does not compute physical Pauli multiplication phases. Use
    ``stabilizer_reduce`` to reduce signed commuting stabilizer generators.
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

@njit(cache=NUMBA_CACHE)
def ingroup(candidates, pauli_set, reduced=False):
    """
    Determine binary-span membership modulo scalar Pauli phases.

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
            the corresponding candidate is linearly dependent on ``pauli_set`` (i.e., belongs to
            its span).
    """
    if candidates.ndim != 1 or pauli_set.ndim != 1:
        raise ValueError("Packed ZX must be a one-dimensional integer array")
    _validate_legacy_array_nb(candidates)
    _validate_legacy_array_nb(pauli_set)
    if candidates[0] != pauli_set[0]:
        raise ValueError("candidates and pauli_set must have the same k value")
    if candidates.shape[0] <= 1:
        return np.zeros(0, dtype=np.bool_)

    if reduced:
        basis = pauli_set.copy()
    else:
        basis = row_reduce(pauli_set.copy())
    #print(toString(basis))

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
    dependent = np.zeros(count, dtype=np.bool_)

    for idx in range(count):
        vec = candidates[idx + 1] >> 1
        remainder = vec
        for r in range(rank):
            pivot = pivot_cols[r]
            if pivot == -1:
                continue
            if (remainder >> pivot) & 1:
                remainder ^= basis_rows[r]
        dependent[idx] = remainder == 0

    return dependent

@njit(cache=NUMBA_CACHE)
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
    return _packed_rows_to_binary(row_reduce(pauli_input))


@njit(cache=NUMBA_CACHE)
def stabilizer_reduce_bits(z_bits, x_bits, phases):
    """Reduce commuting Hermitian generators, retaining their exact phases.

    ``z_bits`` and ``x_bits`` are binary matrices with shape (rows, qubits).
    A row denotes ``i**phases[row]`` times the tensor product of the standard
    Hermitian I/X/Y/Z matrices. Phases must be integers in 0..3; Hermitian
    generators require 0 or 2. This canonical-Y convention differs from
    storing a phase in front of a bare X**x Z**z product.

    Return independent (z, x, phases) uint8 arrays without modifying input.
    Redundant +I relations are removed. Noncommuting generators, imaginary
    phases, or a generated -I raise ValueError. The latter means there is
    no common +1 eigenspace. There is no packed-word qubit limit.
    """
    if z_bits.ndim != 2 or x_bits.ndim != 2 or phases.ndim != 1:
        raise ValueError("Expected two bit matrices and a one-dimensional phase array")
    if z_bits.shape != x_bits.shape or phases.shape[0] != z_bits.shape[0]:
        raise ValueError("Bit matrices must match and phases must have one entry per row")
    rows, qubits = z_bits.shape
    for row in range(rows):
        phase = phases[row]
        if phase != 0 and phase != 2:
            raise ValueError("Stabilizer generators must have Hermitian phases 0 or 2")
        for col in range(qubits):
            if z_bits[row, col] != 0 and z_bits[row, col] != 1:
                raise ValueError("z_bits must contain only binary values")
            if x_bits[row, col] != 0 and x_bits[row, col] != 1:
                raise ValueError("x_bits must contain only binary values")

    z = z_bits.astype(np.uint8)
    x = x_bits.astype(np.uint8)
    phase_out = phases.astype(np.uint8)
    for a in range(rows):
        for b in range(a + 1, rows):
            parity = 0
            for col in range(qubits):
                parity ^= (z[a, col] & x[b, col]) ^ (x[a, col] & z[b, col])
            if parity:
                raise ValueError("Stabilizer generators must commute")

    rank = 0
    for column in range(2 * qubits):
        selected = -1
        for row in range(rank, rows):
            bit = z[row, column] if column < qubits else x[row, column - qubits]
            if bit:
                selected = row
                break
        if selected == -1:
            continue
        if selected != rank:
            for col in range(qubits):
                z[rank, col], z[selected, col] = z[selected, col], z[rank, col]
                x[rank, col], x[selected, col] = x[selected, col], x[rank, col]
            phase_out[rank], phase_out[selected] = phase_out[selected], phase_out[rank]

        for row in range(rows):
            if row == rank:
                continue
            bit = z[row, column] if column < qubits else x[row, column - qubits]
            if not bit:
                continue
            phase = np.int64(phase_out[row]) + np.int64(phase_out[rank])
            for col in range(qubits):
                za, xa = np.int64(z[row, col]), np.int64(x[row, col])
                zb, xb = np.int64(z[rank, col]), np.int64(x[rank, col])
                new_z, new_x = za ^ zb, xa ^ xb
                # P(z,x) = i**(z*x) X**x Z**z on each qubit.
                phase += za * xa + zb * xb + 2 * za * xb - new_z * new_x
                z[row, col], x[row, col] = new_z, new_x
            phase_out[row] = phase % 4
        rank += 1
        if rank == rows:
            break

    for row in range(rank, rows):
        if phase_out[row] == 2:
            raise ValueError("Inconsistent stabilizer generators produce -I")
    return z[:rank].copy(), x[:rank].copy(), phase_out[:rank].copy()


@njit(cache=NUMBA_CACHE)
def stabilizer_reduce(packed):
    """Phase-preserving stabilizer reduction for legacy packed int64 input.

    Input and output are length-prefixed packed arrays for 0..31 qubits.
    The sign bit encodes canonical Hermitian Pauli phase (-1)**sign.
    Unlike ``row_reduce``, this rejects noncommuting or inconsistent input.
    """
    if packed.ndim != 1 or packed.size == 0:
        raise ValueError("Expected a nonempty one-dimensional packed array")
    qubits = np.int64(packed[0])
    if qubits != packed[0] or qubits < 0 or qubits > 31:
        raise ValueError("Legacy stabilizer input requires 0..31 qubits")
    rows = packed.size - 1
    z = np.zeros((rows, qubits), dtype=np.uint8)
    x = np.zeros((rows, qubits), dtype=np.uint8)
    phases = np.zeros(rows, dtype=np.uint8)
    allowed = (np.uint64(1) << np.uint64(2 * qubits + 1)) - np.uint64(1)
    for row in range(rows):
        value = np.int64(packed[row + 1])
        if value != packed[row + 1] or value < 0 or np.uint64(value) > allowed:
            raise ValueError("Packed Pauli value exceeds its declared qubit width")
        phases[row] = 2 * (value & 1)
        for col in range(qubits):
            z[row, col] = (value >> (col + 1)) & 1
            x[row, col] = (value >> (col + 1 + qubits)) & 1
    reduced_z, reduced_x, reduced_phases = stabilizer_reduce_bits(z, x, phases)
    result = np.zeros(reduced_phases.size + 1, dtype=np.int64)
    result[0] = qubits
    for row in range(reduced_phases.size):
        value = np.int64(reduced_phases[row] // 2)
        for col in range(qubits):
            value |= np.int64(reduced_z[row, col]) << (col + 1)
            value |= np.int64(reduced_x[row, col]) << (col + 1 + qubits)
        result[row + 1] = value
    return result
