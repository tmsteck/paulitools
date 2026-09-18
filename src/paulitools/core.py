import numpy as np
from numba import njit, prange, types
from numba.core.errors import NumbaValueError
from numba.extending import overload
from numpy import int64

from ._numba import NUMBA_CACHE

from .large_pauli import (
    MAX_STANDARD_QUBITS,
    PauliInt,
    PauliIntCollection,
    commutes_any as _commutes_any,
    infer_qubits as _infer_qubits_large,
    is_pauliint as _is_pauliint,
    is_pauliint_collection as _is_pauliint_collection,
    pauliints_to_standard,
    standard_to_pauliints,
    symplectic_inner_product_any as _sip_any,
    toZX_large,
    _popcount_uint64,
)


GLOBAL_INTEGER = int64

ASCII_I = np.uint8(ord('I'))
ASCII_X = np.uint8(ord('X'))
ASCII_Y = np.uint8(ord('Y'))
ASCII_Z = np.uint8(ord('Z'))


@njit(cache=NUMBA_CACHE)
def _pack_zx_bitplanes(z_bits, x_bits):
    """Pack Z and X bitplanes into legacy integer representation."""
    num_rows = z_bits.shape[0]
    length = z_bits.shape[1]
    if length > MAX_STANDARD_QUBITS:
        raise ValueError("Legacy ZX supports at most 31 qubits; use toZX_extended")
    output = np.empty(num_rows + 1, dtype=np.int64)
    output[0] = length
    for row in range(num_rows):
        value = np.int64(0)
        for col in range(length):
            if z_bits[row, col] != 0:
                value |= np.int64(1) << (col + 1)
            if x_bits[row, col] != 0:
                value |= np.int64(1) << (col + 1 + length)
        output[row + 1] = value
    return output


@njit(cache=NUMBA_CACHE)
def _pack_pauli_char_matrix(char_matrix, lengths, sign_bits, max_length):
    """Pack Pauli character matrix into ZX legacy integers."""
    num_rows = char_matrix.shape[0]
    if max_length > MAX_STANDARD_QUBITS:
        raise ValueError("Legacy ZX supports at most 31 qubits; use toZX_extended")
    output = np.empty(num_rows + 1, dtype=np.int64)
    output[0] = max_length
    for row in range(num_rows):
        value = np.int64(sign_bits[row])
        row_length = lengths[row]
        for col in range(row_length):
            code = char_matrix[row, col]
            if code == ASCII_X:
                value |= np.int64(1) << (col + 1 + max_length)
            elif code == ASCII_Y:
                value |= np.int64(1) << (col + 1)
                value |= np.int64(1) << (col + 1 + max_length)
            elif code == ASCII_Z:
                value |= np.int64(1) << (col + 1)
        output[row + 1] = value
    return output


def _normalize_binary_entries(array, encoding="auto"):
    arr = np.asarray(array)
    if encoding not in {"auto", "bits", "eigenvalues"}:
        raise ValueError("encoding must be 'auto', 'bits', or 'eigenvalues'")
    if encoding == "eigenvalues":
        if np.any((arr != -1) & (arr != 1)):
            raise ValueError("Eigenvalue inputs must contain only -1 or +1")
        return (arr < 0).astype(np.uint8)
    if encoding == "bits":
        if np.any((arr != 0) & (arr != 1)):
            raise ValueError("Bit inputs must contain only 0 or 1")
        return arr.astype(np.uint8)
    if arr.dtype == np.bool_:
        return arr.astype(np.uint8)

    if np.any(arr == -1):
        if np.any((arr != -1) & (arr != 0) & (arr != 1)):
            raise ValueError("Binary array inputs using ±1 notation must only contain -1, 0, or 1 values.")
        return (arr < 0).astype(np.uint8)

    if np.any((arr != 0) & (arr != 1)):
        raise ValueError("Binary array inputs must contain only 0/1 or ±1 values.")

    return arr.astype(np.uint8)


def _prepare_pauli_char_matrix(strings):
    count = len(strings)
    if count == 0:
        raise ValueError("Input list is empty.")

    lengths = np.zeros(count, dtype=np.int32)
    sign_bits = np.zeros(count, dtype=np.uint8)
    sanitized = []
    max_length = 0
    valid_characters = {'I', 'X', 'Y', 'Z'}

    for idx, item in enumerate(strings):
        if not isinstance(item, str):
            raise ValueError("Input list contains non-string elements.")
        if len(item) == 0:
            body = ""
            sign_bits[idx] = 0
        else:
            sign_char = item[0]
            start = 1 if sign_char in '+-' else 0
            if sign_char == '-':
                sign_bits[idx] = 1
            body = item[start:].upper()
        if not all(ch in valid_characters for ch in body):
            raise ValueError("Input list contains invalid Pauli characters. Only 'I', 'X', 'Y', 'Z' are allowed.")
        sanitized.append(body)
        lengths[idx] = len(body)
        if lengths[idx] > max_length:
            max_length = lengths[idx]

    char_matrix = np.full((count, max_length), ASCII_I, dtype=np.uint8)
    for idx, body in enumerate(sanitized):
        if lengths[idx] == 0:
            continue
        char_codes = np.frombuffer(body.encode('ascii'), dtype=np.uint8)
        char_matrix[idx, :char_codes.size] = char_codes

    return char_matrix, lengths, sign_bits, max_length


def _fast_binary_string_to_zx(strings):
    if isinstance(strings, str):
        strings_iterable = [strings]
    else:
        strings_iterable = list(strings)
        if len(strings_iterable) == 0:
            raise ValueError("Input list is empty.")

    length = len(strings_iterable[0])
    if length % 2 != 0:
        raise ValueError("Binary string length must be even (Z|X format).")

    for item in strings_iterable:
        if len(item) != length:
            raise ValueError("All binary strings must share the same length.")
        if not _is_binary_string(item):
            raise ValueError("binary_string fast path received non-binary content.")

    half = length // 2
    _validate_legacy_width(half)
    total_bits = len(strings_iterable) * length
    bits = np.fromiter(
        (1 if ch == '1' else 0 for s in strings_iterable for ch in s),
        dtype=np.uint8,
        count=total_bits,
    ).reshape(len(strings_iterable), length)

    return _pack_zx_bitplanes(bits[:, :half], bits[:, half:]).astype(GLOBAL_INTEGER)


def _fast_eigen_z_to_zx(array):
    arr = np.asarray(array)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim != 2:
        raise ValueError("eigen_z input must be 1D or 2D array-like.")
    if arr.shape[1] == 0:
        raise ValueError("eigen_z input must have at least one column.")

    _validate_legacy_width(arr.shape[1])
    z_bits = _normalize_binary_entries(arr, encoding="eigenvalues")
    x_bits = np.zeros_like(z_bits, dtype=np.uint8)
    return _pack_zx_bitplanes(z_bits, x_bits).astype(GLOBAL_INTEGER)


def _handle_binary_array_input(array, encoding="auto"):
    arr = np.asarray(array)
    if arr.ndim == 1:
        if arr.size % 2 != 0:
            raise ValueError(f"Binary array length {arr.size} must be even (Z|X format)")
        _validate_legacy_width(arr.size // 2)
        normalized = _normalize_binary_entries(arr, encoding).reshape(1, -1)
    elif arr.ndim == 2:
        if arr.shape[1] % 2 != 0:
            raise ValueError(f"Binary array width {arr.shape[1]} must be even (Z|X format)")
        _validate_legacy_width(arr.shape[1] // 2)
        normalized = _normalize_binary_entries(arr, encoding)
    else:
        raise ValueError("NumPy array input must be 1D or 2D")

    half = normalized.shape[1] // 2
    return _pack_zx_bitplanes(normalized[:, :half], normalized[:, half:]).astype(GLOBAL_INTEGER)


def _tozx_fast_path(input_data, fast_input_type):
    if fast_input_type == "binary_string":
        return _fast_binary_string_to_zx(input_data)
    if fast_input_type == "eigen_z":
        return _fast_eigen_z_to_zx(input_data)
    raise ValueError(f"Unsupported fast_input_type '{fast_input_type}'.")


def _is_binary_string(value):
    return isinstance(value, str) and all(ch in {'0', '1'} for ch in value)

def _validate_legacy_width(length):
    if length < 0 or length > MAX_STANDARD_QUBITS:
        raise ValueError("Legacy ZX supports 0 through 31 qubits; use toZX_extended")


def toZX(input_data, fast_input_type=None, *, encoding="auto"):
    """
    Convert different forms of Pauli string representations to an efficient integer representation.

    Args:
        input_data (str, list, tuple, list of str, np.ndarray): Input representation of Pauli string.
            - Pauli string (e.g., "XYZI")
            - List of tuples with Pauli characters and their indices (e.g., [('X',0), ('Y',1)])
            - List of Pauli strings (e.g., ['XX', 'YY', '-YY'])
            - Binary string representation (e.g., "11000110" = IXYZ, format: Z|X bits)
            - NumPy array of binary arrays in ZX form (e.g., np.array([[1,1,0,0], [0,1,1,0]]))
            - Single binary array in ZX form (e.g., np.array([1,1,0,0]))
        fast_input_type (str, optional): Bypass validation and assume a specific input encoding.
            Supported values:
            - ``"binary_string"``: `input_data` is a binary string or list thereof in Z|X format.
            - ``"eigen_z"``: `input_data` is a ±1 array where -1 -> 1 (Z bit set) and +1 -> 0.
        encoding (str): Numeric Z|X array encoding: ``bits`` validates 0/1;
            ``eigenvalues`` validates ±1 and maps -1 to 1. The compatibility
            default ``auto`` treats the entire array as eigenvalue notation
            when any -1 is present, and as bits otherwise. Explicit encoding
            avoids batch-dependent interpretation of all-+1 rows.

    Returns:
        np.ndarray: ZX legacy representation (first element = number of qubits).
    """

    if encoding not in {"auto", "bits", "eigenvalues"}:
        raise ValueError("encoding must be 'auto', 'bits', or 'eigenvalues'")
    if fast_input_type is not None:
        if encoding != "auto":
            raise ValueError("Specify either fast_input_type or encoding, not both")
        return _tozx_fast_path(input_data, fast_input_type)

    if not isinstance(input_data, (str, list, tuple, np.ndarray)):
        raise ValueError(
            "Unsupported input data type. Must be a Pauli string, list of tuples, list of Pauli strings, or numpy array."
        )

    if isinstance(input_data, np.ndarray):
        return _handle_binary_array_input(input_data, encoding)

    if isinstance(input_data, str):
        if _is_binary_string(input_data):
            return _fast_binary_string_to_zx(input_data)
        char_matrix, lengths, sign_bits, max_length = _prepare_pauli_char_matrix([input_data])
        _validate_legacy_width(max_length)
        return _pack_pauli_char_matrix(char_matrix, lengths, sign_bits, max_length).astype(GLOBAL_INTEGER)

    if isinstance(input_data, tuple):
        input_data = list(input_data)

    if isinstance(input_data, list):
        if len(input_data) == 0:
            raise ValueError("Input list is empty.")

        if all(isinstance(item, str) for item in input_data):
            binary_flags = [_is_binary_string(item) for item in input_data]
            if all(binary_flags):
                return _fast_binary_string_to_zx(input_data)
            if any(binary_flags):
                raise ValueError(
                    "Input list/tuple contains invalid characters. Only 'X', 'Y', 'Z', 'I', '+', '-' or '0', '1' are allowed."
                )
            char_matrix, lengths, sign_bits, max_length = _prepare_pauli_char_matrix(input_data)
            _validate_legacy_width(max_length)
            return _pack_pauli_char_matrix(char_matrix, lengths, sign_bits, max_length).astype(GLOBAL_INTEGER)

        if all(isinstance(item, tuple) for item in input_data):
            if len(input_data) == 0:
                raise ValueError("Input list of tuples is empty.")
            max_index = -1
            pauli_map = {}
            for pauli, index in input_data:
                if not isinstance(pauli, str):
                    raise ValueError("Pauli entries in tuples must be strings.")
                upper = pauli.upper()
                if upper not in {'X', 'Y', 'Z', 'I'}:
                    raise ValueError("Invalid Pauli character in tuple. Only 'X', 'Y', 'Z', 'I' are allowed.")
                if not isinstance(index, int) or index < 0:
                    raise ValueError("Pauli tuple indices must be non-negative integers.")
                pauli_map[index] = upper
                if index > max_index:
                    max_index = index
            num_qubits = max_index + 1
            pauli_str = ['I'] * num_qubits
            for idx, value in pauli_map.items():
                pauli_str[idx] = value
            return toZX(''.join(pauli_str), encoding=encoding)

        raise ValueError("Unsupported input data type in list/tuple. Must be Pauli strings or tuples.")

    raise ValueError("Unsupported input data type. Must be a Pauli string, list of tuples, or list of Pauli strings.")

def toString(integer_rep):
    """
    Convert an integer representation back to a Pauli string.
    
    Args:
        integer_rep (np.ndarray): Integer representation of a Pauli string.
    
    Returns:
        str: Pauli string representation with sign.
        
    Example:
        >>> toString(np.array([2, 7], dtype=int))
        '-ZZ'
        >> toString(np.array([4, 216], dtype=int))
        '+IXYZ'
    """
    
    if not isinstance(integer_rep, np.ndarray) or len(integer_rep) < 2:
        raise ValueError("Input must be a numpy array with at least two elements.")
    
    length = integer_rep[0]
    int_rep = integer_rep[1:]
    output = ""
    for j in prange(len(int_rep)):
        current_int = int_rep[j]
    
        sign = '-' if (current_int & 1) else '+'
        current_int >>= 1
        pauli_str = []
        for i in prange(length):
            z_bit = current_int & 1
            x_bit = (current_int >> length) & 1
            if x_bit and z_bit:
                pauli_str.append('Y')
            elif x_bit:
                pauli_str.append('X')
            elif z_bit:
                pauli_str.append('Z')
            else:
                pauli_str.append('I')
            current_int >>= 1
        output += sign + ''.join(pauli_str)
        if j < len(int_rep) - 1:
            output += ', '
    return output


@njit(cache=NUMBA_CACHE)
def right_pad(sym_form, target_length):
    """
    Right pad the symplectic form to the target length.
    Types are Tupe((int64, ndarray(int64))), int
    
    Args:
        sym_form (tuple): Symplectic form as a tuple (length, integer representation).
        target_length (int): Target length to pad to.
    
    Returns:
        tuple: Padded symplectic form.
    """
    if sym_form.ndim != 1:
        raise ValueError("Packed ZX must be a one-dimensional integer array")
    _validate_legacy_array_nb(sym_form)
    length = sym_form[0]  # Extract the length of the symplectic form
    if length < 0 or length > MAX_STANDARD_QUBITS or target_length < 0 or target_length > MAX_STANDARD_QUBITS:
        raise ValueError("Legacy ZX padding supports 0 through 31 qubits")
    int_rep = sym_form[1:]  # Extract the integer representation of the symplectic form

    # If the current length is already greater than or equal to the target length, return the original symplectic form
    if length >= target_length:
        return sym_form

    # Ensure that the current length is less than the target length
    assert length < target_length, 'Cannot right pad to a smaller size'

    count = len(int_rep)  # Get the number of integers in the integer representation

    # Create a new array to hold the padded symplectic form
    # The first element will be the new length, and the rest will be the padded integer representation
    padded_output = np.zeros(count + 1, dtype=GLOBAL_INTEGER)
    padded_output[0] = target_length  # Set the new length
    for j in range(count):
        padded_output[j + 1] = int_rep[j] & 1

    # Iterate over each bit position in the original length
    for i in prange(length):
        # Iterate over each integer in the integer representation
        for j in prange(count):
            val = int_rep[j]  # Get the current integer value

            # Check if the (i+1)-th bit is set in the current integer
            if (val >> (i + 1)) & 1:
                # Set the corresponding bit in the padded output
                padded_output[j + 1] |= 1 << (i + 1)

            # Check if the (i + 1 + length)-th bit is set in the current integer
            if (val >> (i + 1 + length)) & 1:
                # Set the corresponding bit in the padded output
                padded_output[j + 1] |= 1 << (i + 1 + target_length)


    return padded_output

@njit(cache=NUMBA_CACHE)
def append(sym_forms):
    """
    Concatenate a list of symplectic forms by right padding everything that's smaller than the maximum length.
    
    Args:
        sym_forms (list of tuples): List of tuples containing length and integer representation of symplectic forms.
    
    Returns:
        tuple: Concatenated length and integer representation.
    """
    if len(sym_forms) == 0:
        raise NumbaValueError("Input list is empty.")
    
    if len(sym_forms) == 1:
        _validate_legacy_array_nb(sym_forms[0])
        return sym_forms[0]
    
    #Iterates through and gets the largest length to pad to
    max_length = 0
    total_count = 0
    #Tracks the total number of elements and the maximum string length
    for sym_form in sym_forms:
        _validate_legacy_array_nb(sym_form)
        if sym_form[0] > max_length:
            max_length = sym_form[0]
        total_count += len(sym_form[1:])
    output = np.zeros(total_count + 1, dtype=GLOBAL_INTEGER)
    output[0] = max_length
    
    start_index = 1
    for sym_form in sym_forms:
        #Get the length, insert the padded form into the output array
        count = len(sym_form[1:]) #How many symplectic forms are in the item to concatenate
        padded_form = right_pad(sym_form, max_length)
        #each padded_form does not have the same count, so we need to keep and updated reference index:
        output[start_index: start_index + count] = padded_form[1:]
        start_index += count
    return output


concatenate_ZX = append


def left_pad(sym_form, result_size):
    """Left pads the symplectic form to cover more qubit indices. Keeps the indexing of the original form, and adds I's at the smaller indices
    
    Args:
        sym_form (int, np.ndarray): Symplectic length, array pair
        result_size (int): New length of the symplectic form
    
    Returns:
        tuple: Left padded symplectic form
    """
    if not isinstance(sym_form, np.ndarray) or sym_form.ndim != 1:
        raise ValueError("Packed ZX must be a one-dimensional integer array")
    _validate_legacy_array_nb(sym_form)
    _validate_legacy_width(result_size)
    assert sym_form[0] <= result_size, 'Cannot left pad to a smaller size'
    if sym_form[0] == result_size:
        return sym_form
    
    length = sym_form[0]
    sym_ints = sym_form[1:]
    len_ints = len(sym_ints)
    padded_form = np.zeros(len(sym_form), dtype=GLOBAL_INTEGER)
    padded_form[0] = result_size
    
    shift_amount = result_size - length
    
    # Copy over the signs: The 0 index of the binary form copies to the zero index of the padded form:
    for j in range(len_ints):
        padded_form[j + 1] = sym_ints[j] & 1
    
    for i in range(length):
        for j in range(len_ints):
            val = sym_ints[j]
            if (val >> (i + 1)) & 1:
                padded_form[j + 1] |= 1 << (i + 1 + shift_amount)
            if (val >> (i + 1 + length)) & 1:
                padded_form[j + 1] |= 1 << (i + 1 + result_size + shift_amount)
    
    return padded_form


@njit(cache=NUMBA_CACHE)
def _validate_legacy_array_nb(data):
    """Validate the integer-array ABI once at a public compiled boundary."""
    if data.ndim != 1 or data.size == 0:
        raise ValueError("Packed ZX must be a nonempty one-dimensional integer array")
    k = data[0]
    if k < 0 or k > MAX_STANDARD_QUBITS:
        raise ValueError("Legacy ZX supports 0 through 31 qubits")
    limit = np.uint64(1) << np.uint64(2 * k + 1)
    for i in range(1, data.size):
        if data[i] < 0 or np.uint64(data[i]) >= limit:
            raise ValueError("Packed operator has bits outside its declared qubit width")


@njit(cache=NUMBA_CACHE, inline="always")
def _symplectic_inner_product_int_unchecked(int_rep1, int_rep2, length):
    """Packed parity after a caller has validated the width and representation.

    Keep raising guards outside batch loops: inlining a guard with an exception
    into prange prevents Numba from parallelizing its output-row loop.
    """
    mask = (np.uint64(1) << np.uint64(length)) - np.uint64(1)
    a = np.uint64(int_rep1) >> np.uint64(1)
    b = np.uint64(int_rep2) >> np.uint64(1)
    az = a & mask
    bz = b & mask
    ax = (a >> np.uint64(length)) & mask
    bx = (b >> np.uint64(length)) & mask
    return np.int8(_popcount_uint64((ax & bz) ^ (az & bx)) & 1)


@njit(cache=NUMBA_CACHE, inline="always")
def symplectic_inner_product_int(int_rep1, int_rep2, length):
    """Return symplectic parity using unsigned packed words and LLVM ctpop."""
    if length < 0 or length > MAX_STANDARD_QUBITS:
        raise ValueError("Legacy ZX supports 0 through 31 qubits")
    return _symplectic_inner_product_int_unchecked(int_rep1, int_rep2, length)


@njit(cache=NUMBA_CACHE)
def _symplectic_inner_product_zx_nb(sym_form1, sym_form2):
    if sym_form1.ndim != 1 or sym_form2.ndim != 1:
        raise ValueError("Single-Pauli inputs must be one-dimensional packed arrays")
    _validate_legacy_array_nb(sym_form1)
    _validate_legacy_array_nb(sym_form2)
    if len(sym_form1) != 2 or len(sym_form2) != 2:
        raise ValueError("Each input must contain exactly one packed Pauli")
    length1, length2 = sym_form1.flat[0], sym_form2.flat[0]
    int_rep1, int_rep2 = sym_form1.flat[1], sym_form2.flat[1]
    if length1 < length2:
        int_rep1 = right_pad(np.array([length1, int_rep1]), length2)[1]
    elif length2 < length1:
        int_rep2 = right_pad(np.array([length2, int_rep2]), length1)[1]
    return symplectic_inner_product_int(int_rep1, int_rep2, max(length1, length2))


def _symplectic_inner_product_dispatch(sym_form1, sym_form2, k):
    """Compile-time scalar/array dispatch used by the public njit wrapper."""
    if k is None:
        return _symplectic_inner_product_zx_nb(sym_form1, sym_form2)
    return symplectic_inner_product_int(sym_form1, sym_form2, k)


@overload(_symplectic_inner_product_dispatch)
def _overload_symplectic_inner_product_dispatch(sym_form1, sym_form2, k):
    # Numba versions supporting this package need type-level dispatch to avoid
    # typing ndarray-only validation in the explicit scalar-k specialization.
    if isinstance(k, types.NoneType) or (isinstance(k, types.Omitted) and k.value is None):
        def array_impl(sym_form1, sym_form2, k):
            return _symplectic_inner_product_zx_nb(sym_form1, sym_form2)
        return array_impl

    def scalar_impl(sym_form1, sym_form2, k):
        return symplectic_inner_product_int(sym_form1, sym_form2, k)
    return scalar_impl


@njit(cache=NUMBA_CACHE)
def symplectic_inner_product(sym_form1, sym_form2, k=None):
    """Compute the symplectic product of two single packed operators.

    With explicit ``k``, inputs are packed scalar integers. Otherwise each input
    must be a length-prefixed array containing exactly one operator; the legacy
    API right-pads the shorter width before comparing.
    """
    return _symplectic_inner_product_dispatch(sym_form1, sym_form2, k)


@njit(cache=NUMBA_CACHE)
def _commutes_zx_nb(sym_form1, sym_form2):
    if sym_form1.ndim != 1 or sym_form2.ndim != 1:
        raise ValueError("Packed ZX must be a one-dimensional integer array")
    _validate_legacy_array_nb(sym_form1)
    _validate_legacy_array_nb(sym_form2)
    if len(sym_form1) < 2 or len(sym_form2) < 2:
        raise ValueError("Each input must contain at least one packed Pauli")
    # Preserve the legacy first-operator convention explicitly. Batch callers
    # should use the pairwise matrix APIs instead.
    return np.int8(_symplectic_inner_product_zx_nb(sym_form1[:2], sym_form2[:2]) == 0)

@njit(cache=NUMBA_CACHE)
def _commutes_int_nb(int_sym_form1, int_sym_form2, length):
    return np.int8(symplectic_inner_product_int(int_sym_form1, int_sym_form2, length) == 0)


def commutes(sym_form1, sym_form2, length=None):
    """
    Determine whether two Pauli operators commute.

    With ``length=None``, validate both packed arrays and compare their first
    operators, preserving the historical behavior for multi-operator inputs.
    The shorter operator is implicitly right-padded. Use ``bsip_array`` or
    ``commute_array_fast`` to compare a whole batch. With explicit ``length``,
    inputs are packed scalar integers.
    """
    if length is None:
        return bool(_commutes_zx_nb(sym_form1, sym_form2))
    return bool(
        _commutes_int_nb(
            np.int64(sym_form1),
            np.int64(sym_form2),
            np.int64(length),
        )
    )


@njit(cache=NUMBA_CACHE, parallel=True, nogil=True)
def _bsip_array_parallel(sym_form_input):
    n = sym_form_input.size - 1
    k = sym_form_input[0]
    result = np.empty((n, n), dtype=np.int8)
    # Each worker owns an entire output row; no cross-row writes.
    for i in prange(n):
        for j in range(n):
            result[i, j] = _symplectic_inner_product_int_unchecked(
                sym_form_input[i + 1], sym_form_input[j + 1], k)
    return result


@njit(cache=NUMBA_CACHE, nogil=True)
def bsip_array(sym_form_input, parallel=False):
    """Pairwise symplectic parity: 1 anticommutes, 0 commutes.

    The optional parallel path is useful for large batches; the serial path
    avoids thread-launch overhead for the small default workload.
    """
    _validate_legacy_array_nb(sym_form_input)
    if parallel:
        return _bsip_array_parallel(sym_form_input)
    n = sym_form_input.size - 1
    k = sym_form_input[0]
    result = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(i + 1, n):
            value = _symplectic_inner_product_int_unchecked(
                sym_form_input[i + 1], sym_form_input[j + 1], k)
            result[i, j] = value
            result[j, i] = value
    return result


@njit(cache=NUMBA_CACHE)
def unpack_sym_forms_to_matrices(sym_form_input):
    """
    Converts a list of (length, integer) symplectic forms into Z and X bit matrices.

    Args:
        sym_form_input (list): A list of tuples, where each is (length, int_array).

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the X-matrix and Z-matrix.
    """
    num_operators = len(sym_form_input)
    if num_operators == 0:
        return np.zeros((0, 0), dtype=np.uint8), np.zeros((0, 0), dtype=np.uint8)

    num_qubits = sym_form_input[0][0]
    if num_qubits < 0 or num_qubits > MAX_STANDARD_QUBITS:
        raise ValueError("Legacy ZX supports 0 through 31 qubits")

    # Pre-allocate matrices for performance - use uint8 here, convert later for matmul
    x_matrix = np.zeros((num_operators, num_qubits), dtype=np.uint8)
    z_matrix = np.zeros((num_operators, num_qubits), dtype=np.uint8)
    
    # This single loop unpacks the data; it's much faster than the previous N^2 loop.
    for i in range(num_operators):
        length, int_rep_array = sym_form_input[i]
        if length != num_qubits:
            raise ValueError("All rows must have the same qubit count")
        if int_rep_array.ndim != 1 or int_rep_array.size == 0:
            raise ValueError("Each row must contain a nonempty one-dimensional packed array")
        int_rep = int_rep_array.flat[0]
        limit = np.uint64(1) << np.uint64(2 * length + 1)
        if int_rep < 0 or np.uint64(int_rep) >= limit:
            raise ValueError("Packed operator has bits outside its declared qubit width")
        if length > 0:
            for qubit_idx in range(length):
                # Use bit-shifting to extract X and Z parts for each operator
                x_bit = (int_rep >> (qubit_idx + 1 + length)) & 1
                z_bit = (int_rep >> (qubit_idx + 1)) & 1
                x_matrix[i, qubit_idx] = x_bit
                z_matrix[i, qubit_idx] = z_bit
            
    return x_matrix, z_matrix

@njit(cache=NUMBA_CACHE, nogil=True)
def commute_array_fast(sym_form_input, parallel=False):
    """Pairwise commutation: 1 commutes, 0 anticommutes.

    Uses packed compiled parity kernels rather than unsupported integer BLAS.
    """
    result = bsip_array(sym_form_input, parallel).astype(GLOBAL_INTEGER)
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            result[i, j] = 1 - result[i, j]
    return result


# ---------------------------------------------------------------------------
# Extended representation helpers
# ---------------------------------------------------------------------------

def _should_use_large_representation(input_data, force_large: bool = False) -> bool:
    if force_large:
        return True
    try:
        n_qubits = _infer_qubits_large(input_data)
    except Exception:
        return False
    return n_qubits > MAX_STANDARD_QUBITS


def toZX_extended(input_data, force_large: bool = False, *, encoding="auto"):
    """Extended variant of :func:`toZX` that supports 64+ qubit systems.

    When the number of qubits exceeds :data:`MAX_STANDARD_QUBITS` (31) the legacy
    integer representation would overflow the 64-bit container.  This function
    falls back to :mod:`large_pauli` and returns a
    :class:`~large_pauli.PauliIntCollection` instead.  For smaller systems the
    original representation is preserved unless ``force_large`` is set.
    """

    use_large = _should_use_large_representation(input_data, force_large)
    if use_large:
        # For forced conversion on small instances we reuse the legacy
        # conversion to preserve sign handling before upgrading to PauliInt.
        if force_large and not _should_use_large_representation(input_data):
            legacy = toZX(input_data, encoding=encoding)
            return standard_to_pauliints(legacy)
        return toZX_large(input_data, encoding=encoding)

    legacy_result = toZX(input_data, encoding=encoding)
    if int(legacy_result[0]) > MAX_STANDARD_QUBITS:
        return standard_to_pauliints(legacy_result)
    return legacy_result


def toString_extended(pauli_data) -> str:
    """Extended variant of :func:`toString` that understands large operators."""
    if _is_pauliint(pauli_data):
        return pauli_data.to_string()
    if _is_pauliint_collection(pauli_data):
        return ", ".join(pauli.to_string() for pauli in pauli_data.paulis)
    if isinstance(pauli_data, list) and pauli_data and _is_pauliint(pauli_data[0]):
        return ", ".join(pauli.to_string() for pauli in pauli_data)
    return toString(pauli_data)


def symplectic_inner_product_extended(a, b, k=None):
    """Dispatch symplectic inner product between legacy and extended forms."""
    if k is not None:
        return symplectic_inner_product(a, b, k=k)

    if _is_pauliint_collection(a):
        if len(a) != 1:
            raise ValueError("Expected a single Pauli operator in the collection")
        a = a.paulis[0]
    if _is_pauliint_collection(b):
        if len(b) != 1:
            raise ValueError("Expected a single Pauli operator in the collection")
        b = b.paulis[0]

    if _is_pauliint(a) or _is_pauliint(b):
        return _sip_any(a, b)
    return symplectic_inner_product(a, b)


def commutes_extended(a, b, length=None):
    """Extended commutation check with support for large operators."""
    if length is not None:
        return commutes(a, b, length=length)

    if _is_pauliint_collection(a):
        if len(a) != 1:
            raise ValueError("Expected a single Pauli operator in the collection")
        a = a.paulis[0]
    if _is_pauliint_collection(b):
        if len(b) != 1:
            raise ValueError("Expected a single Pauli operator in the collection")
        b = b.paulis[0]

    if _is_pauliint(a) or _is_pauliint(b):
        return np.int8(_commutes_any(a, b))
    return commutes(a, b)


def to_standard_if_possible(pauli_data):
    """Convert extended representations back to the legacy form when safe."""
    if _is_pauliint(pauli_data):
        collection = PauliIntCollection(pauli_data.n_qubits, [pauli_data])
        return pauliints_to_standard(collection)
    if _is_pauliint_collection(pauli_data):
        return pauliints_to_standard(pauli_data)
    if isinstance(pauli_data, list) and pauli_data and _is_pauliint(pauli_data[0]):
        collection = PauliIntCollection(pauli_data[0].n_qubits, pauli_data)
        return pauliints_to_standard(collection)
    return pauli_data

#@njit()
def commutator(pauli1, pauli2):
    """ Returns the array of all valid communtators between the two sets of paulis
    
    Args:
        pauli1: First Pauli operator or collection (legacy or extended).
        pauli2: Second Pauli operator or collection (legacy or extended).
    Returns:
        List of Pauli operators or collections (legacy or extended) that are the commutators of the inputs.
    """
    n1 = pauli1[0]
    n2 = pauli2[0]
    if n1 != n2:
        raise ValueError("Pauli operators must act on the same number of qubits to compute commutators.")
    if _is_pauliint_collection(pauli1):
        raise Exception("Commutator not implemented for extended pauli strings")
    if _is_pauliint_collection(pauli2):
        raise Exception("Commutator not implemented for extended pauli strings.")
    outputs = []
    for a in pauli1[1:]:
        for b in pauli2[1:]:
            if not commutes(a, b, length=n1):
                #Compute the commutator:
                outputs.append(a ^ b)
    return np.concatenate((np.array([n1]), np.array(outputs)))
