import numpy as np
import unittest
from numba import jit, types, njit, prange
from numba.core.errors import NumbaTypeError, NumbaValueError
from operator import ixor
from numpy import int64


GLOBAL_INTEGER = int64

def toZX(input_data):
    """
    Convert different forms of Pauli string representations to an efficient integer representation.
    
    Args:
        input_data (str, list, tuple, list of str, np.ndarray): Input representation of Pauli string.
            - Pauli string (e.g., "XYZI")
            - List of tuples with Pauli characters and their indices (e.g., [('X',0), ('Y',1), ('Z',2)])
            - List of Pauli strings (e.g., ['XX', 'YY', '-YY'])
            - Binary string representation (e.g., "11000110" = IXYZ, format: Z|X bits)
            - NumPy array of binary arrays in ZX form (e.g., np.array([[1,1,0,0], [0,1,1,0]]))
            - Single binary array in ZX form (e.g., np.array([1,1,0,0]))
    
    Returns:
        tuple: (int, np.ndarray): Length of Pauli strings and their integer representations as a numpy array.
    """
    if not isinstance(input_data, (str, list, tuple, np.ndarray)):
        raise ValueError("Unsupported input data type. Must be a Pauli string, list of tuples, list of Pauli strings, or numpy array.")
    
    # Handle NumPy array inputs (binary arrays in ZX form)
    if isinstance(input_data, np.ndarray):
        # Convert NumPy arrays to binary strings and use existing logic
        if input_data.ndim == 1:
            # Single binary array - convert to binary string
            if len(input_data) % 2 != 0:
                raise ValueError(f"Binary array length {len(input_data)} must be even (Z|X format)")
            binary_string = ''.join(str(int(bit)) for bit in input_data)
            return toZX(binary_string)  # Recursive call with binary string
            
        elif input_data.ndim == 2:
            # Multiple binary arrays - convert each row to binary string
            if input_data.shape[1] % 2 != 0:
                raise ValueError(f"Binary array width {input_data.shape[1]} must be even (Z|X format)")
            binary_strings = [''.join(str(int(bit)) for bit in row) for row in input_data]
            return toZX(binary_strings)  # Recursive call with list of binary strings
        else:
            raise ValueError("NumPy array input must be 1D or 2D")
    
    valid_characters = {'X', 'Y', 'Z', 'I', '+', '-'}
    binary_characters = {'0', '1'}
    
    # Check if input is a binary string (format 6)
    is_binary_string = False
    if isinstance(input_data, str):
        if all(char in binary_characters for char in input_data):
            is_binary_string = True
        elif not all(char in valid_characters for char in input_data):
            raise ValueError("Input string contains invalid characters. Only 'X', 'Y', 'Z', 'I', '+', '-' or '0', '1' are allowed.")
    elif isinstance(input_data, (list, tuple)):
        for item in input_data:
            if isinstance(item, str):
                if all(char in binary_characters for char in item):
                    # This is a binary string in a list
                    continue
                elif not all(char in valid_characters for char in item):
                    raise ValueError("Input list/tuple contains invalid characters. Only 'X', 'Y', 'Z', 'I', '+', '-' or '0', '1' are allowed.")
            elif isinstance(item, tuple):
                for sub_item in item:
                    if isinstance(sub_item, str):
                        if not all(char in valid_characters for char in sub_item):
                            raise ValueError("Input list of tuples contains invalid characters. Only 'X', 'Y', 'Z', 'I', '+', and '-' are allowed.")
                    elif not isinstance(sub_item, int):
                        raise ValueError("Unsupported input data type in tuple. Must be a Pauli string or integer.")
            else:
                raise ValueError("Unsupported input data type in list/tuple. Must be a Pauli string or tuple.")

    def pauli_to_integer(pauli_str, length):
        sign_bit = 0
        if pauli_str.startswith('-'):
            sign_bit = 1
            pauli_str = pauli_str[1:]
        integer_rep = sign_bit
        for i, pauli in enumerate(pauli_str):
            i = i + 1  # Shift for sign
            if pauli not in ['X', 'Y', 'Z', 'I']:
                raise ValueError(f"Invalid Pauli character: {pauli}. Must be one of 'X', 'Y', 'Z', 'I'.")
            if pauli == 'X':
                integer_rep |= (1 << (length + i))
            elif pauli == 'Y':
                integer_rep |= (1 << (length + i))
                integer_rep |= (1 << i)
            elif pauli == 'Z':
                integer_rep |= (1 << i)
            # 'I' means identity, no action needed
        return integer_rep
    
    def binary_string_to_integer(binary_str, length):
        """Convert binary string format (Z|X bits) to integer representation."""
        if len(binary_str) != 2 * length:
            raise ValueError(f"Binary string length {len(binary_str)} does not match expected length {2 * length} for {length} qubits.")
        
        integer_rep = 0  # No sign bit for binary format
        
        # First half is Z bits, second half is X bits
        z_bits = binary_str[:length]
        x_bits = binary_str[length:]
        
        # Set Z bits (positions 1 to length)
        for i, z_bit in enumerate(z_bits):
            if z_bit == '1':
                integer_rep |= (1 << (i + 1))
        
        # Set X bits (positions length+1 to 2*length)
        for i, x_bit in enumerate(x_bits):
            if x_bit == '1':
                integer_rep |= (1 << (i + 1 + length))
        
        return integer_rep
    
    if isinstance(input_data, list):
        if all(isinstance(item, str) for item in input_data):
            # Check if all items are binary strings
            all_binary = all(all(char in binary_characters for char in item) for item in input_data)
            
            if all_binary:
                # Handle list of binary strings
                length = len(input_data[0]) // 2  # Binary strings are 2*qubits long
                pauli_integers = np.empty(len(input_data) + 1, dtype=GLOBAL_INTEGER)
                pauli_integers[0] = length
                for i, binary_str in enumerate(input_data):
                    pauli_integers[i + 1] = binary_string_to_integer(binary_str, length)
                return pauli_integers
            else:
                # Handle list of Pauli strings
                length = np.max(np.array([len(pauli) for pauli in input_data]))
                pauli_integers = np.empty(len(input_data) + 1, dtype=GLOBAL_INTEGER)
                pauli_integers[0] = length
                for i, pauli in enumerate(input_data):
                    pauli_integers[i + 1] = pauli_to_integer(pauli, length)
                return pauli_integers
        elif all(isinstance(item, tuple) for item in input_data):
            num_qubits = np.max(np.array([index for _, index in input_data])) + 1
            pauli_str = ['I'] * num_qubits
            for pauli, index in input_data:
                if pauli not in ['X', 'Y', 'Z', 'I']:
                    raise ValueError(f"Invalid Pauli character: {pauli}. Must be one of 'X', 'Y', 'Z', 'I'.")
                pauli_str[index] = pauli
            pauli_integers = np.empty(2, dtype=GLOBAL_INTEGER)
            pauli_integers[0] = num_qubits
            pauli_integers[1] = pauli_to_integer(''.join(pauli_str), num_qubits)
            return pauli_integers
    elif isinstance(input_data, str):
        if is_binary_string:
            # Handle single binary string
            length = len(input_data) // 2
            pauli_integers = np.empty(2, dtype=GLOBAL_INTEGER)
            pauli_integers[0] = length
            pauli_integers[1] = binary_string_to_integer(input_data, length)
            return pauli_integers
        else:
            # Handle single Pauli string
            length = len(input_data)
            pauli_integers = np.empty(2, dtype=GLOBAL_INTEGER)
            pauli_integers[0] = length
            pauli_integers[1] = pauli_to_integer(input_data, length)
            return pauli_integers
    else:
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


@njit()
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
    length = sym_form[0]  # Extract the length of the symplectic form
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

#@jit(nopython=True)
#1D list of tuples(int, int_array) --> tuple(int, int_array)
#@jit(types.Tuple((int64, types.Array(int64, 1,'C')))(types.List(types.Tuple((int64, types.Array(int64, 1,'C')))),), nopython=True)
@njit()
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
        return sym_forms[0]
    
    #Iterates through and gets the largest length to pad to
    max_length = 0
    total_count = 0
    #Tracks the total number of elements and the maximum string length
    for sym_form in sym_forms:
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


def left_pad(sym_form, result_size):
    """Left pads the symplectic form to cover more qubit indices. Keeps the indexing of the original form, and adds I's at the smaller indices
    
    Args:
        sym_form (int, np.ndarray): Symplectic length, array pair
        result_size (int): New length of the symplectic form
    
    Returns:
        tuple: Left padded symplectic form
    """
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
                padded_form[j + 1] |= 1 << (i + 1 + result_size)
    
    return padded_form


@njit()
def symplectic_inner_product_int(int_rep1, int_rep2, length):
    """
    Compute the symplectic inner product between two integer representations of Pauli strings of the same length.
    
    Args:
        int_rep1 (int): First integer representation.
        int_rep2 (int): Second integer representation.
        length (int): The length of the Pauli strings.
    
    Returns:
        int: Symplectic inner product (0 or 1).
    """
    product = np.int8(0)
    for i in prange(length):
        x1 = (int_rep1 >> (i + 1 + length)) & 1
        z1 = (int_rep1 >> (i + 1)) & 1
        x2 = (int_rep2 >> (i + 1 + length)) & 1
        z2 = (int_rep2 >> (i + 1)) & 1
        product = ixor(product, (x1 * z2) ^ (z1 * x2))
    return product


@njit()
def symplectic_inner_product(sym_form1, sym_form2, k=None):
    """
    Compute the symplectic inner product between two symplectic forms. Wrapper for symplectic_inner_product_int, cleans up the symplectic form structure and length comparisons

    Args:
        sym_form1 (tuple): First symplectic form as a tuple (length, integer representation).
        sym_form2 (tuple): Second symplectic form as a tuple (length, integer representation).
    
    Returns:
        int: Symplectic inner product.
    """
    if k is not None:
        length1 = length2 = k
        int_rep1 = sym_form1
        int_rep2 = sym_form2
        return symplectic_inner_product_int(int_rep1, int_rep2, k)
    else:
        length1 = sym_form1[0]
        int_rep1 = sym_form1[1]
        length2 = sym_form2[0]
        int_rep2 = sym_form2[1]
    
    # Pad the shorter form to match the length of the longer form
    if length1 < length2:
        sym_form1 = right_pad(sym_form1, length2)
        length1 = sym_form1[0]
        int_rep1 = sym_form1[1]
    elif length2 < length1:
        sym_form2 = right_pad(sym_form2, length1)
        length2 = sym_form2[0]
        int_rep2 = sym_form2[1]
    
    return symplectic_inner_product_int(int_rep1, int_rep2, length1)

@njit()
def commutes(a, b, length=None):
    """
    Check if two symplectic forms commute.
    
    Args:
        a (np.ndarray): First symplectic form as a NumPy array with the first element as the length.
        b (np.ndarray): Second symplectic form as a NumPy array with the first element as the length.
        length (int, optional): If provided, `a` and `b` are treated as integer representations.
    
    Returns:
        int: 1 if the two symplectic forms commute, 0 otherwise.
    """
    if length is not None:
        return np.int8(symplectic_inner_product_int(a, b, length) == 0)
    return np.int8(symplectic_inner_product(a, b) == 0)


def bsip_array(sym_form_input):
    """Computes the commutation matrix for a list of symplectic forms
    
    Args:
        sym_form_input (array): array containing all the symplectic forms
    
    Returns:
        array: commutation matrix
    """
    n = len(sym_form_input)-1
    length = sym_form_input[0]
    commutation_matrix = np.zeros((n,n), dtype=np.int8)
    for i in range(n):
        for j in range(i,n):
            p1 = sym_form_input[i+1]
            p2 = sym_form_input[j+1]
            commutation_matrix[i,j] = symplectic_inner_product_int(p1, p2, length=length)
            commutation_matrix[j,i] = commutation_matrix[i,j]
    return commutation_matrix


import numpy as np
@njit()
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

    # Assume the number of qubits is consistent and defined by the first element
    num_qubits = sym_form_input[0][0]

    # Pre-allocate matrices for performance - use uint8 here, convert later for matmul
    x_matrix = np.zeros((num_operators, num_qubits), dtype=np.uint8)
    z_matrix = np.zeros((num_operators, num_qubits), dtype=np.uint8)
    
    # This single loop unpacks the data; it's much faster than the previous N^2 loop.
    for i in range(num_operators):
        length, int_rep_array = sym_form_input[i]
        int_rep = int_rep_array[0]  # Extract the integer from the array
        if length > 0:
            for qubit_idx in range(length):
                # Use bit-shifting to extract X and Z parts for each operator
                x_bit = (int_rep >> (qubit_idx + 1 + length)) & 1
                z_bit = (int_rep >> (qubit_idx + 1)) & 1
                x_matrix[i, qubit_idx] = x_bit
                z_matrix[i, qubit_idx] = z_bit
            
    return x_matrix, z_matrix

@njit()
def commute_array_fast(sym_form_input):
    """
    Computes the commutation matrix for a list of symplectic forms using vectorization.

    Args:
        sym_form_input (np.ndarray): Array containing all the symplectic forms [length, int1, int2, ...]

    Returns:
        np.ndarray: The (N, N) commutation matrix where 1 means commute, 0 means anti-commute.
    """
    # Convert numpy array format to list of tuples format for unpack_sym_forms_to_matrices
    length = sym_form_input[0]
    n_operators = len(sym_form_input) - 1
    
    # Create list of tuples in the format (length, [int_representation])
    sym_form_list = []
    for i in range(1, len(sym_form_input)):
        # Each operator needs to be a tuple (length, array_with_single_int)
        sym_form_list.append((length, np.array([sym_form_input[i]])))
    
    # 1. Unpack the integer representations into a standard NumPy bit-array format.
    x_matrix, z_matrix = unpack_sym_forms_to_matrices(sym_form_list)

    # 2. Convert to int64 for matrix operations (Numba requirement)
    x_matrix_int = x_matrix.astype(np.int64)
    z_matrix_int = z_matrix.astype(np.int64)

    # 3. Compute the symplectic inner product for all pairs using matrix multiplication.
    # Use np.dot instead of @ operator for Numba compatibility
    inner_product_matrix = (np.dot(x_matrix_int, z_matrix_int.T) + np.dot(z_matrix_int, x_matrix_int.T)) % 2
    
    # 4. Invert the bits to match the `commutes` function behavior (1 for commute).
    commutation_matrix = 1 - inner_product_matrix
    
    return commutation_matrix.astype(GLOBAL_INTEGER)