import numpy as np
import unittest
from numba import jit, types, njit, prange
from numba.core.errors import NumbaTypeError, NumbaValueError
from operator import ixor
from numpy import int64

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from core import GLOBAL_INTEGER, symplectic_inner_product, toZX, commutes, symplectic_inner_product_int
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
        yParity = symplectic_inner_product_int(pauli, Y_check[1],k)
        comm_value = symplectic_inner_product_int(shot, pauli, k)
        return np.power(-1, ((yParity + comm_value) % 2))
    
    expectation = 0.0
    for i in range(len(shots)):
        shot_value = shots[i, 0]
        prob = shots[i, 1]
        sign = get_sign(np.array([k, shot_value]), pauli)
        expectation += sign * prob
    
    return expectation
        
        
#@njit()
def filtered_purity(generators, shots, shot_parities=np.zeros(0, dtype=np.int8)):
    """
    Computes the filtered purity of some set of generators given noisy shots. 
    
    $$
    \langle (-1)^{\pi_y(i)(\Pi_{g \in G} ((-1)^{\pi_y(g) + \langle i,g \rangle} == 1))} \rangle_{i \in S}
    $$
    First, compute the inner product between G and the set of shots S. 
    Then, compute as vectors the $Y$ parity of each shot, and of each generator
    
    
    Args:
        samples (ndarray): ZX array of samples (ie, Pauli group element generators $g\in G$)
        shots (nadarray): ZX array of the shots (ie, direct samples with the conjugate problem)
        shot_parities (ndarray): Precomputed Y parities of the shots, if available
    
    Returns:
        float: The computed filtered purity
    """
    if len(generators) <= 1 or len(shots) <= 1:
        raise Exception("Input is trivial. either the input is missing the qubit information at index zero, or it is an empty set being passed")
    
    k = generators[0]
    num_generators = len(generators) - 1
    num_shots = len(shots) - 1
    
    # Precompute all parities
    if len(shot_parities) != num_shots:
        shot_parities = np.zeros(num_shots, dtype=np.int8)
        for i in range(num_shots):
            shot_parities[i] = getParity(np.array([k, shots[i+1]]), 'Y')
    
    gen_parities = np.zeros(num_generators, dtype=np.int8)
    for j in range(num_generators):
        gen_parities[j] = getParity(np.array([k, generators[j+1]]), 'Y')
    
    # Create the full inner product matrix
    inner_matrix = np.zeros((num_shots, num_generators), dtype=np.int8)
    for i in range(num_shots):
        for j in range(num_generators):
            inner_matrix[i, j] = symplectic_inner_product_int(shots[i+1], generators[j+1], k)
    
    # Vectorized computation of the filtered purity
    # Add generator parities to each row of the inner product matrix
    exponent_matrix = (inner_matrix + gen_parities.reshape(1, -1)) % 2
    
    # Sum along generator axis to get total exponent for each shot
    total_exponents = np.sum(exponent_matrix, axis=1) % 2
    
    # Only keep shots where total exponent is 0 (generator product = +1)
    valid_shots = (total_exponents == 0)
    
    # Compute purity contribution from valid shots
    purity_contributions = np.where(shot_parities == 0, 1, -1)
    filtered_contributions = purity_contributions * valid_shots.astype(np.int8)
    
    return np.sum(filtered_contributions) / num_shots
    
    
def filtered_purity_reference(generators, shots, shot_parities=None):
    """
    Reference implementation of filtered purity - slow but easy to verify.
    
    Computes: ⟨(-1)^{π_y(i)} * ∏_{g ∈ G} δ((-1)^{π_y(g) + ⟨i,g⟩} == 1)⟩_{i ∈ S}
    
    Where:
    - π_y(x) is the Y parity of Pauli string x
    - ⟨i,g⟩ is the symplectic inner product between shot i and generator g
    - δ(...) is 1 if condition is true, 0 otherwise
    - The product ∏_{g ∈ G} checks if shot i is stabilized by ALL generators
    
    Args:
        generators (ndarray): ZX array [k, g1, g2, ...] where k is num_qubits
        shots (ndarray): ZX array [k, s1, s2, ...] where k is num_qubits
        shot_parities (ndarray, optional): Precomputed Y parities of shots
    
    Returns:
        float: The filtered purity value
    """
    if len(generators) <= 1 or len(shots) <= 1:
        print("Trivial case: not enough generators or shots")
        return 0.0
    
    k = generators[0]  # number of qubits
    num_generators = len(generators) - 1
    num_shots = len(shots) - 1
    
    print(f"Computing filtered purity for {num_generators} generators and {num_shots} shots on {k} qubits")
    
    # Step 1: Precompute Y parities if not provided
    if shot_parities is None or len(shot_parities) != num_shots:
        print("Computing shot Y parities...")
        shot_parities = np.zeros(num_shots, dtype=np.int8)
        for i in range(num_shots):
            shot_pauli = np.array([k, shots[i+1]])
            shot_parities[i] = getParity(shot_pauli, 'Y')
            if i < 5:  # Debug first few
                print(f"  Shot {i}: {shot_pauli} -> Y parity = {shot_parities[i]}")
    
    # Step 2: Precompute generator Y parities
    print("Computing generator Y parities...")
    gen_parities = np.zeros(num_generators, dtype=np.int8)
    for j in range(num_generators):
        gen_pauli = np.array([k, generators[j+1]])
        gen_parities[j] = getParity(gen_pauli, 'Y')
        print(f"  Generator {j}: {gen_pauli} -> Y parity = {gen_parities[j]}")
    
    # Step 3: For each shot, check if it's stabilized by ALL generators
    print("\nProcessing each shot...")
    total_purity = 0.0
    stabilized_shots = 0
    
    for i in range(num_shots):
        shot_val = shots[i+1]
        shot_y_parity = shot_parities[i]
        
        print(f"\nShot {i}: value={shot_val}, Y_parity={shot_y_parity}")
        
        # Check stabilization by each generator
        is_stabilized_by_all = True
        stabilization_details = []
        
        for j in range(num_generators):
            gen_val = generators[j+1]
            gen_y_parity = gen_parities[j]
            
            # Compute symplectic inner product ⟨shot, generator⟩
            inner_prod = symplectic_inner_product(
                shot_val, 
                gen_val, k
            )
            
            # Compute the exponent for this generator: π_y(g) + ⟨i,g⟩
            exponent = (gen_y_parity + inner_prod) % 2
            
            # Compute the sign: (-1)^exponent
            sign = 1 if exponent == 0 else -1
            
            stabilization_details.append({
                'generator': j,
                'gen_val': gen_val,
                'gen_y_parity': gen_y_parity,
                'inner_product': inner_prod,
                'exponent': exponent,
                'sign': sign
            })
            
            # If any generator gives sign = -1, this shot is not stabilized
            if sign == -1:
                is_stabilized_by_all = False
        
        # Print detailed stabilization info for first few shots
        if i < 3:
            print(f"  Stabilization details:")
            for detail in stabilization_details:
                print(f"    Gen {detail['generator']}: "
                      f"⟨{shot_val},{detail['gen_val']}⟩={detail['inner_product']}, "
                      f"exp={detail['exponent']}, sign={detail['sign']}")
        
        # Only include this shot if stabilized by ALL generators
        if is_stabilized_by_all:
            stabilized_shots += 1
            # Contribution is (-1)^{π_y(shot)}
            shot_sign = 1 if shot_y_parity == 0 else -1
            total_purity += shot_sign
            
            if i < 3:
                print(f"  → STABILIZED: contributing {shot_sign} to purity")
        else:
            if i < 3:
                print(f"  → NOT STABILIZED: contributing 0 to purity")
    
    # Step 4: Compute final result
    filtered_purity_value = total_purity / num_shots
    
    print(f"\nFinal Results:")
    print(f"  Total shots: {num_shots}")
    print(f"  Stabilized shots: {stabilized_shots}")
    print(f"  Total purity sum: {total_purity}")
    print(f"  Filtered purity: {total_purity}/{num_shots} = {filtered_purity_value}")
    
    return filtered_purity_value