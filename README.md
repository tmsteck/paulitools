# PauliTools - Efficient Pauli String Operations

PauliTools is a Python library for efficient manipulation and analysis of Pauli strings using binary symplectic representations. The library leverages Numba for high-performance computations and provides tools for quantum group theory operations.

## Package Structure

The `src/` directory contains the core functionality of PauliTools:

### 📁 Core Files

#### `core.py` - Central Pauli String Operations
The main module containing fundamental operations for Pauli string manipulation.

**Key Functions:**
- **`toZX(input_data)`** - Convert various Pauli string formats to efficient integer representation
  - Supports: strings ("XYZI"), lists of strings (['XX', 'YY']), tuples with indices [('X',0), ('Y',1)]
  - Returns: numpy array with length and integer representations
  
- **`toString(integer_rep)`** - Convert integer representation back to Pauli string format
  - Input: numpy array with integer representations
  - Output: string representation with signs (e.g., '+XYZI', '-ZZ')

- **Padding Operations:**
  - **`right_pad(sym_form, target_length)`** - Right-pad symplectic form with identities
  - **`left_pad(sym_form, result_size)`** - Left-pad symplectic form maintaining qubit indexing
  - **`concatenate_ZX(sym_forms)`** - Concatenate multiple symplectic forms with automatic padding

- **Symplectic Operations:**
  - **`symplectic_inner_product(sym_form1, sym_form2)`** - Compute symplectic inner product
  - **`symplectic_inner_product_int(int_rep1, int_rep2, length)`** - Integer version for performance
  - **`commutes(a, b, length=None)`** - Check if two Pauli operators commute

- **Commutation Analysis:**
  - **`commute_array(sym_form_input)`** - Generate commutation matrix for multiple Pauli operators
  - **`commute_array_fast(sym_form_input)`** - Vectorized version using matrix operations
  - **`unpack_sym_forms_to_matrices(sym_form_input)`** - Convert to X/Z bit matrices for fast operations

#### `group.py` - Pauli Group Theory Operations
Advanced group-theoretic operations for stabilizer groups and quantum error correction.

**Key Functions:**
- **`row_reduce(input_pauli)`** - Gaussian elimination on Pauli operators using bitwise operations
  - Performs row reduction over GF(2) to find linearly independent generators
  - Removes zero rows and maintains sign bit integrity
  
- **`null_space(A)`** - Compute null space of binary matrix over GF(2)
  - Numba-optimized implementation using row reduction
  - Essential for finding stabilizer group properties
  
- **`inner_product(paulis)`** - Compute symplectic inner product matrix
  - Creates NxN matrix of all pairwise symplectic inner products
  - Used for analyzing commutation relationships in groups

- **`radical(paulis, reduced=False)`** - Find the radical (center) of a Pauli group
  - Computes generators that commute with all group elements
  - Optional pre-reduction for performance

- **`centralizer(pauli_input, reduced=False)`** - Compute the centralizer of a Pauli group
  - Finds all operators that commute with the given group
  - Combines radical computation with kernel analysis

#### `util.py` - Utility Functions and Analysis Tools
Supporting functions for probability calculations and specialized operations.

**Key Functions:**
- **Binary Conversion:**
  - **`toBinary(pauli)`** - Convert Pauli integers to binary matrix representation
  - **`convert_array_type(arr, dtype)`** - Type conversion helper for Numba compatibility

- **Parity Analysis:**
  - **`getParity(pauli, basis='Y')`** - Calculate parity of X, Y, or Z operators in a Pauli string
  - **`popcount(n)`** - Efficient bit counting (Hamming weight)

- **Expectation Value Calculations:**
  - **`get_pauli_obs(pauli_input, probs, parallel=False)`** - Calculate Pauli operator expectation values
  - **`get_pauli_pauli_obs(pauli_input, probs, parallel=False)`** - Include Y-parity of measurement outcomes
  - **`Pauli_expectation(shots, pauli)`** - Direct expectation calculation from shot data

- **Advanced Group Operations:**
  - **`getCentralizer(counts, return_generators=False)`** - Generate and analyze stabilizer groups from experimental data
  - Integrates with quantum measurement data to infer underlying group structure

#### `__init__.py` - Package Initialization
Exposes all major functions from core, group, and util modules for easy access:
```python
from .core import *
from .group import *  
from .util import *
```

This allows direct importing: `from paulitools import toZX, commutes, radical, getParity`

## 🔧 Technical Features

### Performance Optimizations
- **Numba JIT Compilation**: Most functions use `@njit()` for near-C performance
- **Bitwise Operations**: Efficient manipulation using integer bit operations
- **Vectorized Operations**: Matrix-based commutation analysis for large operator sets
- **Memory Efficiency**: Compact integer representation of Pauli operators

### Data Formats
- **ZX Representation**: Pauli operators stored as integers with separate X and Z bit fields
- **Sign Handling**: Dedicated sign bit for phase tracking
- **Flexible Input**: Multiple input formats automatically detected and converted

### Mathematical Foundation
- **Symplectic Geometry**: Based on symplectic inner product over GF(2)
- **Stabilizer Formalism**: Full support for stabilizer group operations
- **Linear Algebra over GF(2)**: Gaussian elimination and null space computation

## 🚀 Usage Examples

```python
from paulitools import toZX, toString, commutes, radical

# Convert Pauli strings to efficient representation
paulis = toZX(['XX', 'YY', 'ZZ'])  # All commuting operators
print(f"Encoded as: {paulis}")

# Check commutation
p1 = toZX('XX')
p2 = toZX('ZI')
print(f"XX and ZI commute: {bool(commutes(p1, p2))}")

# Analyze group structure
group_generators = toZX(['XX', 'ZZ', 'XI'])
center = radical(group_generators)
print(f"Group center: {toString(center)}")

# Convert back to string representation
result = toString(paulis)
print(f"Decoded: {result}")
```

## 🧪 Dependencies

- **NumPy**: Array operations and linear algebra
- **Numba**: JIT compilation for performance
- **Galois**: GF(2) arithmetic (optional, for advanced features)
- **Joblib**: Parallel processing (optional)

## 📊 Performance Notes

- Functions with `@njit()` decorator compile on first use (slight initial delay)
- Large operator sets benefit significantly from `commute_array_fast()`
- Binary operations are optimized for up to 64-qubit systems
- Memory usage scales as O(n) for n operators, O(n²) for commutation matrices

## 🔬 Applications

- **Quantum Error Correction**: Stabilizer code analysis
- **Quantum Simulation**: Pauli operator manipulation
- **Quantum Algorithms**: Efficient Hamiltonian representation
- **Research**: Group theory analysis of quantum systems

This library provides a comprehensive toolkit for working with Pauli operators in quantum computing applications, with emphasis on computational efficiency and mathematical rigor.
