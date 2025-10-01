# PauliTools - Efficient Pauli String Operations

PauliTools is a Python library for efficient manipulation and analysis of Pauli strings using binary symplectic representations. The library leverages Numba for high-performance computations and provides tools for quantum group theory operations.

## Package Structure & Quick Reference

PauliTools ships as a single `paulitools` Python package. You can treat it as a
black box—every public entrypoint is exported at module level—so browsing the
`src/` tree is optional. The sections below highlight the most commonly used
APIs.

### Quickstart

```bash
pip install paulitools
```

```python
from paulitools import (
    toZX, toZX_extended,
    row_reduce, radical,
    save_pauli_data, append_pauli_data, load_pauli_data,
)

# Convert strings to binary symplectic form
gens = toZX(["XX", "YY", "ZZ"])

# Analyse group structure
reduced = row_reduce(gens)
center = radical(gens)

# Persist results incrementally
save_pauli_data("stabilizers.pauli", gens)
append_pauli_data("stabilizers.pauli", toZX(["XI", "IZ"]))

restored = load_pauli_data("stabilizers.pauli")
```

### Storage & Persistence API

The `storage` module provides an append-friendly container format that stores
NumPy `.npy` payloads inside a log-structured binary file. Use it whenever you
need to checkpoint work or build large operator libraries.

| Function | Description |
| --- | --- |
| `save_pauli_data(path, data, append=False, user_metadata=None)` | Write a legacy ZX array or `PauliIntCollection` to disk. Set `append=True` to add a new batch to an existing file. |
| `append_pauli_data(path, data)` | Shorthand for `save_pauli_data(..., append=True)`. |
| `load_pauli_data(path, include_metadata=False)` | Read and reconstruct the stored operators. Returns either the legacy array or a `PauliIntCollection`; optionally returns `(data, metadata)` when `include_metadata=True`. |
| `iter_pauli_records(path)` | Stream batches from disk without materialising the full dataset. Useful for large archives. |
| `SerializationError` | Raised when file headers, chunk sizes, or checksums fail validation. |

> **Tip:** All payloads remain NumPy-portable because they are stored as
> `.npy` blobs. If you change the number of qubits or the ZX length, open a new
> file; appends enforce consistent dimensions to keep the container compact.

### Module Overview

The following tables summarise the rest of the public API. All functions are
importable directly from `paulitools`.

#### Core Operations (`paulitools.core`)

| Function | Purpose |
| --- | --- |
| `toZX(input_data, fast_input_type=None)` | Parse Pauli strings, tuples, or binary/eigen arrays into legacy ZX integer form. Fast paths available via ``fast_input_type``. |
| `toZX_extended(input_data, force_large=False)` | Automatically selects the large-operator backend (`PauliIntCollection`) when the system exceeds 31 qubits. |
| `toString(integer_rep)` / `toString_extended(pauli_data)` | Convert legacy or extended forms back to human-readable strings. |
| `right_pad`, `left_pad`, `append` | Resize or concatenate symplectic forms. |
| `symplectic_inner_product`, `symplectic_inner_product_extended` | Compute the symplectic inner product for legacy or mixed representations. |
| `commutes`, `commutes_extended` | Check pairwise commutation. |
| `commute_array_fast`, `bsip_array` | Build commutation matrices for operator sets. |
| `to_standard_if_possible` | Down-convert a large representation to legacy form when the qubit count permits. |

#### Group Theory (`paulitools.group`)

| Function | Purpose |
| --- | --- |
| `row_reduce(paulis)` | Gaussian elimination over GF(2) to find independent generators. |
| `row_space(paulis)` | Generate the row space of an operator set. |
| `null_space(matrix)` | Compute the GF(2) null space of a binary matrix. |
| `inner_product(paulis)` | Produce the full pairwise symplectic inner product matrix. |
| `radical(paulis, reduced=False)` | Find the center (radical) of a stabiliser group. |
| `centralizer(paulis, reduced=False)` | Compute all operators commuting with a given set. |
| `differences(paulis, paulis2=None)` | Compare generators or compute relative differences between two sets. |

#### Utilities (`paulitools.util`)

| Function | Purpose |
| --- | --- |
| `toBinary(pauli)` | Convert legacy ZX integers into `(Z|X)` binary matrices. |
| `convert_array_type(arr, dtype)` | Cast arrays to Numba-friendly dtypes. |
| `popcount`, `getParity(pauli, basis)` | Bit-counting helpers for interpreting operators. |
| `get_pauli_obs`, `get_pauli_pauli_obs`, `Pauli_expectation` | Expectation value utilities for simulation data. |
| `getCentralizer(counts, return_generators=False)` | Stitch measurement data into stabiliser descriptions. |

#### Large Operator Helpers (`paulitools.large_pauli`)

| Class / Function | Purpose |
| --- | --- |
| `PauliInt`, `PauliIntCollection` | Structured storage for arbitrarily large Pauli operators. Compatible with Numba. |
| `create_pauli_struct`, `pauli_struct_set_bits`, `pauli_struct_get_bits` | Low-level struct helpers for Numba-compiled kernels. |
| `symplectic_inner_product_struct`, `commutes_struct` | Symplectic operations on struct tuples. |
| `commutation_matrix(collection)` | Generate a full commutation matrix for `PauliIntCollection`. |
| `toZX_large` | Parse strings, tuples, or binary rows into `PauliIntCollection`. |

All of these are re-exported at the package level, so the following works:

```python
from paulitools import PauliIntCollection, commutation_matrix
```

## 🔧 Technical Features

### Performance Optimizations
- **Numba JIT Compilation**: Most functions use `@njit()` for near-C performance
- **Bitwise Operations**: Efficient manipulation using integer bit operations
- **Vectorized Operations**: Matrix-based commutation analysis for large operator sets
- **Memory Efficiency**: Compact integer representation of Pauli operators

### Fast Conversion Paths
- **Binary Arrays by Default**: NumPy inputs are treated as `(Z|X)` binary bitplanes; values must be 0/1. A `-1/+1` array is automatically interpreted as eigen-Z data (`-1 → 1`, `+1 → 0`).
- **`fast_input_type` Shortcut**: Skip validation when the encoding is known. Use `fast_input_type="binary_string"` for pure Z|X strings or `fast_input_type="eigen_z"` for ±1 eigenvalue tables.
- **Numba-backed Packing**: Internal bit packing is compiled with Numba for low overhead bulk ingestion.

```python
from paulitools import toZX

# Stream large eigenvalue tables straight into ZX form
measurements = [-1, 1, -1, 1]
zx = toZX(measurements, fast_input_type="eigen_z")

# Pre-formatted binary strings can bypass validation too
binary_batches = ["0011", "1100"]
zx_fast = toZX(binary_batches, fast_input_type="binary_string")
```

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
from paulitools import (
  toZX, toString,
  commutes, commutation_matrix,
  toZX_extended, symplectic_inner_product_extended,
  PauliIntCollection,
)

# Legacy representation (≤31 qubits)
paulis = toZX(['XX', 'YY', 'ZZ'])
print("Encoded as:", paulis)

# Symplectic checks
p1 = toZX('XX')
p2 = toZX('ZI')
print("XX and ZI commute:", bool(commutes(p1, p2)))

# Large operators (supports arbitrary qubits)
large = toZX_extended('X' * 70)
print("Large operator type:", type(large))
print("Symplectic overlap:", symplectic_inner_product_extended(large, large))

# Commutation matrix for a PauliIntCollection
collection = PauliIntCollection(3, toZX_extended(['XXI', 'YYI', 'ZZI'], force_large=True).paulis)
print(commutation_matrix(collection))
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
 