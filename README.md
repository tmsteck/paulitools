# PauliTools

PauliTools is a Python library for fast manipulation and analysis of Pauli
strings. It uses binary symplectic representations, Numba-compiled kernels,
and a large-operator backend for stabilizer, commutation, and expectation-value
workflows.

## Installation

From a checkout of this repository:

```bash
python -m pip install -e .
```

The project metadata installs NumPy, Numba, `galois`, and Joblib. Joblib is
used by the parallel expectation helpers, while `galois` is required by
`getCentralizer`.

Run the test suite from the repository root with:

```bash
python -m pytest
```

## Quick start

```python
from paulitools import (
    append_pauli_data,
    centralizer,
    load_pauli_data,
    row_reduce,
    save_pauli_data,
    toString,
    toZX,
)

# Parse Pauli strings into the packed legacy representation.
generators = toZX(["XX", "YY", "ZZ"])

# Run compiled symplectic/group-theory operations.
reduced = row_reduce(generators)
center = centralizer(generators)
print(toString(reduced))
print(toString(center))

# Save and append batches without rewriting the existing records.
save_pauli_data("stabilizers.ptstore", generators)
append_pauli_data("stabilizers.ptstore", toZX(["XI", "IZ"]))
restored = load_pauli_data("stabilizers.ptstore")
```

All of the main public entry points are re-exported from `paulitools`, so
consumers do not need to import the implementation modules directly.

## Representations

PauliTools has two interoperable representations:

| Representation | Use | Conversion |
| --- | --- | --- |
| Legacy packed `numpy.ndarray` | Fast Numba kernels and systems up to 31 qubits | `toZX(...)` |
| `PauliInt` / `PauliIntCollection` | Operators larger than 31 qubits and chunked storage | `toZX_extended(...)` or `toZX_large(...)` |

The legacy array stores the qubit count in element `0`. Each following `int64`
packs the phase sign, Z bits, and X bits. The binary input convention is
`Z|X`: a row of length `2 * n_qubits` contains all Z bits followed by all X
bits.

`toZX_extended` selects the legacy representation for systems up to
`MAX_STANDARD_QUBITS` (31) and returns a `PauliIntCollection` for larger
systems. Pass `force_large=True` to use the large backend for a small system
when backend-independent code or testing requires it.

```python
from paulitools import (
    PauliIntCollection,
    commutation_matrix,
    symplectic_inner_product_extended,
    toString_extended,
    toZX_extended,
)

large = toZX_extended("X" * 64)
assert isinstance(large, PauliIntCollection)
print(toString_extended(large))

x0 = toZX_extended("X" + "I" * 63)
z0 = toZX_extended("Z" + "I" * 63)
print(symplectic_inner_product_extended(x0, z0))  # 1
print(commutation_matrix(toZX_extended(["XX", "YY"], force_large=True)))
```

## Mutable `ZXArray` wrapper

`ZXArray` is the ergonomic, mutable boundary for both backends. It supports
construction from strings, raw packed arrays, binary bit matrices, and
`PauliIntCollection` objects. Use `.legacy_array()` explicitly when passing a
legacy-backed value to a Numba kernel.

```python
from paulitools import ZXArray, toZXArray

paulis = toZXArray(["XX", "-ZI"])
paulis.set_bits(0, 0, z_bit=1, x_bit=0)
paulis.set_sign(1, 1)
paulis.append("YY")

print(paulis.to_strings())       # ['+ZX', '-ZI', '+YY']
print(paulis.binary())           # Z|X bit matrix
print(paulis.legacy_array())     # packed array for row_reduce/centralizer

# The same wrapper can hold arbitrarily large operators.
large = ZXArray.identities(64, count=2)
large.set_bits(0, 0, x_bit=1)
print(large.backend, large.to_strings()[0])
```

Useful constructors and accessors include:

- `ZXArray.empty(n_qubits)` and `ZXArray.identities(n_qubits, count=...)`
- `ZXArray.from_bits(z_bits, x_bits, signs=...)`
- `toZXArray(input_data, force_large=False)`
- `.z_bits()`, `.x_bits()`, `.binary()`, `.signs()`, and `.to_strings()`
- `.get_bits()`, `.set_bits()`, `.set_sign()`, `.set_pauli()`, `.append()`, and `.extend()`
- `.commutes(other)` and `.symplectic_inner_product(other)` for single-Pauli wrappers

## Core operations

The most commonly used conversion and symplectic functions are:

| Function | Purpose |
| --- | --- |
| `toZX(input_data, fast_input_type=None)` | Parse Pauli strings, tuples, binary strings, or binary arrays into legacy packed form. |
| `toString(integer_rep)` | Convert a legacy packed array to signed Pauli strings. |
| `symplectic_inner_product(a, b, k=None)` | Compute the legacy symplectic inner product. |
| `commutes(a, b, length=None)` | Test whether two legacy operators commute. |
| `bsip_array(...)` / `commute_array_fast(...)` | Build dense pairwise symplectic or commutation matrices. |
| `right_pad(...)` / `left_pad(...)` / `append(...)` | Resize or combine packed legacy forms. |

`toZX` accepts ordinary Pauli strings such as `"-XYZI"`, lists of strings,
indexed tuples such as `[("X", 0), ("Z", 2)]`, and `Z|X` binary arrays. For
validated high-throughput inputs, `fast_input_type` accepts:

- `"binary_string"` for `Z|X` strings containing only `0` and `1`;
- `"eigen_z"` for arrays of `-1/+1` eigenvalues, where `-1` sets a Z bit.

The fast modes bypass input validation, so use them only when the encoding is
known to be correct.

## Group and stabilizer operations

Functions in the group-theory workflow operate on packed legacy arrays:

| Function | Purpose |
| --- | --- |
| `row_reduce(paulis)` / `generators(paulis)` | Find an independent GF(2) basis. |
| `row_space(paulis)` | Enumerate the row space. |
| `null_space(matrix)` | Compute a GF(2) null space. |
| `inner_product(paulis)` | Compute the pairwise symplectic inner-product matrix. |
| `radical(paulis, reduced=False)` | Find the center/radical of a Pauli set. |
| `centralizer(paulis, reduced=False)` | Find operators commuting with a Pauli set. |
| `differences(paulis, paulis2=None)` | Compute within-set or pairwise relative differences. |
| `ingroup(candidates, pauli_set, reduced=False)` | Test membership in the generated span. |

For measurement and purity workflows, the package also exports
`filtered_purity`, `filtered_purity_reference`, `get_purity`,
`get_pauli_obs`, `get_pauli_pauli_obs`, `Pauli_expectation`, and
`getCentralizer`.

## Persistent storage

`save_pauli_data` writes a checksum-validated, log-structured archive. The
archive supports both legacy arrays and large `PauliIntCollection` batches;
`ZXArray` values are accepted as well. Records contain NumPy-compatible
payloads and preserve the representation selected for the file.

```python
from paulitools import iter_pauli_records, load_pauli_data, save_pauli_data

save_pauli_data(
    "measurements.ptstore",
    toZXArray(["XX", "YY"]),
    user_metadata={"experiment": 42},
)

data, metadata = load_pauli_data(
    "measurements.ptstore",
    include_metadata=True,
)

for batch in iter_pauli_records("measurements.ptstore"):
    print(batch)
```

Appending requires matching representation and qubit dimensions. Use a new
archive when those dimensions change.

## Performance notes

- Most hot-path conversions, symplectic checks, and packed group operations are
  Numba-compiled.
- The first call to a compiled function may incur JIT compilation overhead.
- `PAULITOOLS_NUMBA_CACHE=1` enables Numba disk caching; caching is disabled by
  default in editable/development contexts.
- Dense commutation matrices require quadratic storage in the number of
  operators, while packed/chunked operator storage scales with the number of
  operators and qubit chunks.

## Demos

The Pauli branching demo can be run from the repository root:

```bash
python -m demos.pauli_branching_demo
```

See [`demos/README.md`](demos/README.md) for command-line arguments and the
demo workflow.

## Applications

PauliTools is intended for quantum error correction, stabilizer-code analysis,
quantum simulation, Pauli Hamiltonian workflows, and other research code that
needs repeated Pauli-string operations.
