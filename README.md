# PauliTools

PauliTools provides width-aware Pauli objects backed by packed integers and
Numba-compiled batch kernels. It supports exact discrete phases, commutation,
GF(2) support spaces, signed stabilizers, and checksummed storage.

Bell-label numerics live in `src/paulitools/bell_sampling/`: paired differences,
streaming filters, reusable sample pools, prepared subspaces and samplers, and
a symplectic Walsh-Hadamard transform. See the [Bell sampling API](docs/BELL_SAMPLING.md)
and [Robels migration guide](docs/ROBELS_MIGRATION.md) for the complete contracts.

## Installation

Python 3.9 or newer is required. The runtime dependencies are NumPy and Numba.
Galois is an optional verification dependency; Joblib is no longer needed.

```bash
python -m pip install -e '.[test]'
python -m pytest
# Optional independent finite-field comparisons:
python -m pip install -e '.[verification]'
```

Some older comparison tests additionally use the separate `ptgalois` package
and skip when it is unavailable. The main correctness tests use explicit
GF(2) calculations and independent operator matrices.

## Bell sampling

```python
import numpy as np
from paulitools import (
    ZXArray, SupportBasis, BellSamplePool, bell_differences, commuting_mask,
)

# Small label examples; real experiments must establish stream independence.
left = ZXArray.from_input(["II", "XI", "ZZ"])
right = ZXArray.from_input(["ZI", "II", "IZ"])
differences = bell_differences(left, right)
radical = differences.center()  # center within the observed support span

pool = BellSamplePool(left)     # copies chunks and caches Y parities once
current = pool.filter([])
proposal = current.extend("ZI") # new state; caller decides whether to accept
ordinary_mask = commuting_mask(left, "ZI")

R = SupportBasis(["ZI", "IZ"])
sampler = R.sampler(exclude_span=SupportBasis("ZI"))
draws = sampler.sample(10, rng=np.random.default_rng(7))
assert R.contains(draws).all()
```

Bell filtering uses `<sample, generator> == Y(generator)`; ordinary commutation
filtering uses zero on the right. Filtered purity divides by the original pool
size. Both ignore global phases. Experimental provenance, adaptive-verifier
guarantees, and acceptance policies stay in the consumer. Prepared samplers
reuse the exclusion decomposition for repeated draws from `R \\ S`.

## Objects carry width and phase

```python
from paulitools import Pauli, ZXArray, toZXArray

x = Pauli("X")
y = Pauli("Y")
assert str(x @ y) == "+iZ"
assert str(y @ x) == "-iZ"
assert x.n_qubits == 1
assert not x.commutes(y)
assert Pauli("-X").equiv(x)       # deliberately ignores phase
assert Pauli("-X") != x          # equality includes phase

paulis = toZXArray(["XX", "-iZI"])
selected = paulis[1]             # independent Pauli, including width and phase
selected.phase = 2
paulis[1] = selected             # write back explicitly
assert paulis.to_strings() == ["+XX", "-ZI"]

# Identity allocation and bit access work with either storage backend.
large = ZXArray.identities(65, count=2)
large.set_bits(0, 64, x_bit=1)
assert large.n_qubits == 65
```

Phase `q` means `i**q` multiplying a tensor product of ordinary Hermitian
I/X/Y/Z matrices: 0, 1, 2, 3 denote +1, +i, -1, -i. Labels accept `+`, `-`,
`+i`, and `-i` prefixes. Use uppercase Pauli letters to avoid ambiguity:
`"-iX"` means minus-i times X, while `"-IX"` means minus I tensor X.
Bare `"iX"` is accepted; lowercase `"ix"` retains the meaning IX.
When migrating old signed lowercase labels, uppercase the Pauli letters first:
`toZXArray("-ix")` now denotes one-qubit -iX, whereas the old wrapper and the
legacy `toZX("-ix")` interpret it as two-qubit -IX. Use `"-IX"` for the latter.

Qubit zero is the **leftmost character** of a label. Numeric matrices use
`Z|X` column order. Collections share one width and reject implicit resizing.
Use `Pauli("X", n_qubits=3)`, `toZXArray(["X", "YY"], n_qubits=3)`, or
`.pad(3, side="right")` to request identity padding explicitly. Padding on the
left is also supported. Empty collections require a declared width through
`ZXArray.empty(n_qubits)`; zero-qubit operators are supported.

Numeric object inputs are 0/1 bits by default. Eigenvalues require an explicit
encoding, so an all-ones row has an unambiguous meaning:

```python
import numpy as np

assert str(Pauli(np.array([1, 1]))) == "+Y"
assert str(Pauli(np.array([1, 1]), encoding="eigenvalues")) == "+I"
a = ZXArray.from_bits([[0, 1]], [[1, 0]], phases=[3])
b = ZXArray.from_eigenvalues([[1, -1]], [[-1, 1]], phases=[3])
assert a == b
```

`.binary()`, `.z_bits()`, and `.x_bits()` return support bits; `.phases()`
returns the corresponding phase exponents. `.signs()` is limited to real
phases. Indexing and slicing return copies; assignment, `.set_bits()`,
`.set_phase()`, `.append()`, and `.extend()` mutate a collection.

## Batch algebra and compiled interoperability

`@` performs an ordered Pauli product row by row, broadcasting a single
operator on either side. Widths must match. `.symplectic_matrix(other=None)`
returns a rectangular uint8 matrix with 1 meaning **anticommutes**;
`.commutation_matrix(other=None)` returns booleans with True meaning
**commutes**. Both accept `parallel=True` for compiled row parallelism.

Objects are Python interfaces to numeric storage. Convert once outside a
repeated compiled workflow:

```python
from paulitools import row_reduce, toZX

operators = toZXArray(["XX", "ZZ"])
raw = operators.legacy_array(copy=False)  # explicit alias, real phases only
basis = row_reduce(raw)                  # existing compiled API
assert np.array_equal(raw, toZX(["XX", "ZZ"]))

# General phase-aware kernel buffers, independent of the source object:
n_qubits, z_chunks, x_chunks, phases = operators.kernel_args()
```

The legacy ABI is an `int64` vector `[n_qubits, packed_operator, ...]`. Bit 0
of each packed value is the real sign; bits 1..n encode Z and bits n+1..2n
encode X. It supports at most 31 qubits. Larger objects use 64-bit chunks.
`force_large=True` selects chunks for smaller systems as well.

`ZXArray.from_raw(raw, copy=False)` explicitly wraps a packed array. Numeric
input to `Pauli(...)` or `toZXArray(...)` always denotes a bit/eigenvalue
representation, so packed input must use `from_raw`. Raw aliases are advanced
interfaces: callers must preserve widths, shapes, and valid packed values.

Legacy arrays and `PauliInt`/`PauliIntCollection` encode only real signs.
Exporting an object with an imaginary phase through `.legacy_array()`,
`.pauliint_collection()`, `.data`, or `.packed_values` raises an error.
`.kernel_args()` supports all phases and returns independent contiguous
chunk buffers; it is not a zero-copy view. Legacy export above 31 qubits also
raises instead of truncating.

## Center, centralizer, and stabilizers

For a support space S, the center is S intersect S-perp; the full ambient
centralizer is S-perp. The distinction matters even for one generator:

```python
s = toZXArray(["XI"])
assert len(s.center()) == 1
assert len(s.centralizer()) == 3
assert np.all(s.centralizer().commutation_matrix(s))

# XX * ZZ = -YY, so the signs of a stabilizer relation matter.
valid = toZXArray(["XX", "ZZ", "-YY"]).stabilizer_basis()
assert len(valid) == 2
# toZXArray(["XX", "ZZ", "YY"]).stabilizer_basis() raises: group contains -I.
```

| Interface | Meaning and return format |
| --- | --- |
| `ZXArray.support_basis()` | Independent GF(2) support basis; deliberately discards phases. |
| `ZXArray.center()` | Center support basis, as positive canonical Pauli representatives. |
| `ZXArray.centralizer()` | Full ambient commuting support basis, as positive representatives. |
| `ZXArray.stabilizer_basis()` | Phase-preserving basis; requires commuting Hermitian generators and rejects a -I relation. |
| `row_reduce(raw)` / `generators(raw)` | Historical packed support reduction; not signed stabilizer reduction. |
| `center(raw)` | Center basis as a binary Z|X matrix. |
| `ambient_centralizer(raw)` | Full ambient centralizer as a binary Z|X matrix. |
| `centralizer(raw)` | Compatibility alias for the historical **center** behavior. |
| `radical(raw, reduced=False)` | Coefficient null space of the Gram matrix of `row_reduce(raw)`; with `reduced=True`, coefficients refer to the supplied rows. |
| `stabilizer_reduce(raw)` | Phase-preserving packed stabilizer basis, up to 31 qubits. |
| `stabilizer_reduce_bits(z, x, phases)` | Phase-preserving array kernel for any width. |

The existing compiled `centralizer` name retains its historical semantics to
avoid silently changing downstream calculations. Migrate callers deliberately
to `center` or `ambient_centralizer`. Object group methods work with either
backend. Support-space methods do not certify a signed stabilizer state.

Other packed utilities include `null_space`, `inner_product`, `ingroup`,
`differences`, and `row_space`. `row_space` returns an independent binary Z|X
basis; it does not enumerate all combinations of generators.

## Legacy parsing and conventions

`toZX(...)` produces packed arrays; `toZX_extended(...)` chooses packed or
chunked storage. Their numeric `encoding="auto"` mode remains for compatibility:
if any entry is -1, the complete array is interpreted as +/-1 eigenvalues;
otherwise it is interpreted as 0/1 bits. Prefer `encoding="bits"` or
`encoding="eigenvalues"` in new code. `fast_input_type="binary_string"` and
`"eigen_z"` select the specialized Z|X-string and Z-only-eigenvalue paths.

Legacy string lists and scalar pair comparisons retain historical right
padding. Legacy parsers accept real signs only. Use the object constructors
for full phases and explicit width handling.

The raw `commutes` array interface retains its historical first-operator
behavior for collections. Use pairwise matrix functions for whole collections;
single-operator object methods reject multirow operands.

`bsip_array(raw)` and the historical large `commutation_matrix(collection)`
use 1=anticommutes. `commute_array_fast(raw)` uses 1=commutes. Their optional
`parallel=True` paths preserve those existing meanings. Object methods use
consistent named conventions described above.

## Bell-outcome estimators

`get_pauli_obs(P, probs)` evaluates
`sum_s p(s) (-1)^(sign(P) + Y(P) + symplectic(P,s))`, where Y is the number
of Y factors modulo two. `get_pauli_pauli_obs` adds Y(s) to that exponent.
These are explicit Bell-outcome conventions, not generic state-expectation
reconstruction. Outcome signs are ignored; an observable's minus sign negates
its estimate. Observables can be strings, packed arrays, or real-phase objects
of up to 31 qubits. Both functions accept `parallel=True`.

Probabilities must be finite, nonnegative, nonempty, and sum to one within
rounding tolerance. Normalize counts explicitly before passing them.
`Pauli_expectation(shots, P)` computes the first convention for one observable;
`shots` contains `(packed_outcome, probability)` rows. Preserve large packed
integers in nested rows or object arrays; float storage is rejected at its
consecutive-integer precision boundary.

`getCentralizer(counts)` retains the historical center-of-differences result
and now uses packed kernels without Galois. `filtered_purity`,
`filtered_purity_reference`, and `get_purity` document their Bell-parity
conventions in their docstrings.

## Persistent storage

```python
from paulitools import save_pauli_data, append_pauli_data, load_pauli_data

save_pauli_data("operators.ptstore", operators,
                user_metadata={"experiment": 42})
append_pauli_data("operators.ptstore", toZXArray(["-YY"]))
restored, metadata = load_pauli_data(
    "operators.ptstore", as_zxarray=True, include_metadata=True)
assert restored.to_strings() == ["+XX", "+ZZ", "-YY"]
```

Archives contain checksummed records and preserve their packed or chunked
backend. Appends require matching width and representation. Cooperating
writers acquire an operating-system file lock before initialization,
validation, and writing; this also protects simultaneous creation. Reads
should occur after writers finish: a read during an append is not a snapshot.
Checksums detect incomplete/corrupted records, but writes are not crash-atomic.

The existing version-1 format supports real signs only. Saving an imaginary
phase raises before modifying the destination. A versioned full-phase storage
format remains follow-up work. `iter_pauli_records(path)` streams raw batches;
default `load_pauli_data` still returns the original raw representation.

`load_legacy_payload(path, *, include_metadata=False)` is a separate reader
for historical opaque int64 payload archives, including Robels' physical
Harvard readouts. It preserves every bit and validates structure/checksums,
but returns data that may not be valid packed Paulis. Use the consumer's
dataset decoder next; normal Pauli readers and writers remain strict.

## Performance and development

- Keep arithmetic in batches; object construction and formatting run in Python.
- Packed/chunked symplectic kernels use bit operations and compiled popcounts.
  Numeric batch multiplication and matrix/estimator kernels release the GIL.
- Threaded matrix and estimator paths are opt-in. They may be slower for small
  workloads; crossover thresholds have not been established by benchmarks.
- Dense pairwise matrices require quadratic output storage. GF(2) reduction
  still has sequential pivot dependencies; it is not advertised as parallel.
- The first call includes JIT compilation. `PAULITOOLS_NUMBA_CACHE=1` enables
  disk caching; the default remains off for editable development.
- Run `NUMBA_BOUNDSCHECK=1 NUMBA_NUM_THREADS=2 python -m pytest` for the guarded
  validation configuration. No floating-point fast-math is used for algebra.

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for scope, validation,
and remaining work. The optional branching demo is documented in
[demos/README.md](demos/README.md).
