# Bell sampling numerical API

The implementation lives in `src/paulitools/bell_sampling/`. All public names
below are available from both `paulitools` and `paulitools.bell_sampling`.
The historical raw-array kernels remain available with their existing ABI.
Runtime dependencies remain NumPy and Numba; Galois is a verification extra.

## Scope and inputs

This package computes with Bell outcome **labels** and GF(2) support spaces.
It does not create physical Bell samples, certify independence, implement an
adaptive verifier, or choose stopping rules and statistical thresholds. Those
contracts remain with the consuming experiment package.

Inputs accept `ZXArray`, `Pauli`, Pauli labels, or binary `Z|X` matrices.
Collections have one explicit qubit width. Qubit zero is the leftmost label
character and bit zero of each Z/X plane. Numeric object inputs are binary,
not packed integers: use `ZXArray.from_raw(raw)` for the legacy
`[n_qubits, packed, ...]` ABI. Width mismatches raise; pad explicitly when
needed. Empty collections use `ZXArray.empty(n_qubits)`; a bare `[]` is also
accepted for generators/candidates when an existing object supplies the width.
Zero-qubit inputs are supported. Packed storage is used through 31 qubits;
larger objects use 64-bit chunks.

Bell labels and support operations deliberately ignore **all global phases**.
They do not replace exact operator multiplication (`@`) or signed stabilizer
reduction (`.stabilizer_basis()`). Support outputs have phase zero. For example,
`bell_differences("X", "Y")` is Z, while `Pauli("X") @ Pauli("Y")` is +iZ.

## Paired differences

| Function | Contract |
| --- | --- |
| `bell_differences(left, right, *, parallel=False)` | Pair corresponding rows by support XOR. Equal widths and counts are required; no broadcasting. |
| `cyclic_bell_differences(samples, *, parallel=False)` | XOR each row with its successor, including last with first. Diagnostic with dependent outputs. |

Both return a new `ZXArray`, preserve empty input widths, and discard phases.
Use two independently obtained streams when the estimator requires independent
Bell differences. The API cannot establish independence from the values alone.
Cyclic pairing reuses observations and must not silently replace that protocol.

## Masks and scores

Write `Y(s) = sum_q z_q x_q mod 2` and
`<s,g> = z_s . x_g + x_s . z_g mod 2`.

| Function | Result |
| --- | --- |
| `y_parities(paulis, *, parallel=False)` | `uint8` vector of Y parities. |
| `commuting_mask(samples, generators, *, parallel=False)` | Boolean mask for `<s,g> == 0` for every generator. Generators need not commute mutually. |
| `bell_filter_mask(samples, generators, *, parallel=False)` | Boolean mask for `<s,g> == Y(g)` for every generator. Requires mutually commuting generators. |
| `bell_purity(samples)` | Mean of `(-1)**Y(s)` over all samples. |
| `bell_filtered_purity(samples, generators, *, parallel=False)` | Sum of `mask[s] * (-1)**Y(s)` divided by the **original** sample count. |

The two masks differ for odd-Y generators. On an isotropic (commuting)
support space Y is linear, making the shifted Bell mask independent of the
chosen generating basis. Noncommuting Bell generators raise `ValueError`.
Overall generator signs/phases are ignored, so this is not projection onto
the +1 eigenspace of signed operators.

Empty generators select every sample. Empty samples give empty masks but
undefined mean scores, which raise `ValueError`. A nonempty pool with no
survivors has filtered score zero. Scores are numerical estimators and are
not clipped to a physical interval. Statistical validity depends on how
the caller acquired and selected the samples.

## Repeated proposals on one pool

`BellSamplePool(samples)` copies sample chunk buffers and caches Y parities.
Its public properties are `n_qubits`, `n_samples`, and `purity`.
`pool.filter(generators, *, parallel=False)` returns a `BellFilterState`.

The state exposes `score`, `mask`, and `generators`. The last two return copies.
`state.extend(additions, *, parallel=False)` returns a **new** state, validates
commutation with current generators, and evaluates only the added constraints
on surviving rows. The old state is unchanged, so a rejected proposal cannot
pollute later scores. The original denominator is retained. Replacing/removing
constraints requires `pool.filter(...)` again.

```python
from paulitools import BellSamplePool, bell_filtered_purity

samples = ["II", "XI", "YI", "ZI", "IZ", "YY"]
pool = BellSamplePool(samples)
current = pool.filter([])
proposal = current.extend("ZI")
assert proposal.score == bell_filtered_purity(samples, "ZI")
assert current.score == pool.purity
# The caller's verifier decides whether to retain proposal as current.
```

Pool/state objects are numerical snapshots. They neither enforce fresh-sample
rules nor provide adaptive-query guarantees. Keep those policies in Robels.

## Prepared support spaces

`SupportBasis(data, *, n_qubits=None)` owns a phase-free canonical basis.
It prepares left-to-right pivots in binary `Z|X` order once. Do not combine
its pivot assumptions with the descending packed-bit pivots of raw
`row_reduce`. Public accessors cannot mutate the prepared data.

| Member | Contract |
| --- | --- |
| `.n_qubits`, `.rank` | Width and GF(2) dimension; the span has `2**rank` elements. |
| `.basis`, `.to_zxarray()` | Independent positive-phase canonical generators. |
| `.binary()` | Independent `uint8` matrix of shape `(rank, 2*n_qubits)`. |
| `.contains(candidates)` | Boolean vector of phase-free membership results. |
| `.coset_reduce(candidates)` | Canonical positive representatives modulo this span. Members reduce to identity. |
| `.linear_combinations(coefficients)` | Batch XOR of canonical generators. Binary shape `(count, rank)` or `(rank,)` for one output. |
| `.union(other)` | A new basis of the linear span sum, not the ordinary set union. |
| `.intersection(other)` | A new basis of the common subspace. |
| `.quotient_dimension(other)` | `dim((self + other) / other)`, valid even without containment. |
| `.sample(count, *, rng, exclude_identity=False, exclude_span=None)` | Uniform support draws with replacement, preparing the exclusion for this call. |
| `.sampler(*, exclude_identity=False, exclude_span=None)` | Reusable `SupportSampler` with fixed exclusions. |

Methods taking `other` accept another `SupportBasis` or ordinary Pauli input.
`contains`, `coset_reduce`, and filter functions accept Pauli collections;
use `prepared.basis` when passing a prepared space into those functions.
No commutation or signed-stabilizer assumption is imposed by `SupportBasis`.

`SupportSampler.sample(count, *, rng)` requires an explicit
`numpy.random.Generator`; it owns no RNG. `exclude_span=S` draws uniformly
from `R \\ S`, even when S is not contained in R. Identity is already excluded
by any excluded span. With neither exclusion, identity is allowed. An empty
allowed set accepts count zero and raises for a positive count.

Preparation decomposes coefficient space into the intersection and a
complement. Sampling rejects only an all-zero complement, with acceptance
probability at least one half. It never relies on repeated full-span rejection
when almost all points are excluded. Reuse the sampler while R and S remain
fixed; rebuild after changing either span.

```python
import numpy as np
from paulitools import SupportBasis

R = SupportBasis(["ZI", "IZ"])
S = SupportBasis("ZI")
sampler = R.sampler(exclude_span=S)
draws = sampler.sample(20, rng=np.random.default_rng(7))
assert R.contains(draws).all()
assert not S.contains(draws).any()

# Regression: packed and binary pivot conventions must not be mixed.
assert SupportBasis(["ZZ", "YX"]).coset_reduce("ZZ").to_strings() == ["+II"]
```

For a set B, `B.center()` returns the radical `span(B) intersect span(B)^perp`;
`B.centralizer()` returns the full ambient commuting space. These methods are
on `ZXArray`. The historical raw `centralizer` function remains a compatibility
alias for `center`; new consumers should choose the explicit mathematical API.

## Symplectic Walsh-Hadamard transform

`symplectic_fwht(values, *, inverse=False)` accepts a finite numeric 1D vector
of length `4**n` and returns `float64` or `complex128`, without mutating input.
Indices are `z_integer | (x_integer << n)`; one-qubit order is I, Z, X, Y.
The forward result at i is `sum_j (-1)**<i,j> * values[j]`. The inverse applies
the same transform divided by `4**n`. Length one supports zero qubits.
Integer conversion may round beyond float64's exact integer range.

The transform performs compiled `O(n * 4**n)` work using `O(4**n)` output
storage. It still requires full Pauli enumeration. Bell-specific coefficient
powers, Y signs, probability normalization, and tolerances belong to Robels.
Arbitrary input ordering must be scattered to the canonical indices first.
Nonfinite input and nonfinite arithmetic results raise errors.

## Compilation and ownership

| Source file | Responsibility |
| --- | --- |
| `differences.py` | Packed support XOR of paired or cyclic labels. |
| `filters.py` | Chunk parity, streaming masks, signed totals, immutable pool/state snapshots. |
| `subspace.py` | Prepared canonical bases, membership/cosets, combinations, intersections, uniform proposals. |
| `transforms.py` | Generic symplectic transform. |
| `_inputs.py` | Shared object-boundary validation. |

Repeated numerical loops run in Numba nopython kernels. Chunk XOR/filter
kernels release the GIL; sample rows in differences, masks, and parity have
optional `parallel=True` paths. Serial is the default. Basis preparation and
GF(2) elimination remain sequential. Python owns validation, RNG draws, and
object construction. Large `ZXArray` output construction still creates
per-row objects; kernel efficiency does not imply zero wrapper overhead.
Prepare bases/samplers/pools once and batch candidates when possible.
Private kernel signatures are not a public ABI.

Tests in `testing/bell_*_test.py` use explicit characters, dense operators,
brute-force GF(2) spans, dense transforms, and optional Galois comparisons.
They cover serial/parallel agreement, incremental/full agreement, phases,
zero width, empty inputs, and 31/32/64/65/129-qubit boundaries. See
[Robels migration](ROBELS_MIGRATION.md) for consumer changes and validation.
