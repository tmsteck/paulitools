# Robels migration to the Bell sampling API

This guide describes the consumer changes for the September 2026 PauliTools
implementation. Robels was inspected at `/Users/thomassteckmann/Robels`;
its `paulitools-local` link points to this package. This change does **not**
edit Robels or run production numerical studies. Apply the migration in
bounded steps and retain the existing dense/reference paths as test oracles.

Read [BELL_SAMPLING.md](BELL_SAMPLING.md) for signatures and mathematical
contracts. All new functions/classes are top-level PauliTools exports.

## 1. Repair parsing and the historical archive boundary

In `src/RoBell/pauli.py`, route string labels and string lists in `as_zx` and
`strings_to_zx` through `toZXArray`, retaining `_validate_zx_header` to check an
expected width. For Robels, `n_qubits` currently means validation; do not
accidentally turn it into implicit padding. Preserve the explicit 1D packed
array branch using `ZXArray.from_raw`. Keep dataset-specific tuple/readout
decoding in Robels rather than treating every numeric input as packed data.

This fixes two exposed cases: legacy parsing reads `+iX` as two-qubit IX,
and the legacy parser cannot represent 32-qubit strings. New object parsing
preserves one-qubit +iX and supports larger widths. Existing signed lowercase
labels beginning with i require the documented uppercase migration (`-IX`
versus `-iX`). In `as_legacy_zx`, preserve conversion errors: imaginary phases
are another reason export can fail, so do not relabel every `ValueError` as
"too large".

In `src/RoBell/harvard.py`, replace the archive import with:

```python
from paulitools.storage import load_legacy_payload as _load_raw_pauli_data
```

Update the matching loader/decoder docstrings. Keep
`decode_harvard_physical_samples` unchanged. The archive's header 64
describes 64 readouts for 32 physical qubits, not a valid legacy 64-qubit Pauli.
Its last readout uses bit zero, so it must not be interpreted as a Pauli sign.
The new reader returns the original int64 header/payload and validates the
archive structure, record counts, dtypes, and checksums. It intentionally does
not interpret packed Pauli semantics. It is a reader for historical payloads,
not a new permissive writer; ordinary `load_pauli_data` and `save_pauli_data`
retain strict width/high-bit validation. Use the ordinary loader with
`as_zxarray=True` for real Pauli archives.

The committed Harvard fixture was checked through the new reader and existing
Robels decoder: 5,000 shots, 32 physical qubits, last-X-column count 2,441,
block-0 parity/purity 0.3388. The unmodified consumer still needs the import
change; adding the new reader alone does not redirect Robels automatically.

## 2. Replace local algebra with prepared package operations

| Robels function or call site | PauliTools replacement |
| --- | --- |
| `pauli.row_reduce_zx` | `as_zx(data).support_basis()`; retain a `SupportBasis` for repeated queries. |
| `pauli.radical_zx`, deprecated `centralizer_zx` | `as_zx(data).center()`; **not** `.centralizer()`. |
| `pauli.commuting_basis_zx` | `as_zx(data).centralizer()` for the full ambient space. |
| `pauli.in_span` | Prepared `SupportBasis(S).contains(candidates)`. |
| `pauli.group_contains` | Retain the one-row check; return `bool(prepared.contains(single)[0])`. |
| `pauli.coset_reduce` | Prepared `SupportBasis(S).coset_reduce(candidates)`. |
| `pauli.quotient_dim` | `SupportBasis(A).quotient_dimension(S)`. |
| `pauli.uniform_group_element` | `prepared.sample(1, rng=rng, exclude_identity=exclude_identity)`, or a reusable sampler. Preserve the wrapper's default `exclude_identity=True`; the new API defaults to False. |
| `pauli.y_parities_zx` | `y_parities`. |
| `pauli.filtered_purity_score` | `bell_filtered_purity(samples, generators)`; note samples-first order. |
| `pauli.symplectic_matrix` | For binary-matrix inputs, `toZXArray(left_rows).symplectic_matrix(toZXArray(right_rows))`; 1 means anticommutes. |
| `algorithm.post_select_commuting` | Compute `mask = commuting_mask(samples, generators)` and return `samples[mask]`. |
| `bell_processing.postselect_commuting` | Return `(samples[mask], mask)` using the same mask API. |
| `exact_validation.centralizer_mask` | `commuting_mask(samples, generators)`. |
| `algorithm.merge_generators` | `SupportBasis(current).union(additions).basis`. |
| `first_acceptance._truth_overlap` | Compute intersection rank with `SupportBasis(estimated).intersection(truth).rank`; retain the tuple `(intersection_rank, estimated_rank - intersection_rank)` and its event definition. |
| `harvard.empirical_bell_purity` | `bell_purity(samples)` after physical-readout decoding. |

Keep a prepared basis across repeated membership/coset calls; reconstructing it
for every row defeats the preparation. Replace row-level Python XOR and
commutation loops with batch calls. Small compatibility wrappers may remain
while callers migrate, but core arithmetic should have one implementation.

The old `coset_reduce` mixes descending packed pivots with ascending binary
pivots. For the commuting basis `{ZZ, YX}`, it can map the member ZZ to XY.
The replacement maps it to II. Add this regression to Robels alongside tests
that equal cosets reduce identically. Canonical basis ordering may change;
compare spans instead of literal generator rows. Seeded proposal trajectories
can also change despite identical distributions, so record the dependency
revision and avoid resuming old trajectories under a new basis/RNG mapping.

The manuscript's radical is `B intersect C(B)`. Its clean/intermediate radical
steps must continue to use `.center()`. Full `.centralizer()` is for ambient
commutation constraints and generally has a different dimension.

## 3. Keep Bell samples and Bell differences distinct

Change independent-pair paths in `pauli.difference_samples` and
`BellDifferences.from_sample_pairs` to `bell_differences(left, right)` after
validating stream provenance. Both streams must have equal widths/counts.
Route explicitly cyclic diagnostics to `cyclic_bell_differences(samples)`.
Do not infer the choice from a missing second stream in new experiment code.

Retain `BellSamples`, `BellDifferences`, `BellDataset`, independent-copy
accounting, stream identifiers, and experiment metadata in Robels. The library
returns labels and cannot certify their sampling distribution. Passing Bell
difference labels into a raw-Bell purity estimator is not made valid by their
shared `ZXArray` representation.

## 4. Reuse verifier pools and implement the intended proposal distribution

For a fixed pool in `algorithm.AdaptiveVerifier`, prepare a `BellSamplePool`
once. Store a `BellFilterState` corresponding to the accepted current span.
Score a proposed extension using `state.extend(additions).score`. Retain the
new state only when the Robels verifier accepts it. Rejected proposals must
leave the current state unchanged. If the pool changes, create a new pool and
state. If a proposal removes/replaces constraints, use `pool.filter(...)`.

The shifted Bell filter is `<s,g> == Y(g)` and uses the original sample count
as denominator. Ordinary postselection is `<s,g> == 0`. Do not interchange
the APIs or divide a Bell filtered score by the survivor count. Add odd-Y,
empty-generator, rejected-proposal, and incremental-versus-full tests.

The manuscript's intermediate proposal step draws uniformly from `R \\ S`.
Current proposal loops draw from `R \\ {I}` then skip elements in S, consuming
an L iteration. These differ when S is nontrivial. Prepare:

```python
R = SupportBasis(radical_generators)
S = SupportBasis(current_generators)
sampler = R.sampler(exclude_span=S)
# In each proposal iteration, preserving Robels' stopping and acceptance rules:
candidate = sampler.sample(1, rng=rng)
```

Rebuild the sampler if R or S changes. If `R.quotient_dimension(S) == 0`, no
candidate exists; Robels must terminate or report that branch explicitly.
This provides L draws from the intended set when it is nonempty. The
first-acceptance protocol with S initially empty is not evidence that the
nonempty-S path was already correct. If reproducing a historical policy is
necessary, label it separately rather than silently changing its meaning.

Keep thresholds, confidence budgets, clean/intermediate branching, bootstrap
settings, stopping rules, trial schedules, and theorem/heuristic labels in
Robels. Faster cached numerical scores do not establish an adaptive verifier
guarantee.

## 5. Replace dense character matrices in exact validation

`exact_validation.enumerate_pauli_zx` uses packed values `arange(4**n) << 1`,
matching the new transform index `z | (x << n)`. For that full canonical
enumeration, replace the dense sign-matrix product with:

```python
# a contains decomposition.coefficients in canonical enumeration order.
parity = y_parities(paulis).astype(np.int64)  # avoid unsigned subtraction
raw_bell = symplectic_fwht((1 - 2 * parity) * np.square(a))
raw_differences = (4.0 ** n) * symplectic_fwht(np.square(np.square(a)))
```

Retain the existing coefficient convention, clipping tolerance, negative-mass
checks, normalization, and output label mapping in Robels. For custom
decomposition order or subsets, scatter coefficients into a full canonical
vector before the transform and explicitly select the required output labels;
do not blindly apply the formula to an arbitrary row ordering. Duplicate
labels need a defined coefficient-combination policy. Dense state/operator
construction and the physics of a²/a⁴ remain independent validation code.

This removes the `4**n` by `4**n` character matrix, with transform work
`O(n*4**n)` and storage `O(4**n)`. Full coefficient enumeration is still
exponential, and end-to-end speedup has not been benchmarked here.

## Acceptance checks and rollout

1. Make the parser/archive changes and add phase, 32/65-qubit, and Harvard
   fixture regressions. Preserve all existing readout expectations.
2. Migrate algebra wrappers and compare spaces/results against the existing
   reference paths, including the ZZ/YX coset regression and center versus
   ambient dimensions. Preserve signed operations where phases matter.
3. Integrate pool/state and prepared sampler reuse, with rejected-proposal
   isolation and nonempty excluded-span tests. Audit RNG and L accounting.
4. Replace dense transforms only after small-system probability arrays agree
   with dense formulas, including Y signs, canonical ordering, and inverse
   normalization. Keep those dense formulas in tests.
5. Update Robels' `AGENTS.md`, dependency instructions, and benchmark metadata
   to name these package APIs. Validate in its maintained `robels-modern`
   environment before starting new production studies.

The bounded existing consumer checks are:

```bash
python -m pytest -q tests/test_pauli_contracts.py tests/test_ws1_primitives.py \
  tests/test_bell_processing.py tests/test_verifier.py \
  tests/test_difference_streaming.py tests/test_harvard_physical.py
```

These tests alone do not establish the new proposal policy or adaptive
statistical guarantees; the new regressions above are part of the migration.
Run PauliTools' `testing/bell_*_test.py` and `testing/legacy_payload_test.py`
as the dependency contract. Galois remains an optional, independent test
oracle and must not re-enter Robels' or PauliTools' production hot paths.

The six existing consumer files above passed **48 tests** in `robels-modern`
during a read-only rehearsal with `_load_raw_pauli_data` rebound in memory to
the new payload reader. No Robels source files were changed. This checks the
archive migration and current compatibility; it does not test the remaining
wrapper, incremental-verifier, proposal-policy, or transform migrations.
