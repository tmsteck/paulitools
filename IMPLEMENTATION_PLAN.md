# PauliTools correctness and interface plan

This plan follows the September 2026 audit. Preserve the existing length-prefixed
`int64` ABI for compiled consumers and preserve unrelated working-tree changes.

## Contracts

- Character position zero denotes qubit zero; binary matrices are rows of `Z|X`.
- A public Pauli object always carries its width. Collections have one shared
  width. Object construction rejects unequal widths unless a target width is
  explicitly supplied; legacy string parsers retain documented right padding.
- Public object numeric inputs use explicit 0/1 bits by default. Eigenvalue
  inputs use a separate constructor/encoding. Legacy `encoding="auto"` remains
  available for compatibility, with its ambiguity documented.
- Full phase is `i**q` times a tensor product of ordinary Hermitian I/X/Y/Z,
  with `q` in 0..3. Legacy sign bits represent q=0 or q=2. Conversions that
  cannot represent imaginary phase raise rather than discard information.
- Indexing objects returns independent copies. Explicit raw-buffer access may
  alias storage and is the advanced boundary to compiled kernels.
- Support-space reduction deliberately forgets phase. Signed stabilizer
  reduction validates Hermiticity, commutation, and absence of a -I relation.
- Keep historical compiled `centralizer` behavior for compatibility and name
  it explicitly `center`; add `ambient_centralizer` for the full commuting
  space. The new object `.centralizer()` uses the full mathematical meaning.
- Existing low-level symplectic matrices keep 1=anticommutes. New object
  `commutation_matrix()` returns booleans with True=commutes.

## Implementation sequence

1. **Representation correctness.** Width guards before shifts; correct signed
   padding; consistent large/binary/lowercase parsing; high-bit validation;
   repaired scalar dispatch and compiled commutation matrices.
2. **Usable objects and phases.** Extend ZXArray, add single-Pauli access,
   indexing/assignment, explicit bit/eigenvalue construction, padding, exact
   multiplication/equality, and phase-safe conversion boundaries.
3. **Mathematical contracts.** Explicit center and ambient centralizer; stable
   coefficient-returning radical; signed stabilizer reduction with independent
   matrix-algebra tests and object-returning group methods.
4. **Utilities and dependencies.** Repair and document weighted Bell-character
   estimators; remove runtime Galois and Joblib dependence; handle empty filters;
   make reference dependencies optional.
5. **Integration and verification.** End-to-end object/storage behavior,
   regression tests, exhaustive small cases, randomized chunk boundaries,
   nopython compilation and bounds-check runs; README and AGENTS updates.

## Deliberate compatibility limits

- Existing `.ptstore` formats represent real signs only. Until a versioned
  full-phase format is designed, saving an imaginary-phase object must fail
  clearly. Existing real-phase archives stay compatible.
- The historical `centralizer` name cannot silently change for downstream
  compiled callers. A future major release can retire that alias after callers
  migrate to `center` or `ambient_centralizer`.
- Arbitrary Hamiltonian coefficients are outside the discrete Pauli group.
- Full-phase label parsing reserves +i/-i prefixes. Existing signed lowercase
  labels beginning with i should be uppercased before migration; for example,
  old -ix means -IX, while the new object parser reads it as -iX.

## Implemented performance choices

- Packed and chunked symplectic products use unsigned bit operations and LLVM
  `ctpop`; compiled scalar and batch signatures were inspected.
- Matrix and weighted-estimator row loops have separate optional parallel
  kernels. Serial paths remain the default. No fast-math is used.
- Full-phase multiplication packs a whole batch into contiguous chunk buffers
  and uses one compiled call. Object indexing does not allocate a full index
  range for a scalar selection. Explicit kernel buffers can be bound once for
  repeated compiled workflows.
- A local warm-kernel spot check (Python 3.10.9, NumPy 1.25.2, Numba 0.58.1;
  two threads, median of five calls, 1024 by 1024 output) measured 2.249 ms
  serial / 1.242 ms parallel at 31 qubits, and 3.655 / 1.460 ms at 65 qubits.
  This is an observation for the chunk matrix kernels on this machine, not an
  end-to-end speedup or a measured threading crossover. Object conversion,
  parsing, and compilation are excluded.

## Follow-up performance and storage work

Use contiguous arrays and one compiled call per batch now. Keep optional
parallel matrix/expectation paths where directly useful. Further work should
measure thread thresholds, memory use, cold compilation, and supported NumPy /
Numba environments before changing defaults. Cooperating archive writers now
use OS file locking, tested locally with overlapping threads and processes.
Concurrent read snapshots, crash-atomic writes, Windows lock validation, and
full-phase archive support remain follow-up work.

## Completion record

Steps 1-5 are implemented. README and AGENTS document the new object API,
phase/width conventions, legacy compatibility, dependencies, and storage limits.

Validation on September 16, 2026:

- Python 3.10.9 / NumPy 1.25.2 / Numba 0.58.1: **295 passed, 8 skipped**.
- Python 3.9.6 / NumPy 1.25.1 / Numba 0.58.1 with `NUMBA_BOUNDSCHECK=1`:
  **295 passed, 8 skipped**. Both runs used two Numba threads and disabled the
  JIT/pytest caches. The skips are optional Galois/ptgalois reference comparisons.
- The installed Galois initially failed to import because its Numba cache had
  no writable locator. With `NUMBA_CACHE_DIR` set to a temporary writable
  directory, the existing Galois null-space comparison passed separately
  (30 random matrices, sizes 2/5/20, bounds checking enabled). No dependency or
  environment files were changed. The seven ptgalois comparisons remain
  unavailable locally.
- All five README Python examples executed together successfully in a temporary
  directory. The complete public API imports and compiled object algebra run
  in the Python 3.9 environment without Galois or Joblib installed.
- Independent dense-matrix tests cover every phased one- and two-qubit product
  on both backends, stabilizer group/projector preservation, and -I rejection.
  Finite GF(2) oracles check ambient-centralizer and center spaces. Randomized
  products/commutation cover 31/32/64/65/129-qubit boundaries.
- Tests cover empty/zero-width inputs, invalid packed headers/high bits,
  probability and float-integer precision guards, copy/aliasing contracts, and
  archive corruption. Overlapping threads and four spawned writer processes
  exercise storage initialization and append locking on macOS.
- Nopython signatures and LLVM popcount lowering were verified. Parallel
  diagnostics confirm the weighted-estimator row loop is transformed.
- `git diff --check` passes. Pre-existing unrelated files are preserved.

Remaining work is deliberately separated from this implementation: a versioned
full-phase archive format, Windows lock validation, concurrent-read snapshots
and crash recovery, workload-specific threading thresholds, and a broader
NumPy/Numba version matrix. Legacy reflected-list APIs (`append` and tuple
unpacking) still produce three Numba pending-deprecation warnings in this
suite; migration should preserve existing compiled callers. The historical
raw `centralizer` name remains a documented compatibility alias for `center`.

## Bell sampling extraction: implemented September 16, 2026

The shared numerical layer now lives in `src/paulitools/bell_sampling/`, with
top-level exports and separate modules for paired differences, streaming
filters, prepared support spaces/samplers, and the symplectic transform.

- `bell_differences` requires explicit paired streams; the separate cyclic
  operation documents sample reuse. Both XOR supports and discard phases.
- Ordinary commutation masks and Y-shifted Bell masks have separate APIs.
  Compiled row kernels avoid allocating a sample-by-generator matrix and have
  opt-in parallel paths. Filtered scores retain the original denominator.
- `BellSamplePool` snapshots chunks/parities once; immutable filter states
  evaluate added constraints without changing an accepted state on rejection.
- `SupportBasis` fixes the incompatible-pivot coset reduction used in Robels,
  including the commuting `{ZZ, YX}` regression. Membership, combinations,
  intersections, quotient dimensions, and cosets work across chunk boundaries.
- Prepared `SupportSampler` draws uniformly from a span, nonidentity elements,
  or R outside S. Its exclusion decomposition is reused across proposal calls;
  RNG ownership and acceptance policy stay with the consumer.
- `symplectic_fwht` replaces a dense character transform with compiled
  O(n*4**n) work and O(4**n) storage. It does not infer physical normalization.
- `load_legacy_payload` preserves opaque historical int64 archives, with
  structural/checksum validation. Strict Pauli readers/writers remain strict;
  the Harvard readout decoder stays in Robels.
- An additional compiler defect was found on Numba 0.65.1: inlining a raising
  scalar width guard into packed matrix `prange` blocked parallelization.
  Validation now occurs at the public batch boundary; a private unchecked
  parity helper serves the loop. Diagnostics confirm output-row parallelism.

API contracts and examples are in `docs/BELL_SAMPLING.md`. The detailed
`docs/ROBELS_MIGRATION.md` maps consumer wrappers, fixes parser/archive
boundaries, preserves scalar/tuple return contracts, distinguishes radical
from ambient centralizer, and specifies the R-outside-S proposal correction.
Robels source was not edited. Its provenance, verifier guarantees, thresholds,
stopping rules, physical coefficient formulas, and experiment schedules remain
consumer responsibilities. No production simulations were run.

Validation of this extension:

- Final Python 3.10.9 / NumPy 1.25.2 / Numba 0.58.1 suite: **455 passed**.
  This includes 20 new independent Galois support/pipeline comparisons and
  the existing Galois/ptgalois comparisons, using a writable temporary Numba
  cache directory. The recognized optional-Galois cache-locator failure skips
  with an actionable message when that directory is unavailable; other runtime
  errors are not swallowed by the new oracle tests.
- Final Python 3.9.6 / NumPy 1.25.1 / Numba 0.58.1 suite: **427 passed,
  9 skipped**. Skips are optional Galois/ptgalois dependencies absent there.
- Python 3.14.5 / NumPy 2.4.6 / Numba 0.65.1 full integration suite:
  **448 passed, 7 skipped**, plus 70 subtests. This exposed the parallel-loop
  issue above. After its repair, the affected core/boundary suites passed
  **93 tests and 24 subtests**, with no performance warning. Direct serial/
  parallel checks covered empty, singleton, and 100-row matrices at 0/1/31
  qubits. The skips are unavailable ptgalois comparisons.
- All these package runs enabled bounds checking and used two Numba threads.
  Three existing reflected-list pending-deprecation warnings remain.
- New independent tests cover character-level differences, dense transposition
  and commutation masks, canonical cosets and finite spans, sampler exclusions,
  incremental/full score equality, dense transform/inverse equality, and raw
  archive corruption. Boundaries include 0/1/31/32/64/65/129 qubits.
- Read-only Robels migration rehearsal in `robels-modern`: **48 passed**
  across the six selected primitive/processing/verifier/Harvard test files.
  Only `_load_raw_pauli_data` was rebound in memory to `load_legacy_payload`;
  this demonstrates the archive import migration, not completion of the
  remaining consumer changes. The unchanged old import still rejects the
  opaque Harvard header, as strict Pauli loading should.
- Six README and two Bell API examples executed successfully in a temporary
  directory. Top-level exports and setuptools subpackage discovery were
  checked. `git diff --check` passes; unrelated dirty files are preserved.

No end-to-end Bell-workload speedup or threading crossover is claimed. Large
object construction still has per-row Python overhead; reuse prepared objects
and batch calls. The earlier storage/platform follow-ups remain deferred.
