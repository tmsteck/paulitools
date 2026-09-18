# AGENTS: PauliTools

This file defines durable guidance for contributions in this repository.

## Repository focus
- Primary design goal: fast, repeatedly-called Pauli-string computation (Numba-first) and unified data formatting for ZX representations.
- Execution model: heavy operations stay in compiled kernels under `src/paulitools/`; avoid Python loops in callers.
- The package exports a complete top-level API from `paulitools` (`src/paulitools/__init__.py`).
- Use existing tests in `testing/` as the behavior contract.

## Key architecture
- `src/paulitools/core.py`
  - Fast legacy-representation parser and core algebra helpers.
  - Canonical packed form stores operators in length-prefixed signed `int64` vectors.
- `src/paulitools/group.py`
  - Group-theory helpers for stabilizer workflows (row reductions, radicals, centralizers).
- `src/paulitools/large_pauli.py`
  - Scales beyond 31 qubits with chunked `PauliInt` / `PauliIntCollection`.
- `src/paulitools/zx_array.py`
  - Mutable width- and phase-aware collection over packed arrays and chunked data.
- `src/paulitools/pauli.py` and `src/paulitools/_phase.py`
  - Single-Pauli interface and compiled full-phase batch algebra.
- `src/paulitools/storage.py`
  - Log-structured `.ptstore` archive with metadata + checksum-validated records.
- `src/paulitools/util.py`
  - Convenience parity/expectation utilities and purity helpers; some are compiled.
- `src/paulitools/_numba.py`
  - Controls cache behavior via `PAULITOOLS_NUMBA_CACHE`.
- `src/paulitools/bell_sampling/`
  - Compiled Bell-label differences/filters, prepared support bases and samplers, immutable pool/state snapshots, and symplectic Walsh-Hadamard transform.
  - Start with [docs/BELL_SAMPLING.md](docs/BELL_SAMPLING.md); downstream integration is specified in [docs/ROBELS_MIGRATION.md](docs/ROBELS_MIGRATION.md).

## Unified ZX data contracts

### Legacy packed representation (default)
- Container type: `np.ndarray` with `dtype=int64`.
- First element is `k` = number of qubits.
- Remaining elements are packed Pauli operators, each containing:
  - bit 0: phase sign (`0` for `+`, `1` for `-`),
  - bits `1..k`: Z bits,
  - bits `k+1..2k`: X bits.
- This format is the default for `<= 31` qubits.

### Large representation
- `PauliInt` and `PauliIntCollection` represent arbitrary qubit counts with 64-bit chunks.
- Use `MAX_STANDARD_QUBITS = 31` as the cutoff for legacy representation.

### Ergonomic object wrapper
- `Pauli` and `ZXArray` / `toZXArray(...)` are the preferred external-facing APIs.
- Phase q is `i**q` times ordinary tensor I/X/Y/Z. Multiplication must preserve all four phases; raw sign-only exports reject imaginary phases.
- Qubit zero is the leftmost label character. Numeric object inputs are 0/1 `Z|X` bits by default; eigenvalues require explicit encoding.
- Object collections require equal widths unless padding is explicitly requested. Indexing/slicing return copies; assignment writes back.
- `ZXArray.legacy_array(copy=False)` is the explicit boundary for passing wrapped legacy data into Numba hot-path kernels.
- `kernel_args()` returns independent `(n_qubits, z_chunks, x_chunks, phases)` buffers for full-phase compiled workflows; bind them once outside repeated kernels.
- Do not make `ZXArray` replace the raw packed-array ABI for `row_reduce`, `centralizer`, `differences`, or other compiled kernels.

### Storage format (`.ptstore`)
- Magic: `PTSTORE1\n`, JSON header (`version`, `format`).
- Supported `format`: `legacy` or `pauliint`.
- Legacy files store packed values payload; large format stores per-batch `signs`, `z_chunks`, `x_chunks`.
- Appends are record-based and validated by header + checksums.
- Cooperating writers lock initialization, validation, and writes; concurrent reads are not snapshots and writes are not crash-atomic.
- Version 1 stores real signs only. Reject imaginary-phase inputs before opening/modifying an archive. Do not silently erase phases.
- `load_legacy_payload` is the explicit reader for opaque historical int64 payload archives. It checks structure/checksums but does not interpret the header or payload as valid Paulis. Keep dataset-specific decoding in the consumer; do not weaken normal readers/writers.

## API guide and important functions

### 0. Must use first (hot-path)
- `toZX(input_data, fast_input_type=None)`
  - Legacy real-phase packed parser, limited to 31 qubits. Use object parsing for full phases or larger inputs.
- `toZX_extended(input_data, force_large=False)`
  - Use when qubit counts are large or unknown.
- `toZXArray(input_data, force_large=False)`
  - Use for external construction/mutation/access while preserving explicit conversion to kernel formats.
- `symplectic_inner_product(a, b, k=None)`
- `commutes(a, b, length=None)`
  - Core commutation checks.
- `bsip_array(sym_form_input)` and `commute_array_fast(sym_form_input)`
  - Dense commutation matrices.
- `commutation_matrix(collection: PauliIntCollection)`
  - Large-representation commutation matrix.
- `row_reduce(input_pauli)`
  - Phase-blind support-basis reduction before repeated operations.

### 1. Group/theory core utilities
- `radical(paulis, reduced=False)`
  - Coefficient null space of the reduced support Gram matrix; with `reduced=True`, coefficients refer to the supplied rows.
- `centralizer(pauli_input, reduced=False)`
  - Compatibility alias for historical center-within-span behavior. Do not silently change this compiled API.
- `center(pauli_input, reduced=False)` / `ambient_centralizer(pauli_input, reduced=False)`
  - Explicit center / full ambient centralizer, returning binary Z|X bases.
- `ZXArray.center()` / `ZXArray.centralizer()`
  - Object-returning center / full ambient centralizer; positive support representatives.
- `stabilizer_reduce`, `stabilizer_reduce_bits`, and `ZXArray.stabilizer_basis()`
  - Preserve phases and validate commuting Hermitian generators, rejecting -I relations.
- `differences(paulis, paulis2=None)`
  - Difference rows for Bell-style contracts and sanity tests.
- `ingroup(candidates, pauli_set, reduced=False)`
  - Membership test in the span.
- `null_space(A)`, `inner_product(paulis)`, `row_space(pauli_input)`

### 2. Storage and interoperability
- `save_pauli_data(path, data, append=False, user_metadata=None)`
- `append_pauli_data(path, data)`
- `load_pauli_data(path, include_metadata=False, as_zxarray=False)`
- `load_legacy_payload(path, *, include_metadata=False)`
- `iter_pauli_records(path)`
- `SerializationError`

### 3. Extended representation APIs
- `PauliInt`, `PauliIntCollection`
- `create_pauli_struct`, `pauli_struct_set_bits`, `pauli_struct_get_bits`, `pauli_struct_to_binary`
- `symplectic_inner_product_struct`, `commutes_struct`, `to_standard_if_possible`
- `toString_extended`, `commutes_extended`, `symplectic_inner_product_extended`

### 4. Utility helpers
- `toString(integer_rep)`
- `toBinary(pauli)`, `convert_array_type`
- `popcount`, `getParity`
- `filtered_purity`, `get_purity`, `filtered_purity_reference`
- `get_pauli_obs`, `get_pauli_pauli_obs`, `Pauli_expectation`
- `getCentralizer` (historical center of outcome differences; uses packed kernels).
- Weighted estimators require normalized probabilities and document their Bell-outcome sign/Y conventions. They are not generic state-expectation reconstruction.

### 5. Bell sampling and prepared support API
- `bell_differences(left, right, *, parallel=False)` requires two equal-count, equal-width streams; `cyclic_bell_differences` is an explicitly dependent diagnostic.
- `commuting_mask(samples, generators)` tests symplectic product zero; `bell_filter_mask` tests equality to generator Y parity and rejects noncommuting generators. Both accept optional `parallel=True`.
- `y_parities`, `bell_purity`, `bell_filtered_purity` ignore global phases. Filtered scores divide by the original sample count, never survivors. Empty means raise.
- `BellSamplePool` snapshots chunks/parities. `pool.filter(...)` creates a `BellFilterState`; `.extend(...)` returns a new proposal evaluating only added constraints. A rejected proposal must not mutate current state.
- `SupportBasis` prepares phase-free left-to-right Z|X pivots once for membership, cosets, combinations, span sums/intersections, and quotient dimensions. Do not mix this pivot order with raw packed `row_reduce`.
- `SupportBasis.sampler(exclude_span=S)` prepares uniform sampling from R outside S; reuse `.sample(count, rng=explicit_generator)` until either span changes. Sampling policy and RNG ownership remain with callers.
- `symplectic_fwht` uses canonical index `z | (x << n)`, forward unnormalized, inverse divided by `4**n`. Coefficient powers/Y signs/probability physics stay in Robels.
- Preserve independent dense/GF(2)/Galois tests in `testing/bell_*_test.py`; Galois is optional verification only.
- Keep BellSamples/BellDifferences provenance, verifier schedules/thresholds, adaptive-query guarantees, and experiment accounting in Robels. The numerical package cannot infer independence.

## External-package rule (mandatory convention)
- When writing code in external consumers/packages, use PauliTools APIs first.
- Reuse existing `paulitools` entry points, especially parsing, symplectic checks, centralizer, and storage.
- Do not copy core Pauli arithmetic into external code.
- If external code needs behavior not present, add/extend in `paulitools` and expose it from top-level API rather than maintaining divergent logic.

## Performance / import notes
- Numba JIT is central; many exported kernels carry `@njit` and must be treated as the default optimization path.
- Runtime dependencies are NumPy and Numba only; Galois belongs to optional verification. Python minimum is 3.9.
- Opt-in matrix/estimator parallelism uses compiled row loops. Benchmark before choosing threaded defaults; GF(2) pivot reduction remains sequential.
- Preserve independent matrix/GF(2) oracle tests and exercise boundary inputs with `NUMBA_BOUNDSCHECK=1`.
- `PAULITOOLS_NUMBA_CACHE` defaults to off in editable contexts (`NUMBA_CACHE=False`). Enable only where stable cache behavior is desired.

## Practical constraints
- Avoid changing file layout or names unless required for API stability.
- Prefer API-level compatibility and signature stability over short-term convenience.
