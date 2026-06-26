# AGENTS: PauliTools

This file defines durable guidance for contributions in this repository.

## Repository focus
- Primary design goal: fast, repeatedly-called Pauli-string computation (Numba-first) and unified data formatting for ZX representations.
- Execution model: most heavy operations should stay in the hot path functions in `src/paulitools/*.py` and avoid Python loops in callers.
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
  - Mutable ergonomic wrapper over legacy packed arrays and large `PauliIntCollection` data.
- `src/paulitools/storage.py`
  - Log-structured `.ptstore` archive with metadata + checksum-validated records.
- `src/paulitools/util.py`
  - Convenience parity/expectation utilities and purity helpers; some are compiled.
- `src/paulitools/_numba.py`
  - Controls cache behavior via `PAULITOOLS_NUMBA_CACHE`.

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
- `ZXArray` / `toZXArray(...)` are the preferred external-facing APIs for mutable construction, blank/identity allocation, bit access, and backend-neutral conversion.
- `ZXArray.legacy_array(copy=False)` is the explicit boundary for passing wrapped legacy data into Numba hot-path kernels.
- Do not make `ZXArray` replace the raw packed-array ABI for `row_reduce`, `centralizer`, `differences`, or other compiled kernels.

### Storage format (`.ptstore`)
- Magic: `PTSTORE1\n`, JSON header (`version`, `format`).
- Supported `format`: `legacy` or `pauliint`.
- Legacy files store packed values payload; large format stores per-batch `signs`, `z_chunks`, `x_chunks`.
- Appends are record-based and validated by header + checksums.

## API guide and important functions

### 0. Must use first (hot-path)
- `toZX(input_data, fast_input_type=None)`
  - Primary parser for all Pauli input forms.
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
  - Basis reduction before repeated repeated operations.

### 1. Group/theory core utilities
- `radical(paulis, reduced=False)`
  - Center/radical computation.
- `centralizer(pauli_input, reduced=False)`
  - Centralizer of a Pauli set (core benchmark target).
- `differences(paulis, paulis2=None)`
  - Difference rows for Bell-style contracts and sanity tests.
- `ingroup(candidates, pauli_set, reduced=False)`
  - Membership test in the span.
- `null_space(A)`, `inner_product(paulis)`, `row_space(pauli_input)`

### 2. Storage and interoperability
- `save_pauli_data(path, data, append=False, user_metadata=None)`
- `append_pauli_data(path, data)`
- `load_pauli_data(path, include_metadata=False)`
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
- `getCentralizer` (requires optional `galois`; prefer `group.centralizer` for core path).

## External-package rule (mandatory convention)
- When writing code in external consumers/packages, use PauliTools APIs first.
- Reuse existing `paulitools` entry points, especially parsing, symplectic checks, centralizer, and storage.
- Do not copy core Pauli arithmetic into external code.
- If external code needs behavior not present, add/extend in `paulitools` and expose it from top-level API rather than maintaining divergent logic.

## Performance / import notes
- Numba JIT is central; many exported kernels carry `@njit` and must be treated as the default optimization path.
- `PAULITOOLS_NUMBA_CACHE` defaults to off in editable contexts (`NUMBA_CACHE=False`). Enable only where stable cache behavior is desired.

## Practical constraints
- Avoid changing file layout or names unless required for API stability.
- Prefer API-level compatibility and signature stability over short-term convenience.
