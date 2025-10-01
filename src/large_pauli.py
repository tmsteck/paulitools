from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Union

import numpy as np
from numba import njit

# Number of bits stored per chunk. We use unsigned 64-bit chunks to avoid
# sign-extension issues when performing bit manipulation.
_CHUNK_BITS = 64

# The legacy representation in `core.toZX` stores the Pauli operators inside a
# single signed 64-bit integer that reserves one bit for the overall phase.
# This limits the usable number of qubits to 31 (2 * 31 + 1 = 63 bits).  The
# extended representation kicks in for any system that exceeds this limit.
MAX_STANDARD_QUBITS = 31


def _required_chunks(n_qubits: int) -> int:
    """Return how many 64-bit chunks are required to store *n_qubits*."""
    if n_qubits <= 0:
        raise ValueError("Number of qubits must be positive")
    return (n_qubits + _CHUNK_BITS - 1) // _CHUNK_BITS


# ----------------------------------------------------------------------
# Numba-friendly Pauli helpers
# ----------------------------------------------------------------------


@njit(cache=True)
def _required_chunks_nb(n_qubits: int) -> int:
    if n_qubits <= 0:
        raise ValueError("Number of qubits must be positive")
    return (n_qubits + _CHUNK_BITS - 1) // _CHUNK_BITS


@njit(cache=True)
def create_pauli_struct(n_qubits: int, sign: int = 0):
    chunk_count = _required_chunks_nb(n_qubits)
    z_chunks = np.zeros(chunk_count, dtype=np.uint64)
    x_chunks = np.zeros(chunk_count, dtype=np.uint64)
    return n_qubits, np.uint8(sign & 1), z_chunks, x_chunks


@njit(cache=True)
def pauli_struct_set_bits(pauli_struct, qubit_idx: int, x_bit: int, z_bit: int):
    n_qubits, sign, z_chunks, x_chunks = pauli_struct
    if qubit_idx < 0 or qubit_idx >= n_qubits:
        raise IndexError("qubit index out of range")
    chunk_idx = qubit_idx // _CHUNK_BITS
    bit_pos = qubit_idx % _CHUNK_BITS
    mask = np.uint64(1) << np.uint64(bit_pos)
    if x_bit:
        x_chunks[chunk_idx] |= mask
    else:
        x_chunks[chunk_idx] &= ~mask
    if z_bit:
        z_chunks[chunk_idx] |= mask
    else:
        z_chunks[chunk_idx] &= ~mask


@njit(cache=True)
def pauli_struct_get_bits(pauli_struct, qubit_idx: int):
    n_qubits, sign, z_chunks, x_chunks = pauli_struct
    if qubit_idx < 0 or qubit_idx >= n_qubits:
        raise IndexError("qubit index out of range")
    chunk_idx = qubit_idx // _CHUNK_BITS
    bit_pos = qubit_idx % _CHUNK_BITS
    mask = np.uint64(1) << np.uint64(bit_pos)
    x_bit = 1 if (x_chunks[chunk_idx] & mask) != 0 else 0
    z_bit = 1 if (z_chunks[chunk_idx] & mask) != 0 else 0
    return x_bit, z_bit


@njit(cache=True)
def _popcount_uint64(value: np.uint64) -> int:
    count = 0
    while value != 0:
        value &= value - np.uint64(1)
        count += 1
    return count


@njit(cache=True)
def symplectic_inner_product_struct(pauli_a, pauli_b) -> int:
    n_qubits_a, _, z_chunks_a, x_chunks_a = pauli_a
    n_qubits_b, _, z_chunks_b, x_chunks_b = pauli_b
    if n_qubits_a != n_qubits_b:
        raise ValueError("Symplectic inner product requires equal numbers of qubits")
    parity = 0
    for idx in range(z_chunks_a.shape[0]):
        chunk = (x_chunks_a[idx] & z_chunks_b[idx]) ^ (z_chunks_a[idx] & x_chunks_b[idx])
        parity ^= _popcount_uint64(chunk) & 1
    return parity


@njit(cache=True)
def commutes_struct(pauli_a, pauli_b) -> np.int8:
    return np.int8(symplectic_inner_product_struct(pauli_a, pauli_b) == 0)


@njit(cache=True)
def pauli_struct_to_binary(pauli_struct):
    n_qubits, _, z_chunks, x_chunks = pauli_struct
    binary = np.zeros(2 * n_qubits, dtype=np.uint8)
    for qubit in range(n_qubits):
        chunk_idx = qubit // _CHUNK_BITS
        bit_pos = qubit % _CHUNK_BITS
        mask = np.uint64(1) << np.uint64(bit_pos)
        binary[qubit] = 1 if (z_chunks[chunk_idx] & mask) != 0 else 0
        binary[qubit + n_qubits] = 1 if (x_chunks[chunk_idx] & mask) != 0 else 0
    return binary


@njit(cache=True)
def pauli_struct_copy(pauli_struct):
    n_qubits, sign, z_chunks, x_chunks = pauli_struct
    z_copy = z_chunks.copy()
    x_copy = x_chunks.copy()
    return n_qubits, sign, z_copy, x_copy


@dataclass
class PauliInt:
    """Large Pauli operator stored as bit chunks.

    Attributes
    ----------
    n_qubits:
        Number of qubits the operator acts on.
    sign:
        0 for a +1 phase, 1 for -1.
    z_chunks / x_chunks:
        Arrays of unsigned 64-bit integers storing the Z/X components for each
        qubit.  Bit ``i`` of ``z_chunks[j]`` corresponds to the Z component of
        qubit ``i + 64 * j``.
    """

    n_qubits: int
    sign: int
    z_chunks: np.ndarray
    x_chunks: np.ndarray

    def __post_init__(self) -> None:
        chunk_count = _required_chunks(self.n_qubits)
        self.sign = int(self.sign) & 1
        self.z_chunks = np.asarray(self.z_chunks, dtype=np.uint64)
        self.x_chunks = np.asarray(self.x_chunks, dtype=np.uint64)
        if self.z_chunks.shape != (chunk_count,):
            raise ValueError(
                f"z_chunks must have shape {(chunk_count,)}, got {self.z_chunks.shape}"
            )
        if self.x_chunks.shape != (chunk_count,):
            raise ValueError(
                f"x_chunks must have shape {(chunk_count,)}, got {self.x_chunks.shape}"
            )

    @classmethod
    def zeros(cls, n_qubits: int, sign: int = 0) -> "PauliInt":
        chunk_count = _required_chunks(n_qubits)
        return cls(
            n_qubits=n_qubits,
            sign=sign,
            z_chunks=np.zeros(chunk_count, dtype=np.uint64),
            x_chunks=np.zeros(chunk_count, dtype=np.uint64),
        )

    def copy(self) -> "PauliInt":
        n_qubits, sign, z_chunks, x_chunks = pauli_struct_copy(self.as_struct())
        return PauliInt(
            n_qubits=int(n_qubits),
            sign=int(sign),
            z_chunks=z_chunks,
            x_chunks=x_chunks,
        )

    # ------------------------------------------------------------------
    # Bit access helpers
    # ------------------------------------------------------------------
    def set_bits(self, qubit_idx: int, *, x_bit: int, z_bit: int) -> None:
        pauli_struct_set_bits(self.as_struct(), qubit_idx, int(x_bit), int(z_bit))

    def get_bits(self, qubit_idx: int) -> Tuple[int, int]:
        return pauli_struct_get_bits(self.as_struct(), qubit_idx)

    # ------------------------------------------------------------------
    # Representations
    # ------------------------------------------------------------------
    def to_binary(self) -> np.ndarray:
        """Return the operator in ZX binary form (Z bits | X bits)."""
        return pauli_struct_to_binary(self.as_struct())

    def to_string(self, include_sign: bool = True) -> str:
        chars: List[str] = []
        for qubit in range(self.n_qubits):
            x_bit, z_bit = self.get_bits(qubit)
            if x_bit and z_bit:
                chars.append("Y")
            elif x_bit:
                chars.append("X")
            elif z_bit:
                chars.append("Z")
            else:
                chars.append("I")
        pauli_body = "".join(chars) or "I"
        if include_sign:
            prefix = "-" if self.sign else "+"
            return prefix + pauli_body
        return pauli_body

    def as_struct(self):
        """Return a tuple representation compatible with Numba."""
        return (self.n_qubits, np.uint8(self.sign), self.z_chunks, self.x_chunks)

    @classmethod
    def from_struct(cls, pauli_struct):
        n_qubits, sign, z_chunks, x_chunks = pauli_struct
        return cls(
            n_qubits=int(n_qubits),
            sign=int(sign),
            z_chunks=np.asarray(z_chunks, dtype=np.uint64).copy(),
            x_chunks=np.asarray(x_chunks, dtype=np.uint64).copy(),
        )


@dataclass
class PauliIntCollection:
    """Container for multiple :class:`PauliInt` objects."""

    n_qubits: int
    paulis: Sequence[PauliInt]

    def __post_init__(self) -> None:
        for pauli in self.paulis:
            if pauli.n_qubits != self.n_qubits:
                raise ValueError(
                    "All Pauli operators in the collection must share the same number of qubits"
                )

    def __len__(self) -> int:
        return len(self.paulis)

    def __iter__(self):
        return iter(self.paulis)

    def copy(self) -> "PauliIntCollection":
        return PauliIntCollection(
            n_qubits=self.n_qubits,
            paulis=[p.copy() for p in self.paulis],
        )

    def to_binary(self) -> np.ndarray:
        matrix = np.zeros((len(self.paulis), 2 * self.n_qubits), dtype=np.uint8)
        for row, pauli in enumerate(self.paulis):
            matrix[row] = pauli.to_binary()
        return matrix

    def to_strings(self) -> List[str]:
        return [pauli.to_string() for pauli in self.paulis]

    def as_structs(self) -> List[Tuple[int, np.uint8, np.ndarray, np.ndarray]]:
        return [pauli.as_struct() for pauli in self.paulis]


# ----------------------------------------------------------------------
# Conversion helpers
# ----------------------------------------------------------------------


def _strip_sign(pauli_str: str) -> Tuple[int, str]:
    sign = 0
    if pauli_str.startswith("-"):
        sign = 1
        pauli_str = pauli_str[1:]
    elif pauli_str.startswith("+"):
        pauli_str = pauli_str[1:]
    return sign, pauli_str


def pauli_string_to_pauliint(pauli_str: str) -> PauliInt:
    sign, body = _strip_sign(pauli_str)
    pauli = PauliInt.zeros(len(body), sign=sign)
    for idx, char in enumerate(body):
        if char == "X":
            pauli.set_bits(idx, x_bit=1, z_bit=0)
        elif char == "Z":
            pauli.set_bits(idx, x_bit=0, z_bit=1)
        elif char == "Y":
            pauli.set_bits(idx, x_bit=1, z_bit=1)
        elif char == "I":
            continue
        else:
            raise ValueError(f"Invalid Pauli character '{char}' in '{pauli_str}'")
    return pauli


def binary_row_to_pauliint(binary_row: np.ndarray) -> PauliInt:
    if binary_row.ndim != 1:
        raise ValueError("Binary row must be 1D")
    if binary_row.dtype not in (np.uint8, np.int8, np.int64, np.uint64, np.bool_):
        binary_row = binary_row.astype(np.uint8)
    if len(binary_row) % 2 != 0:
        raise ValueError("Binary ZX rows must have even length (Z|X bits)")
    n_qubits = len(binary_row) // 2
    pauli = PauliInt.zeros(n_qubits)
    z_bits = binary_row[:n_qubits]
    x_bits = binary_row[n_qubits:]
    for idx in range(n_qubits):
        pauli.set_bits(idx, x_bit=int(x_bits[idx]) & 1, z_bit=int(z_bits[idx]) & 1)
    return pauli


def standard_to_pauliints(zx_form: np.ndarray) -> PauliIntCollection:
    n_qubits = int(zx_form[0])
    paulis: List[PauliInt] = []
    for val in zx_form[1:]:
        pauli = PauliInt.zeros(n_qubits, sign=int(val) & 1)
        shifted = int(val) >> 1
        for idx in range(n_qubits):
            z_bit = (shifted >> idx) & 1
            x_bit = (shifted >> (idx + n_qubits)) & 1
            pauli.set_bits(idx, x_bit=x_bit, z_bit=z_bit)
        paulis.append(pauli)
    return PauliIntCollection(n_qubits=n_qubits, paulis=paulis)


def pauliints_to_standard(collection: PauliIntCollection) -> np.ndarray:
    if collection.n_qubits > MAX_STANDARD_QUBITS:
        raise ValueError(
            "Cannot convert PauliIntCollection with more than 31 qubits to the standard 64-bit representation"
        )
    result = np.zeros(len(collection.paulis) + 1, dtype=np.int64)
    result[0] = collection.n_qubits
    for idx, pauli in enumerate(collection.paulis, start=1):
        value = pauli.sign
        for qubit in range(pauli.n_qubits):
            x_bit, z_bit = pauli.get_bits(qubit)
            if z_bit:
                value |= 1 << (qubit + 1)
            if x_bit:
                value |= 1 << (qubit + 1 + pauli.n_qubits)
        result[idx] = value
    return result


# ----------------------------------------------------------------------
# High-level parsing utilities
# ----------------------------------------------------------------------


def infer_qubits(input_data: Union[str, Sequence, np.ndarray]) -> int:
    if isinstance(input_data, str):
        stripped = input_data[1:] if input_data.startswith(("+", "-")) else input_data
        if set(stripped) <= {"0", "1"}:
            return len(stripped) // 2
        return len(stripped)
    if isinstance(input_data, np.ndarray):
        if input_data.ndim == 1:
            return len(input_data) // 2
        if input_data.ndim == 2:
            return input_data.shape[1] // 2
        raise ValueError("Unsupported ndarray shape for Pauli data")
    if isinstance(input_data, (list, tuple)):
        if not input_data:
            raise ValueError("Cannot infer qubits from empty input")
        first = input_data[0]
        if isinstance(first, str):
            return infer_qubits(first)
        if isinstance(first, (list, tuple)):
            max_idx = 0
            for pauli, index in input_data:
                max_idx = max(max_idx, int(index))
            return max_idx + 1
        if isinstance(first, np.ndarray):
            return infer_qubits(first)
    raise ValueError("Unable to infer number of qubits from provided data")


def toZX_large(input_data: Union[str, Sequence, np.ndarray]) -> PauliIntCollection:
    n_qubits = infer_qubits(input_data)

    if isinstance(input_data, str):
        pauli = pauli_string_to_pauliint(input_data)
        return PauliIntCollection(n_qubits, [pauli])

    if isinstance(input_data, np.ndarray):
        if input_data.ndim == 1:
            pauli = binary_row_to_pauliint(input_data)
            return PauliIntCollection(n_qubits, [pauli])
        if input_data.ndim == 2:
            paulis = [binary_row_to_pauliint(row) for row in input_data]
            return PauliIntCollection(n_qubits, paulis)
        raise ValueError("Unsupported ndarray shape for Pauli input")

    if isinstance(input_data, (list, tuple)):
        if all(isinstance(item, str) for item in input_data):
            paulis = [pauli_string_to_pauliint(item) for item in input_data]
            return PauliIntCollection(n_qubits, paulis)
        if all(isinstance(item, np.ndarray) for item in input_data):
            paulis = [binary_row_to_pauliint(np.asarray(item)) for item in input_data]
            return PauliIntCollection(n_qubits, paulis)
        if all(isinstance(item, (list, tuple)) for item in input_data):
            pauli = PauliInt.zeros(n_qubits)
            for pauli_char, index in input_data:
                x_bit = 0
                z_bit = 0
                if pauli_char == "X":
                    x_bit = 1
                elif pauli_char == "Z":
                    z_bit = 1
                elif pauli_char == "Y":
                    x_bit = z_bit = 1
                elif pauli_char != "I":
                    raise ValueError(f"Invalid Pauli character '{pauli_char}' in tuple input")
                pauli.set_bits(int(index), x_bit=x_bit, z_bit=z_bit)
            return PauliIntCollection(n_qubits, [pauli])

    raise ValueError("Unsupported input data type for large Pauli conversion")


# ----------------------------------------------------------------------
# Symplectic operations
# ----------------------------------------------------------------------


def symplectic_inner_product_pauliint(a: PauliInt, b: PauliInt) -> int:
    return int(symplectic_inner_product_struct(a.as_struct(), b.as_struct()))


def commutes_pauliint(a: PauliInt, b: PauliInt) -> bool:
    return bool(commutes_struct(a.as_struct(), b.as_struct()))


def _ensure_pauli_struct(value):
    if isinstance(value, PauliInt):
        return value.as_struct()
    if isinstance(value, PauliIntCollection):
        if len(value) != 1:
            raise ValueError("Expected PauliIntCollection with a single element")
        return value.paulis[0].as_struct()
    if isinstance(value, tuple) and len(value) == 4:
        return value
    if isinstance(value, np.ndarray):
        collection = standard_to_pauliints(value)
        if len(collection) != 1:
            raise ValueError("Expected single Pauli operator in array input")
        return collection.paulis[0].as_struct()
    raise TypeError("Unsupported Pauli representation for symplectic operations")


def symplectic_inner_product_any(
    a: Union[PauliInt, np.ndarray],
    b: Union[PauliInt, np.ndarray],
) -> int:
    """Dispatch symplectic inner product between standard and extended forms."""
    struct_a = _ensure_pauli_struct(a)
    struct_b = _ensure_pauli_struct(b)
    return int(symplectic_inner_product_struct(struct_a, struct_b))


def commutes_any(
    a: Union[PauliInt, np.ndarray],
    b: Union[PauliInt, np.ndarray],
) -> bool:
    return bool(commutes_struct(_ensure_pauli_struct(a), _ensure_pauli_struct(b)))


def commutation_matrix(collection: PauliIntCollection) -> np.ndarray:
    size = len(collection)
    mat = np.zeros((size, size), dtype=np.int8)
    structs = collection.as_structs()
    for i, pauli_i in enumerate(structs):
        for j in range(i, size):
            value = symplectic_inner_product_struct(pauli_i, structs[j])
            mat[i, j] = mat[j, i] = value
    return mat


def is_pauliint(obj: object) -> bool:
    return isinstance(obj, PauliInt)


def is_pauliint_collection(obj: object) -> bool:
    return isinstance(obj, PauliIntCollection)


__all__ = [
    "MAX_STANDARD_QUBITS",
    "PauliInt",
    "PauliIntCollection",
    "create_pauli_struct",
    "pauli_struct_set_bits",
    "pauli_struct_get_bits",
    "pauli_struct_to_binary",
    "pauli_struct_copy",
    "toZX_large",
    "standard_to_pauliints",
    "pauliints_to_standard",
    "pauli_string_to_pauliint",
    "binary_row_to_pauliint",
    "infer_qubits",
    "symplectic_inner_product_struct",
    "symplectic_inner_product_pauliint",
    "commutes_struct",
    "commutes_pauliint",
    "symplectic_inner_product_any",
    "commutes_any",
    "commutation_matrix",
    "is_pauliint",
    "is_pauliint_collection",
]
