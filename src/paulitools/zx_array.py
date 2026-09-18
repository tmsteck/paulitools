"""Width-aware Pauli collections with explicit phases and packed kernel interop."""

from __future__ import annotations

import operator
from typing import List

import numpy as np

from .core import GLOBAL_INTEGER, toZX_extended
from .large_pauli import (
    MAX_STANDARD_QUBITS, PauliInt, PauliIntCollection,
    pauliints_to_standard, standard_to_pauliints,
)
from ._phase import multiply_chunks, pack_bit_matrices, unpack_chunk_matrices, row_basis_bits


def _split_label(label):
    """Return canonical phase and body; +i/-i are unambiguous phase prefixes.

    Bare i followed by uppercase Pauli letters is also accepted. Lowercase
    strings such as 'ix' keep their historical meaning as the two-qubit IX.
    """
    if not isinstance(label, str):
        raise TypeError("Pauli labels must be strings")
    phase = 0
    if label.startswith(("+i", "-i")):
        phase, body = (1 if label[0] == "+" else 3), label[2:]
    elif label.startswith("i") and label[1:] and all(c in "IXYZ" for c in label[1:]):
        phase, body = 1, label[1:]
    elif label.startswith(("+", "-")):
        phase, body = (0 if label[0] == "+" else 2), label[1:]
    else:
        body = label
    body = body.upper()
    if any(c not in "IXYZ" for c in body):
        raise ValueError("Expected I/X/Y/Z with optional +, -, +i, or -i phase")
    return phase, body


class ZXArray:
    """A homogeneous-width collection of Pauli operators.

    Qubit zero is the leftmost label character. Phase q means i**q times a
    tensor product of ordinary Hermitian I/X/Y/Z. Indexing returns independent
    copies; explicit raw access may alias storage. Legacy sign storage remains
    authoritative for the real component, with a separate imaginary-phase bit.
    Raw legacy/PauliInt exports reject imaginary phases instead of losing them.
    """

    _LEGACY = "legacy"
    _PAULIINT = "pauliint"

    def __init__(self, data, *, backend: str) -> None:
        if backend == self._LEGACY:
            self._data = self._validate_legacy_array(data, copy=False)
        elif backend == self._PAULIINT:
            if not isinstance(data, PauliIntCollection):
                raise TypeError("pauliint backend requires a PauliIntCollection")
            self._validate_n_qubits(data.n_qubits)
            if any(not isinstance(p, PauliInt) or p.n_qubits != data.n_qubits for p in data.paulis):
                raise ValueError("All Pauli rows must match the collection width")
            self._data = data
        else:
            raise ValueError("backend must be 'legacy' or 'pauliint'")
        self._backend = backend
        self._imaginary = np.zeros(self.n_paulis, dtype=np.uint8)

    @classmethod
    def from_input(cls, input_data, force_large=False, *, encoding="bits", n_qubits=None):
        """Construct without implicit padding; n_qubits explicitly right-pads.

        Numeric input is a Z|X bit matrix by default. Use encoding='eigenvalues'
        for +/-1 entries. Packed legacy arrays use from_raw, never inference.
        """
        from .pauli import Pauli
        if n_qubits is not None:
            n_qubits = cls._validate_n_qubits(n_qubits)
        if encoding not in {"bits", "eigenvalues", "auto"}:
            raise ValueError("encoding must be 'bits', 'eigenvalues', or 'auto'")
        if isinstance(input_data, Pauli):
            result = input_data.to_zxarray()
        elif isinstance(input_data, ZXArray):
            result = input_data.copy()
        elif isinstance(input_data, PauliInt):
            result = cls.from_collection(PauliIntCollection(input_data.n_qubits, [input_data.copy()]))
        elif isinstance(input_data, PauliIntCollection):
            result = cls.from_collection(input_data, copy=True)
        else:
            if isinstance(input_data, (list, tuple)) and input_data and all(isinstance(p, Pauli) for p in input_data):
                input_data = [p.to_string() for p in input_data]
            labels = [input_data] if isinstance(input_data, str) else input_data
            is_labels = isinstance(labels, (list, tuple)) and all(isinstance(v, str) for v in labels)
            if is_labels and not labels:
                if n_qubits is None:
                    raise ValueError("Empty input requires n_qubits or ZXArray.empty(n_qubits)")
                return cls.empty(n_qubits, force_large)
            if is_labels and not all(s and set(s) <= {"0", "1"} for s in labels):
                parsed = [_split_label(s) for s in labels]
                widths = [len(body) for _, body in parsed]
                width = max(widths)
                if n_qubits is None and any(w != width for w in widths):
                    raise ValueError("Pauli widths differ; supply n_qubits for explicit right padding")
                target = width if n_qubits is None else cls._validate_n_qubits(n_qubits)
                if target < width:
                    raise ValueError("n_qubits cannot truncate a Pauli")
                phases = np.asarray([q for q, _ in parsed], dtype=np.uint8)
                if target == 0:
                    result = cls.identities(0, len(parsed), force_large)
                else:
                    bodies = [body + "I" * (target - len(body)) for _, body in parsed]
                    converted = toZX_extended(bodies, force_large=force_large, encoding=encoding)
                    result = cls.from_collection(converted) if isinstance(converted, PauliIntCollection) else cls.from_raw(converted)
                for row, phase in enumerate(phases):
                    result.set_phase(row, phase)
            else:
                # Ordinary nested numeric sequences have the same explicit bit
                # contract as ndarrays; indexed (character, qubit) tuples remain
                # the legacy parser's responsibility.
                if isinstance(input_data, (list, tuple)) and input_data and not is_labels:
                    if not all(isinstance(v, tuple) and len(v) == 2 and isinstance(v[0], str) for v in input_data):
                        input_data = np.asarray(input_data)
                converted = toZX_extended(input_data, force_large=force_large, encoding=encoding)
                result = cls.from_collection(converted) if isinstance(converted, PauliIntCollection) else cls.from_raw(converted)
        if force_large and not result.is_large:
            phases = result.phases()
            result = cls.from_collection(standard_to_pauliints(result._data))
            result._imaginary = phases & 1
        if n_qubits is not None and result.n_qubits != n_qubits:
            result = result.pad(n_qubits)
        return result

    @classmethod
    def from_raw(cls, raw, copy=False):
        """Wrap an explicitly length-prefixed legacy array; copy=False may alias."""
        return cls(cls._validate_legacy_array(raw, copy=copy), backend=cls._LEGACY)

    @classmethod
    def from_collection(cls, collection, copy=False):
        if not isinstance(collection, PauliIntCollection):
            raise TypeError("collection must be a PauliIntCollection")
        return cls(collection.copy() if copy else collection, backend=cls._PAULIINT)

    @classmethod
    def empty(cls, n_qubits, force_large=False):
        n_qubits = cls._validate_n_qubits(n_qubits)
        if force_large or n_qubits > MAX_STANDARD_QUBITS:
            return cls.from_collection(PauliIntCollection(n_qubits, []))
        return cls.from_raw(np.asarray([n_qubits], dtype=GLOBAL_INTEGER))

    @classmethod
    def identities(cls, n_qubits, count=1, force_large=False):
        n_qubits, count = cls._validate_n_qubits(n_qubits), cls._validate_count(count)
        if force_large or n_qubits > MAX_STANDARD_QUBITS:
            return cls.from_collection(PauliIntCollection(n_qubits, [PauliInt.zeros(n_qubits) for _ in range(count)]))
        raw = np.zeros(count + 1, dtype=GLOBAL_INTEGER)
        raw[0] = n_qubits
        return cls.from_raw(raw)

    @classmethod
    def from_bits(cls, z_bits, x_bits, signs=None, force_large=False, *, phases=None):
        z = cls._normalise_bit_matrix(z_bits, name="z_bits")
        x = cls._normalise_bit_matrix(x_bits, name="x_bits")
        if z.shape != x.shape:
            raise ValueError("z_bits and x_bits must have matching shapes")
        if phases is not None and signs is not None:
            raise ValueError("Supply phases or signs, not both")
        count, width = z.shape
        q = cls._normalise_phases(phases, count) if phases is not None else 2 * cls._normalise_signs(signs, count)
        z_chunks, x_chunks = pack_bit_matrices(z, x)
        return cls._from_chunks(width, z_chunks, x_chunks, q, force_large)

    @classmethod
    def from_eigenvalues(cls, z_values, x_values, *, phases=None, force_large=False):
        """Explicit +/-1 encoding, with -1 -> bit 1 and +1 -> bit 0."""
        z, x = np.asarray(z_values), np.asarray(x_values)
        if np.any((z != -1) & (z != 1)) or np.any((x != -1) & (x != 1)):
            raise ValueError("Eigenvalues must contain only -1 and +1")
        return cls.from_bits(z < 0, x < 0, phases=phases, force_large=force_large)

    @classmethod
    def _from_chunks(cls, width, z, x, phases, force_large=False):
        count = z.shape[0]
        if force_large or width > MAX_STANDARD_QUBITS:
            paulis = [PauliInt(width, int(phases[r]) // 2, z[r].copy(), x[r].copy()) for r in range(count)]
            result = cls.from_collection(PauliIntCollection(width, paulis))
        else:
            raw = np.zeros(count + 1, dtype=np.int64)
            raw[0] = width
            if width:
                raw[1:] = ((z[:, 0] << np.uint64(1)) | (x[:, 0] << np.uint64(width + 1))).astype(np.int64)
            raw[1:] |= np.asarray(phases, dtype=np.int64) >> 1
            result = cls.from_raw(raw)
        result._imaginary = np.asarray(phases, dtype=np.uint8) & 1
        return result

    @property
    def n_qubits(self):
        return int(self._data.n_qubits) if self.is_large else int(self._data[0])

    @property
    def n_paulis(self):
        return len(self._data.paulis) if self.is_large else self._data.size - 1

    @property
    def backend(self):
        return self._backend

    @property
    def is_large(self):
        return self._backend == self._PAULIINT

    @property
    def data(self):
        """Explicit aliasing access to real-phase backend storage."""
        self._require_real_phase()
        return self._data

    @property
    def packed_values(self):
        self._require_real_phase()
        if self.is_large:
            raise TypeError("packed_values is only available for the legacy backend")
        return self._data[1:]

    def __len__(self):
        return self.n_paulis

    def __repr__(self):
        return f"ZXArray(n_qubits={self.n_qubits}, n_paulis={self.n_paulis}, backend='{self.backend}')"

    def copy(self):
        result = ZXArray.from_collection(self._data, copy=True) if self.is_large else ZXArray.from_raw(self._data, copy=True)
        result._imaginary = self._imaginary.copy()
        return result

    def _take(self, indices):
        indices = np.asarray(indices, dtype=np.intp).reshape(-1)
        if self.is_large:
            result = ZXArray.from_collection(PauliIntCollection(self.n_qubits, [self._data.paulis[int(i)].copy() for i in indices]))
        else:
            result = ZXArray.from_raw(np.concatenate((self._data[:1], self._data[1:][indices])))
        result._imaginary = self._imaginary[indices].copy()
        return result

    def _selected_rows(self, key):
        """Normalize selection without allocating the whole index range for a row."""
        if isinstance(key, (bool, np.bool_)):
            raise IndexError("Use an integer row, slice, or one-dimensional selector")
        if isinstance(key, slice):
            return np.arange(*key.indices(self.n_paulis), dtype=np.intp)
        try:
            return self._validate_row(operator.index(key))
        except TypeError:
            return np.arange(self.n_paulis)[key]

    def __getitem__(self, key):
        from .pauli import Pauli
        selected = self._selected_rows(key)
        if np.ndim(selected) == 0:
            return Pauli._from_zxarray(self._take([int(selected)]))
        if np.ndim(selected) != 1:
            raise IndexError("Pauli collection indices must select a one-dimensional collection")
        return self._take(selected)

    def __setitem__(self, key, value):
        selected = np.asarray(self._selected_rows(key))
        if selected.ndim > 1:
            raise IndexError("Pauli collection indices must select a one-dimensional collection")
        indices = selected.reshape(-1)
        incoming = self._coerce_value(value).copy()
        self._require_same_qubits(incoming)
        if incoming.n_paulis not in (1, len(indices)):
            raise ValueError("Assignment requires matching row counts or one Pauli")
        source = np.zeros(len(indices), dtype=np.intp) if incoming.n_paulis == 1 else np.arange(len(indices))
        z, x = incoming._chunk_arrays()
        replacement = self._from_chunks(self.n_qubits, z[source], x[source], incoming.phases()[source], self.is_large)
        if self.is_large:
            paulis = list(self._data.paulis)
            for i, row in enumerate(indices):
                paulis[int(row)] = replacement._data.paulis[i]
            self._data = PauliIntCollection(self.n_qubits, paulis)
        else:
            self._data[indices + 1] = replacement._data[1:]
        self._imaginary[indices] = replacement._imaginary

    def _require_real_phase(self):
        if np.any(self._imaginary):
            raise ValueError("Legacy sign storage cannot represent +/-i phases; use phases() and binary()/kernel_args()")

    def legacy_array(self, copy=False):
        """Explicit kernel boundary; same-backend copy=False aliases storage."""
        self._require_real_phase()
        if self.is_large:
            return pauliints_to_standard(self._data)
        return self._data.copy() if copy else self._data

    def pauliint_collection(self, copy=False):
        self._require_real_phase()
        if self.is_large:
            return self._data.copy() if copy else self._data
        return standard_to_pauliints(self._data)

    def _chunk_arrays(self):
        chunks = (self.n_qubits + 63) // 64
        if self.n_paulis == 0 or chunks == 0:
            return (np.zeros((self.n_paulis, chunks), dtype=np.uint64),
                    np.zeros((self.n_paulis, chunks), dtype=np.uint64))
        if self.is_large:
            return np.stack([p.z_chunks for p in self._data.paulis]), np.stack([p.x_chunks for p in self._data.paulis])
        values = self._data[1:].astype(np.uint64)
        mask = np.uint64((1 << self.n_qubits) - 1)
        z = ((values >> np.uint64(1)) & mask).reshape(-1, 1)
        x = ((values >> np.uint64(self.n_qubits + 1)) & mask).reshape(-1, 1)
        return z, x

    def kernel_args(self):
        """Return independent (n_qubits, z_chunks, x_chunks, phases) buffers.

        Bind once outside a repeatedly called njit workflow. Mutating these
        buffers does not mutate this object. Legacy zero-copy access remains
        available through legacy_array(copy=False) for real phases.
        """
        z, x = self._chunk_arrays()
        return self.n_qubits, z, x, self.phases()

    def _bit_matrices(self):
        z, x = self._chunk_arrays()
        return unpack_chunk_matrices(z, x, self.n_qubits)

    def z_bits(self):
        return self._bit_matrices()[0]

    def x_bits(self):
        return self._bit_matrices()[1]

    def binary(self):
        """Return support bits as an independent (n_paulis, 2*n_qubits) Z|X matrix."""
        z, x = self._bit_matrices()
        return np.concatenate((z, x), axis=1)

    def _sign_bits(self):
        if self.is_large:
            return np.asarray([p.sign for p in self._data.paulis], dtype=np.uint8)
        return (self._data[1:] & 1).astype(np.uint8)

    def signs(self):
        """Return 0/1 real signs; imaginary phases require phases()."""
        self._require_real_phase()
        return self._sign_bits()

    def phases(self):
        """Return independent q values for the coefficient i**q."""
        return 2 * self._sign_bits() + self._imaginary

    def to_strings(self) -> List[str]:
        z, x = self._bit_matrices()
        letters, prefixes = "IZXY", ("+", "+i", "-", "-i")
        return [prefixes[int(q)] + "".join(letters[int(z[r, c]) + 2 * int(x[r, c])] for c in range(self.n_qubits))
                for r, q in enumerate(self.phases())]

    def get_bits(self, row, qubit):
        row, qubit = self._validate_row(row), self._validate_qubit(qubit)
        if self.is_large:
            x, z = self._data.paulis[row].get_bits(qubit)
            return int(z), int(x)
        value = int(self._data[row + 1])
        return (value >> (qubit + 1)) & 1, (value >> (qubit + 1 + self.n_qubits)) & 1

    def set_bits(self, row, qubit, *, z_bit=None, x_bit=None):
        row, qubit = self._validate_row(row), self._validate_qubit(qubit)
        z, x = self.get_bits(row, qubit)
        z = z if z_bit is None else self._validate_bit(z_bit, name="z_bit")
        x = x if x_bit is None else self._validate_bit(x_bit, name="x_bit")
        if self.is_large:
            self._data.paulis[row].set_bits(qubit, x_bit=x, z_bit=z)
        else:
            value = int(self._data[row + 1])
            zm, xm = 1 << (qubit + 1), 1 << (qubit + 1 + self.n_qubits)
            value = (value | zm) if z else (value & ~zm)
            self._data[row + 1] = (value | xm) if x else (value & ~xm)

    def set_phase(self, row, phase):
        row = self._validate_row(row)
        q = operator.index(phase)
        if q not in (0, 1, 2, 3):
            raise ValueError("phase must be an integer in 0..3")
        if self.is_large:
            self._data.paulis[row].sign = q >> 1
        else:
            self._data[row + 1] = (int(self._data[row + 1]) & ~1) | (q >> 1)
        self._imaginary[row] = q & 1

    def set_sign(self, row, sign):
        """Replace the total phase by +1 (0) or -1 (1)."""
        self.set_phase(row, 2 * self._validate_bit(sign, name="sign"))

    def set_pauli(self, row, value):
        self[self._validate_row(row)] = value

    def append(self, value):
        incoming = self._coerce_value(value)
        if incoming.n_paulis != 1:
            raise ValueError("append requires exactly one Pauli")
        self.extend(incoming)

    def extend(self, value):
        incoming = self._coerce_value(value)
        self._require_same_qubits(incoming)
        if not incoming.n_paulis:
            return
        z, x = incoming._chunk_arrays()
        converted = self._from_chunks(self.n_qubits, z, x, incoming.phases(), self.is_large)
        imaginary = np.concatenate((self._imaginary, converted._imaginary))
        if self.is_large:
            self._data = PauliIntCollection(self.n_qubits, list(self._data.paulis) + list(converted._data.paulis))
        else:
            self._data = np.concatenate((self._data, converted._data[1:]))
        self._imaginary = imaginary

    def pad(self, n_qubits, side="right"):
        """Return a copy with explicitly inserted identity qubits."""
        width = self._validate_n_qubits(n_qubits)
        if width < self.n_qubits:
            raise ValueError("Padding cannot truncate a Pauli")
        if side not in ("right", "left"):
            raise ValueError("side must be 'right' or 'left'")
        z, x = self._bit_matrices()
        padding = width - self.n_qubits
        before, after = (0, padding) if side == "right" else (padding, 0)
        return ZXArray.from_bits(np.pad(z, ((0, 0), (before, after))), np.pad(x, ((0, 0), (before, after))),
                                 phases=self.phases(), force_large=self.is_large)

    def multiply(self, other):
        """Exact ordered product self @ other, with single-row broadcasting."""
        other = self._coerce_value(other)
        self._require_same_qubits(other)
        za, xa = self._chunk_arrays()
        zb, xb = other._chunk_arrays()
        z, x, q = multiply_chunks(za, xa, self.phases(), zb, xb, other.phases())
        return self._from_chunks(self.n_qubits, z, x, q, self.is_large or other.is_large)

    def __matmul__(self, other):
        return self.multiply(other)

    def symplectic_matrix(self, other=None, *, parallel=False):
        """Rectangular support pairing, 1=anticommutes; phases are irrelevant."""
        from .large_pauli import _symplectic_matrix_chunks, _symplectic_matrix_chunks_parallel
        other = self if other is None else self._coerce_value(other)
        self._require_same_qubits(other)
        za, xa = self._chunk_arrays()
        zb, xb = other._chunk_arrays()
        kernel = _symplectic_matrix_chunks_parallel if parallel else _symplectic_matrix_chunks
        return kernel(za, xa, zb, xb)

    def commutation_matrix(self, other=None, *, parallel=False):
        """Boolean matrix with True=commutes, for either backend."""
        return self.symplectic_matrix(other, parallel=parallel) == 0

    def symplectic_inner_product(self, other):
        other = self._coerce_value(other)
        if self.n_paulis != 1 or other.n_paulis != 1:
            raise ValueError("Expected single-Pauli operands; use symplectic_matrix for collections")
        return int(self.symplectic_matrix(other)[0, 0])

    def commutes(self, other):
        return self.symplectic_inner_product(other) == 0

    def equiv(self, other):
        """Whether collections are equal in order and width, ignoring phase."""
        other = self._coerce_value(other)
        if self.n_qubits != other.n_qubits or self.n_paulis != other.n_paulis:
            return False
        za, xa = self._chunk_arrays()
        zb, xb = other._chunk_arrays()
        return bool(np.array_equal(za, zb) and np.array_equal(xa, xb))

    def __eq__(self, other):
        if not isinstance(other, ZXArray):
            return NotImplemented
        return self.equiv(other) and bool(np.array_equal(self.phases(), other.phases()))

    def support_basis(self):
        """Independent support basis; explicitly discards phases."""
        bits = row_basis_bits(self.binary())
        return ZXArray.from_bits(bits[:, :self.n_qubits], bits[:, self.n_qubits:], force_large=self.is_large)

    def center(self):
        """Basis of S intersect S-perp, as canonical positive representatives."""
        from .group import null_space, matmul_mod2
        basis = self.support_basis()
        coeff = null_space(basis.symplectic_matrix())
        bits = matmul_mod2(coeff, basis.binary())
        return ZXArray.from_bits(bits[:, :self.n_qubits], bits[:, self.n_qubits:], force_large=self.is_large)

    def centralizer(self):
        """Full ambient commuting support space S-perp, including outside S."""
        from .group import null_space
        z, x = self._bit_matrices()
        bits = null_space(np.concatenate((x, z), axis=1))
        return ZXArray.from_bits(bits[:, :self.n_qubits], bits[:, self.n_qubits:], force_large=self.is_large)

    def stabilizer_basis(self):
        """Phase-preserving commuting Hermitian basis; reject any -I relation."""
        from .group import stabilizer_reduce_bits
        z, x = self._bit_matrices()
        zr, xr, phases = stabilizer_reduce_bits(z, x, self.phases())
        return ZXArray.from_bits(zr, xr, phases=phases, force_large=self.is_large)

    def _require_same_qubits(self, other):
        if self.n_qubits != other.n_qubits:
            raise ValueError("Pauli widths differ; use pad(n_qubits) explicitly")

    def _validate_row(self, row):
        row = operator.index(row)
        if row < 0:
            row += self.n_paulis
        if row < 0 or row >= self.n_paulis:
            raise IndexError("row index out of range")
        return row

    def _validate_qubit(self, qubit):
        qubit = operator.index(qubit)
        if qubit < 0 or qubit >= self.n_qubits:
            raise IndexError("qubit index out of range")
        return qubit

    @staticmethod
    def _validate_n_qubits(value):
        value = operator.index(value)
        if value < 0:
            raise ValueError("n_qubits must be non-negative")
        return value

    @staticmethod
    def _validate_count(value):
        value = operator.index(value)
        if value < 0:
            raise ValueError("count must be non-negative")
        return value

    @staticmethod
    def _validate_bit(value, *, name):
        value = operator.index(value)
        if value not in (0, 1):
            raise ValueError(f"{name} must be 0 or 1")
        return value

    @staticmethod
    def _validate_legacy_array(raw, copy):
        array = np.asarray(raw)
        if array.ndim != 1 or not array.size or array.dtype.kind not in "iu":
            raise TypeError("Legacy ZX data must be a nonempty one-dimensional integer array")
        width = int(array[0])
        if width < 0 or width > MAX_STANDARD_QUBITS:
            raise ValueError("Legacy ZX arrays require 0..31 qubits")
        if np.any(array[1:] < 0) or np.any(array[1:] > (1 << (2 * width + 1)) - 1):
            raise ValueError("Packed values contain bits outside their declared width")
        return np.array(array, dtype=np.int64, copy=True) if copy else np.asarray(array, dtype=np.int64)

    @staticmethod
    def _normalise_bit_matrix(bits, *, name):
        matrix = np.asarray(bits)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2 or np.any((matrix != 0) & (matrix != 1)):
            raise ValueError(f"{name} must be a 1D or 2D array of 0/1 bits")
        return matrix.astype(np.uint8)

    @staticmethod
    def _normalise_values(values, count, limit, name):
        if values is None:
            return np.zeros(count, dtype=np.uint8)
        array = np.asarray(values)
        if array.ndim == 0:
            array = np.full(count, array.item())
        if array.ndim != 1 or array.size != count:
            raise ValueError(f"{name} must be scalar or have one entry per row")
        if np.any(~np.isin(array, np.arange(limit))):
            raise ValueError(f"{name} must contain integers in 0..{limit - 1}")
        return array.astype(np.uint8)

    @classmethod
    def _normalise_signs(cls, signs, count):
        return cls._normalise_values(signs, count, 2, "signs")

    @classmethod
    def _normalise_phases(cls, phases, count):
        return cls._normalise_values(phases, count, 4, "phases")

    @classmethod
    def _coerce_value(cls, value, force_large=False):
        return value if isinstance(value, ZXArray) and not force_large else cls.from_input(value, force_large)


def toZXArray(input_data, force_large=False, *, encoding="bits", n_qubits=None):
    return ZXArray.from_input(input_data, force_large, encoding=encoding, n_qubits=n_qubits)


def is_zxarray(obj):
    return isinstance(obj, ZXArray)
