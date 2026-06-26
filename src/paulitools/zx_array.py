"""Ergonomic object wrapper for PauliTools ZX data."""

from __future__ import annotations

from typing import List

import numpy as np

from .core import (
    GLOBAL_INTEGER,
    commutes_extended,
    symplectic_inner_product,
    symplectic_inner_product_extended,
    toZX_extended,
)
from .large_pauli import (
    MAX_STANDARD_QUBITS,
    PauliInt,
    PauliIntCollection,
    pauliints_to_standard,
    standard_to_pauliints,
)


class ZXArray:
    """Mutable wrapper for legacy packed ZX arrays and PauliInt collections.

    The wrapper is a Python convenience layer.  It deliberately leaves the
    legacy packed ``np.ndarray[int64]`` layout as the ABI for Numba kernels.
    """

    _LEGACY = "legacy"
    _PAULIINT = "pauliint"

    def __init__(self, data, *, backend: str) -> None:
        if backend == self._LEGACY:
            self._data = self._validate_legacy_array(data, copy=False)
        elif backend == self._PAULIINT:
            if not isinstance(data, PauliIntCollection):
                raise TypeError("pauliint backend requires a PauliIntCollection")
            self._data = data
        else:
            raise ValueError("backend must be 'legacy' or 'pauliint'")
        self._backend = backend

    @classmethod
    def from_input(cls, input_data, force_large: bool = False) -> "ZXArray":
        if isinstance(input_data, ZXArray):
            return input_data.copy()
        converted = toZX_extended(input_data, force_large=force_large)
        if isinstance(converted, PauliIntCollection):
            return cls.from_collection(converted, copy=False)
        return cls.from_raw(converted, copy=False)

    @classmethod
    def from_raw(cls, raw, copy: bool = False) -> "ZXArray":
        return cls(cls._validate_legacy_array(raw, copy=copy), backend=cls._LEGACY)

    @classmethod
    def from_collection(
        cls,
        collection: PauliIntCollection,
        copy: bool = False,
    ) -> "ZXArray":
        if not isinstance(collection, PauliIntCollection):
            raise TypeError("collection must be a PauliIntCollection")
        data = collection.copy() if copy else collection
        return cls(data, backend=cls._PAULIINT)

    @classmethod
    def empty(cls, n_qubits: int, force_large: bool = False) -> "ZXArray":
        n_qubits = cls._validate_n_qubits(n_qubits)
        if force_large or n_qubits > MAX_STANDARD_QUBITS:
            return cls.from_collection(PauliIntCollection(n_qubits, []), copy=False)
        raw = np.empty(1, dtype=GLOBAL_INTEGER)
        raw[0] = n_qubits
        return cls.from_raw(raw, copy=False)

    @classmethod
    def identities(
        cls,
        n_qubits: int,
        count: int = 1,
        force_large: bool = False,
    ) -> "ZXArray":
        n_qubits = cls._validate_n_qubits(n_qubits)
        count = cls._validate_count(count)
        if force_large or n_qubits > MAX_STANDARD_QUBITS:
            return cls.from_collection(
                PauliIntCollection(
                    n_qubits,
                    [PauliInt.zeros(n_qubits) for _ in range(count)],
                ),
                copy=False,
            )
        raw = np.zeros(count + 1, dtype=GLOBAL_INTEGER)
        raw[0] = n_qubits
        return cls.from_raw(raw, copy=False)

    @classmethod
    def from_bits(
        cls,
        z_bits,
        x_bits,
        signs=None,
        force_large: bool = False,
    ) -> "ZXArray":
        z_matrix = cls._normalise_bit_matrix(z_bits, name="z_bits")
        x_matrix = cls._normalise_bit_matrix(x_bits, name="x_bits")
        if z_matrix.shape != x_matrix.shape:
            raise ValueError("z_bits and x_bits must have matching shapes")

        count, n_qubits = z_matrix.shape
        sign_bits = cls._normalise_signs(signs, count)

        if force_large or n_qubits > MAX_STANDARD_QUBITS:
            paulis = []
            for row in range(count):
                pauli = PauliInt.zeros(n_qubits, sign=int(sign_bits[row]))
                for qubit in range(n_qubits):
                    pauli.set_bits(
                        qubit,
                        x_bit=int(x_matrix[row, qubit]),
                        z_bit=int(z_matrix[row, qubit]),
                    )
                paulis.append(pauli)
            return cls.from_collection(PauliIntCollection(n_qubits, paulis), copy=False)

        raw = np.empty(count + 1, dtype=GLOBAL_INTEGER)
        raw[0] = n_qubits
        for row in range(count):
            value = int(sign_bits[row])
            for qubit in range(n_qubits):
                if z_matrix[row, qubit]:
                    value |= 1 << (qubit + 1)
                if x_matrix[row, qubit]:
                    value |= 1 << (qubit + 1 + n_qubits)
            raw[row + 1] = value
        return cls.from_raw(raw, copy=False)

    @property
    def n_qubits(self) -> int:
        if self._backend == self._LEGACY:
            return int(self._data[0])
        return int(self._data.n_qubits)

    @property
    def n_paulis(self) -> int:
        if self._backend == self._LEGACY:
            return int(self._data.size - 1)
        return len(self._data.paulis)

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def is_large(self) -> bool:
        return self._backend == self._PAULIINT

    @property
    def data(self):
        return self._data

    @property
    def packed_values(self) -> np.ndarray:
        if self._backend != self._LEGACY:
            raise TypeError("packed_values is only available for the legacy backend")
        return self._data[1:]

    def __len__(self) -> int:
        return self.n_paulis

    def __repr__(self) -> str:
        return (
            f"ZXArray(n_qubits={self.n_qubits}, "
            f"n_paulis={self.n_paulis}, backend='{self.backend}')"
        )

    def copy(self) -> "ZXArray":
        if self._backend == self._LEGACY:
            return ZXArray.from_raw(self._data, copy=True)
        return ZXArray.from_collection(self._data, copy=True)

    def legacy_array(self, copy: bool = False) -> np.ndarray:
        if self._backend == self._LEGACY:
            return self._data.copy() if copy else self._data
        return pauliints_to_standard(self._data)

    def pauliint_collection(self, copy: bool = False) -> PauliIntCollection:
        if self._backend == self._PAULIINT:
            return self._data.copy() if copy else self._data
        collection = standard_to_pauliints(self._data)
        return collection.copy() if copy else collection

    def z_bits(self) -> np.ndarray:
        if self._backend == self._PAULIINT:
            return self._data.to_binary()[:, : self.n_qubits].astype(np.uint8, copy=False)
        z_matrix, _ = self._legacy_bit_matrices()
        return z_matrix

    def x_bits(self) -> np.ndarray:
        if self._backend == self._PAULIINT:
            return self._data.to_binary()[:, self.n_qubits :].astype(np.uint8, copy=False)
        _, x_matrix = self._legacy_bit_matrices()
        return x_matrix

    def binary(self) -> np.ndarray:
        if self._backend == self._PAULIINT:
            return self._data.to_binary().astype(np.uint8, copy=False)
        z_matrix, x_matrix = self._legacy_bit_matrices()
        return np.concatenate([z_matrix, x_matrix], axis=1)

    def signs(self) -> np.ndarray:
        if self._backend == self._PAULIINT:
            return np.asarray([pauli.sign for pauli in self._data.paulis], dtype=np.uint8)
        return (self._data[1:] & 1).astype(np.uint8, copy=True)

    def to_strings(self) -> List[str]:
        if self._backend == self._PAULIINT:
            return self._data.to_strings()
        output: List[str] = []
        for row in range(self.n_paulis):
            sign = "-" if int(self._data[row + 1]) & 1 else "+"
            chars = []
            for qubit in range(self.n_qubits):
                z_bit, x_bit = self.get_bits(row, qubit)
                if z_bit and x_bit:
                    chars.append("Y")
                elif x_bit:
                    chars.append("X")
                elif z_bit:
                    chars.append("Z")
                else:
                    chars.append("I")
            output.append(sign + "".join(chars))
        return output

    def get_bits(self, row: int, qubit: int) -> tuple[int, int]:
        row = self._validate_row(row)
        qubit = self._validate_qubit(qubit)
        if self._backend == self._PAULIINT:
            x_bit, z_bit = self._data.paulis[row].get_bits(qubit)
            return int(z_bit), int(x_bit)
        value = int(self._data[row + 1])
        z_bit = (value >> (qubit + 1)) & 1
        x_bit = (value >> (qubit + 1 + self.n_qubits)) & 1
        return int(z_bit), int(x_bit)

    def set_bits(
        self,
        row: int,
        qubit: int,
        *,
        z_bit=None,
        x_bit=None,
    ) -> None:
        row = self._validate_row(row)
        qubit = self._validate_qubit(qubit)
        current_z, current_x = self.get_bits(row, qubit)
        new_z = current_z if z_bit is None else self._validate_bit(z_bit, name="z_bit")
        new_x = current_x if x_bit is None else self._validate_bit(x_bit, name="x_bit")

        if self._backend == self._PAULIINT:
            self._data.paulis[row].set_bits(qubit, x_bit=new_x, z_bit=new_z)
            return

        value = int(self._data[row + 1])
        z_mask = 1 << (qubit + 1)
        x_mask = 1 << (qubit + 1 + self.n_qubits)
        value = (value | z_mask) if new_z else (value & ~z_mask)
        value = (value | x_mask) if new_x else (value & ~x_mask)
        self._data[row + 1] = value

    def set_sign(self, row: int, sign) -> None:
        row = self._validate_row(row)
        sign_bit = self._validate_bit(sign, name="sign")
        if self._backend == self._PAULIINT:
            self._data.paulis[row].sign = sign_bit
            return

        value = int(self._data[row + 1])
        self._data[row + 1] = (value | 1) if sign_bit else (value & ~1)

    def set_pauli(self, row: int, value) -> None:
        row = self._validate_row(row)
        incoming = self._coerce_value(value, force_large=self.is_large)
        if incoming.n_paulis != 1:
            raise ValueError("set_pauli requires exactly one Pauli operator")
        self._require_same_qubits(incoming)

        if self._backend == self._LEGACY:
            self._data[row + 1] = incoming.legacy_array(copy=False)[1]
            return

        paulis = list(self._data.paulis)
        paulis[row] = incoming.pauliint_collection(copy=True).paulis[0]
        self._data = PauliIntCollection(self.n_qubits, paulis)

    def append(self, value) -> None:
        incoming = self._coerce_value(value, force_large=self.is_large)
        if incoming.n_paulis != 1:
            raise ValueError("append requires exactly one Pauli operator")
        self.extend(incoming)

    def extend(self, value) -> None:
        incoming = self._coerce_value(value, force_large=self.is_large)
        self._require_same_qubits(incoming)
        if incoming.n_paulis == 0:
            return

        if self._backend == self._LEGACY:
            raw = incoming.legacy_array(copy=False)
            self._data = np.concatenate([self._data, raw[1:].astype(GLOBAL_INTEGER)])
            return

        collection = incoming.pauliint_collection(copy=True)
        self._data = PauliIntCollection(
            self.n_qubits,
            list(self._data.paulis) + list(collection.paulis),
        )

    def symplectic_inner_product(self, other) -> int:
        other_zx = self._coerce_value(other, force_large=self.is_large)
        if self.n_paulis != 1 or other_zx.n_paulis != 1:
            raise ValueError("symplectic_inner_product requires single-Pauli ZXArray objects")
        if not self.is_large and not other_zx.is_large:
            return int(
                symplectic_inner_product(
                    self.legacy_array(copy=False),
                    other_zx.legacy_array(copy=False),
                    None,
                )
            )
        return int(
            symplectic_inner_product_extended(
                self._single_extended_value(),
                other_zx._single_extended_value(),
            )
        )

    def commutes(self, other) -> bool:
        other_zx = self._coerce_value(other, force_large=self.is_large)
        if self.n_paulis != 1 or other_zx.n_paulis != 1:
            raise ValueError("commutes requires single-Pauli ZXArray objects")
        return bool(
            commutes_extended(
                self._single_extended_value(),
                other_zx._single_extended_value(),
            )
        )

    def _legacy_bit_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        n_qubits = self.n_qubits
        z_matrix = np.zeros((self.n_paulis, n_qubits), dtype=np.uint8)
        x_matrix = np.zeros((self.n_paulis, n_qubits), dtype=np.uint8)
        for row in range(self.n_paulis):
            value = int(self._data[row + 1])
            for qubit in range(n_qubits):
                z_matrix[row, qubit] = (value >> (qubit + 1)) & 1
                x_matrix[row, qubit] = (value >> (qubit + 1 + n_qubits)) & 1
        return z_matrix, x_matrix

    def _single_extended_value(self):
        if self.n_paulis != 1:
            raise ValueError("Expected a single Pauli operator")
        if self._backend == self._LEGACY:
            return self._data
        return self._data

    def _require_same_qubits(self, other: "ZXArray") -> None:
        if self.n_qubits != other.n_qubits:
            raise ValueError("Pauli data must have the same number of qubits")

    def _validate_row(self, row: int) -> int:
        row = int(row)
        if row < 0 or row >= self.n_paulis:
            raise IndexError("row index out of range")
        return row

    def _validate_qubit(self, qubit: int) -> int:
        qubit = int(qubit)
        if qubit < 0 or qubit >= self.n_qubits:
            raise IndexError("qubit index out of range")
        return qubit

    @staticmethod
    def _validate_n_qubits(n_qubits: int) -> int:
        n_qubits = int(n_qubits)
        if n_qubits < 0:
            raise ValueError("n_qubits must be non-negative")
        return n_qubits

    @staticmethod
    def _validate_count(count: int) -> int:
        count = int(count)
        if count < 0:
            raise ValueError("count must be non-negative")
        return count

    @staticmethod
    def _validate_bit(value, *, name: str) -> int:
        bit = int(value)
        if bit not in (0, 1):
            raise ValueError(f"{name} must be 0 or 1")
        return bit

    @classmethod
    def _validate_legacy_array(cls, raw, copy: bool) -> np.ndarray:
        if copy:
            array = np.array(raw, dtype=GLOBAL_INTEGER, copy=True)
        else:
            array = np.asarray(raw, dtype=GLOBAL_INTEGER)
        if array.ndim != 1 or array.size == 0:
            raise TypeError("Legacy ZX arrays must be one-dimensional and non-empty")
        n_qubits = int(array[0])
        if n_qubits < 0:
            raise ValueError("Legacy ZX arrays must have a non-negative qubit count")
        if n_qubits > MAX_STANDARD_QUBITS:
            raise ValueError("Legacy ZX arrays cannot store more than 31 qubits")
        return array

    @classmethod
    def _normalise_bit_matrix(cls, bits, *, name: str) -> np.ndarray:
        matrix = np.asarray(bits)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        elif matrix.ndim != 2:
            raise ValueError(f"{name} must be a 1D or 2D array")
        if matrix.shape[1] < 0:
            raise ValueError(f"{name} has invalid width")
        if np.any((matrix != 0) & (matrix != 1)):
            raise ValueError(f"{name} must contain only 0/1 values")
        return matrix.astype(np.uint8, copy=False)

    @classmethod
    def _normalise_signs(cls, signs, count: int) -> np.ndarray:
        if signs is None:
            return np.zeros(count, dtype=np.uint8)
        sign_array = np.asarray(signs)
        if sign_array.ndim == 0:
            sign_array = np.full(count, int(sign_array), dtype=np.uint8)
        else:
            sign_array = sign_array.reshape(-1)
        if sign_array.size != count:
            raise ValueError("signs must be scalar or have one entry per Pauli row")
        if np.any((sign_array != 0) & (sign_array != 1)):
            raise ValueError("signs must contain only 0/1 values")
        return sign_array.astype(np.uint8, copy=False)

    @classmethod
    def _coerce_value(cls, value, *, force_large: bool = False) -> "ZXArray":
        if isinstance(value, ZXArray):
            return value
        if isinstance(value, PauliInt):
            return cls.from_collection(PauliIntCollection(value.n_qubits, [value]), copy=False)
        if isinstance(value, PauliIntCollection):
            return cls.from_collection(value, copy=False)
        return cls.from_input(value, force_large=force_large)


def toZXArray(input_data, force_large: bool = False) -> ZXArray:
    """Convert Pauli input data to a mutable :class:`ZXArray` wrapper."""

    return ZXArray.from_input(input_data, force_large=force_large)


def is_zxarray(obj: object) -> bool:
    return isinstance(obj, ZXArray)


__all__ = ["ZXArray", "toZXArray", "is_zxarray"]
