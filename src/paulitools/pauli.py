"""Single-Pauli interface; numeric work is delegated to packed batch kernels."""

from __future__ import annotations

from .zx_array import ZXArray


class Pauli:
    """One width-aware operator, with phase i**q times ordinary I/X/Y/Z.

    Construction and collection indexing copy their input. Mutation therefore
    never unexpectedly changes the collection an operator was selected from.
    Use collection assignment to write a modified operator back.
    """

    def __init__(self, value, *, n_qubits=None, force_large=False, encoding="bits"):
        self._zx = ZXArray.from_input(value, force_large, n_qubits=n_qubits, encoding=encoding)
        if len(self._zx) != 1:
            raise ValueError("Pauli requires exactly one operator")

    @classmethod
    def _from_zxarray(cls, value):
        if len(value) != 1:
            raise ValueError("Pauli requires exactly one operator")
        obj = cls.__new__(cls)
        obj._zx = value
        return obj

    @classmethod
    def from_raw(cls, value):
        return cls._from_zxarray(ZXArray.from_raw(value, copy=True))

    @classmethod
    def identity(cls, n_qubits, *, force_large=False):
        return cls._from_zxarray(ZXArray.identities(n_qubits, force_large=force_large))

    @property
    def n_qubits(self):
        return self._zx.n_qubits

    @property
    def phase(self):
        return int(self._zx.phases()[0])

    @phase.setter
    def phase(self, value):
        self._zx.set_phase(0, value)

    def copy(self):
        return self._from_zxarray(self._zx.copy())

    def to_zxarray(self):
        return self._zx.copy()

    def legacy_array(self, copy=True):
        return self._zx.legacy_array(copy=copy)

    def kernel_args(self):
        return self._zx.kernel_args()

    def binary(self):
        return self._zx.binary()[0]

    def to_string(self):
        return self._zx.to_strings()[0]

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        return f"Pauli({self.to_string()!r}, n_qubits={self.n_qubits})"

    def get_bits(self, qubit):
        """Return (z_bit, x_bit), with qubit zero the leftmost character."""
        return self._zx.get_bits(0, qubit)

    def set_bits(self, qubit, *, z_bit=None, x_bit=None):
        self._zx.set_bits(0, qubit, z_bit=z_bit, x_bit=x_bit)

    def pad(self, n_qubits, side="right"):
        return self._from_zxarray(self._zx.pad(n_qubits, side))

    def multiply(self, other):
        other_zx = other._zx if isinstance(other, Pauli) else ZXArray.from_input(other)
        product = self._zx.multiply(other_zx)
        return self._from_zxarray(product) if len(product) == 1 else product

    def __matmul__(self, other):
        """Ordered matrix product; X @ Y equals +iZ."""
        return self.multiply(other)

    def commutes(self, other):
        return self._zx.commutes(other)

    def symplectic_inner_product(self, other):
        return self._zx.symplectic_inner_product(other)

    def equiv(self, other):
        """Equality of support and width, ignoring phase explicitly."""
        return self._zx.equiv(other)

    def __eq__(self, other):
        if not isinstance(other, Pauli):
            return NotImplemented
        return self._zx == other._zx

    def __neg__(self):
        result = self.copy()
        result.phase = (self.phase + 2) % 4
        return result

    def adjoint(self):
        result = self.copy()
        result.phase = (-self.phase) % 4
        return result
