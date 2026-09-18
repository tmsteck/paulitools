"""Shared validation for the object-facing Bell numerical API.

Numeric inputs use the ZXArray bit-matrix convention. Packed legacy arrays
must be wrapped explicitly with ZXArray.from_raw; phases are interpreted by
the operation, never inferred from integer encodings here.
"""

import operator
import numpy as np

from ..pauli import Pauli
from ..zx_array import ZXArray


def as_collection(value, *, n_qubits=None):
    if n_qubits is not None:
        n_qubits = nonnegative_count(n_qubits, name="n_qubits")
    if isinstance(value, ZXArray):
        result = value
    elif isinstance(value, Pauli):
        result = value.to_zxarray()
    else:
        # Explicit width on an empty collection is useful at API boundaries.
        if isinstance(value, (list, tuple)) and len(value) == 0 and n_qubits is not None:
            result = ZXArray.empty(n_qubits)
        else:
            result = ZXArray.from_input(value)
    if n_qubits is not None and result.n_qubits != n_qubits:
        raise ValueError("Pauli widths differ; pad inputs explicitly")
    return result


def nonnegative_count(value, *, name="count"):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not bool")
    result = operator.index(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result
