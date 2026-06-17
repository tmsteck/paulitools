"""Package-level import smoke tests for editable installs."""

from __future__ import annotations


def test_package_import_exposes_large_pauli_helpers():
    import paulitools

    collection = paulitools.toZX_extended("X" * 64)
    reduced = paulitools.row_reduce(paulitools.toZX(["ZI", "IZ"]))

    assert collection.n_qubits == 64
    assert reduced.tolist() == [2, 4, 2]
    assert paulitools.toString_extended(collection) == "+" + "X" * 64
