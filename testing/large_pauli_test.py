import os
import sys
import unittest

import numpy as np
from numba import njit

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from paulitools import (
    MAX_STANDARD_QUBITS,
    PauliInt,
    PauliIntCollection,
    create_pauli_struct,
    pauli_struct_set_bits,
    pauli_struct_get_bits,
    pauli_struct_to_binary,
    toZX,
    toZX_extended,
    toZX_large,
    toString_extended,
    symplectic_inner_product_extended,
    commutes_extended,
    to_standard_if_possible,
    commutation_matrix,
    symplectic_inner_product_struct,
    commutes_struct,
)


@njit(cache=True)
def _compiled_symplectic(pauli_a, pauli_b):
    return symplectic_inner_product_struct(pauli_a, pauli_b)


@njit(cache=True)
def _compiled_commutes(pauli_a, pauli_b):
    return commutes_struct(pauli_a, pauli_b)


class TestLargePauliConversions(unittest.TestCase):
    def test_toZX_extended_uses_large_for_64_qubits(self):
        pauli_string = "X" * 64
        result = toZX_extended(pauli_string)
        self.assertIsInstance(result, PauliIntCollection)
        self.assertEqual(result.n_qubits, 64)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.paulis[0].to_string(), "+" + "X" * 64)

    def test_toZX_extended_force_large_on_small_system(self):
        pauli_string = "XZ"
        result = toZX_extended(pauli_string, force_large=True)
        self.assertIsInstance(result, PauliIntCollection)
        legacy = toZX(pauli_string)
        self.assertEqual(to_standard_if_possible(result).tolist(), legacy.tolist())

    def test_toZX_extended_matches_legacy_when_small(self):
        pauli_string = "XZ"
        extended = toZX_extended(pauli_string)
        legacy = toZX(pauli_string)
        self.assertIsInstance(extended, np.ndarray)
        np.testing.assert_array_equal(extended, legacy)

    def test_toZX_large_from_binary_array(self):
        binary = np.zeros(128, dtype=np.uint8)
        binary[0] = 1  # Z on qubit 0
        binary[64] = 1  # X on qubit 0
        collection = toZX_large(binary)
        self.assertEqual(collection.n_qubits, 64)
        self.assertEqual(collection.paulis[0].to_binary()[0], 1)
        self.assertEqual(collection.paulis[0].to_binary()[64], 1)

    def test_toString_extended_round_trip(self):
        pauli_string = "-" + "YZ" * 32
        collection = toZX_extended(pauli_string)
        text = toString_extended(collection)
        self.assertEqual(text, pauli_string)

    def test_symplectic_inner_product_extended_matches_expectation(self):
        x_string = "X" + "I" * 63
        z_string = "Z" + "I" * 63
        x_pauli = toZX_extended(x_string)
        z_pauli = toZX_extended(z_string)
        value = symplectic_inner_product_extended(x_pauli, z_pauli)
        self.assertEqual(value, 1)
        self.assertEqual(commutes_extended(x_pauli, z_pauli), 0)

    def test_commutation_matrix_large(self):
        x0 = toZX_extended("X" + "I" * 63)
        z0 = toZX_extended("Z" + "I" * 63)
        collection = PauliIntCollection(64, [x0.paulis[0], z0.paulis[0]])
        mat = commutation_matrix(collection)
        self.assertEqual(mat.shape, (2, 2))
        self.assertEqual(mat[0, 1], 1)
        self.assertEqual(mat[1, 0], 1)
        self.assertEqual(mat[0, 0], 0)

    def test_to_standard_if_possible_rejects_large(self):
        pauli_string = "X" * (MAX_STANDARD_QUBITS + 1)
        collection = toZX_extended(pauli_string)
        with self.assertRaises(ValueError):
            to_standard_if_possible(collection)

    def test_to_standard_if_possible_small(self):
        pauli_string = "YI"
        collection = toZX_extended(pauli_string, force_large=True)
        converted = to_standard_if_possible(collection)
        self.assertIsInstance(converted, np.ndarray)
        self.assertEqual(converted[0], len(pauli_string))


class TestLargePauliNumba(unittest.TestCase):
    def test_struct_helpers_are_numba_compilable(self):
        pa = create_pauli_struct(70)
        pb = create_pauli_struct(70)

        pauli_struct_set_bits(pa, 5, 1, 0)  # X on qubit 5
        pauli_struct_set_bits(pb, 5, 0, 1)  # Z on qubit 5

        x_bit, z_bit = pauli_struct_get_bits(pa, 5)
        self.assertEqual((x_bit, z_bit), (1, 0))

        binary = pauli_struct_to_binary(pa)
        self.assertEqual(binary[5], 0)
        self.assertEqual(binary[75], 1)

        symp = _compiled_symplectic(pa, pb)
        self.assertEqual(symp, 1)

        comm = _compiled_commutes(pa, pb)
        self.assertEqual(int(comm), 0)

        # Ensure PauliInt wrapper remains in sync with struct updates
        pa_pauli = PauliInt.from_struct(pa)
        pb_pauli = PauliInt.from_struct(pb)
        self.assertFalse(commutes_extended(pa_pauli, pb_pauli))


if __name__ == "__main__":
    unittest.main()
