import tempfile
import unittest
from pathlib import Path

import numpy as np

from paulitools import (
    PauliIntCollection,
    ZXArray,
    centralizer,
    differences,
    load_pauli_data,
    row_reduce,
    save_pauli_data,
    toZX,
    toZXArray,
    toZX_extended,
)


class TestZXArrayConstruction(unittest.TestCase):
    def test_empty_and_identity_legacy_construction(self):
        empty = ZXArray.empty(4)
        self.assertEqual(empty.backend, "legacy")
        self.assertEqual(empty.n_qubits, 4)
        self.assertEqual(empty.n_paulis, 0)
        np.testing.assert_array_equal(empty.legacy_array(copy=False), np.array([4], dtype=np.int64))
        self.assertEqual(empty.to_strings(), [])

        identities = ZXArray.identities(3, count=2)
        self.assertEqual(identities.backend, "legacy")
        self.assertEqual(identities.to_strings(), ["+III", "+III"])
        np.testing.assert_array_equal(
            identities.binary(),
            np.zeros((2, 6), dtype=np.uint8),
        )

    def test_raw_wrapping_is_zero_copy_for_legacy_backend(self):
        raw = toZX(["XX", "YZ"])
        wrapped = ZXArray.from_raw(raw)

        self.assertIs(wrapped.legacy_array(copy=False), raw)
        values = wrapped.packed_values
        values[0] = toZX("ZZ")[1]

        self.assertEqual(raw[1], toZX("ZZ")[1])
        self.assertEqual(wrapped.to_strings()[0], "+ZZ")

    def test_collection_wrapping_and_safe_legacy_conversion(self):
        collection = toZX_extended(["XY", "ZZ"], force_large=True)
        wrapped = ZXArray.from_collection(collection)

        self.assertTrue(wrapped.is_large)
        self.assertIs(wrapped.pauliint_collection(copy=False), collection)
        self.assertEqual(wrapped.to_strings(), ["+XY", "+ZZ"])
        np.testing.assert_array_equal(wrapped.legacy_array(), toZX(["XY", "ZZ"]))

    def test_from_bits_includes_signs(self):
        z_bits = np.array([[0, 1], [1, 1]], dtype=np.uint8)
        x_bits = np.array([[1, 0], [1, 1]], dtype=np.uint8)
        wrapped = ZXArray.from_bits(z_bits, x_bits, signs=[0, 1])

        self.assertEqual(wrapped.to_strings(), ["+XZ", "-YY"])
        np.testing.assert_array_equal(wrapped.z_bits(), z_bits)
        np.testing.assert_array_equal(wrapped.x_bits(), x_bits)
        np.testing.assert_array_equal(wrapped.signs(), np.array([0, 1], dtype=np.uint8))


class TestZXArrayMutation(unittest.TestCase):
    def test_mutates_legacy_bits_sign_and_pauli(self):
        wrapped = ZXArray.identities(2)

        wrapped.set_bits(0, 0, z_bit=1)
        self.assertEqual(wrapped.get_bits(0, 0), (1, 0))
        self.assertEqual(wrapped.to_strings(), ["+ZI"])

        wrapped.set_bits(0, 1, x_bit=1)
        self.assertEqual(wrapped.to_strings(), ["+ZX"])

        wrapped.set_bits(0, 0, x_bit=1)
        self.assertEqual(wrapped.to_strings(), ["+YX"])

        wrapped.set_sign(0, 1)
        self.assertEqual(wrapped.to_strings(), ["-YX"])

        wrapped.set_pauli(0, "IZ")
        self.assertEqual(wrapped.to_strings(), ["+IZ"])

    def test_append_and_extend_preserve_backend(self):
        legacy = ZXArray.from_input(["XX"])
        legacy.append("YY")
        legacy.extend(ZXArray.from_input(["ZZ", "II"]))
        self.assertEqual(legacy.backend, "legacy")
        self.assertEqual(legacy.to_strings(), ["+XX", "+YY", "+ZZ", "+II"])

        large = ZXArray.from_input(["XX"], force_large=True)
        large.append("YY")
        large.extend(ZXArray.from_input(["ZZ"], force_large=True))
        self.assertTrue(large.is_large)
        self.assertEqual(large.to_strings(), ["+XX", "+YY", "+ZZ"])
        self.assertIsInstance(large.data, PauliIntCollection)


class TestZXArrayBackendParity(unittest.TestCase):
    def test_legacy_and_forced_large_views_match_for_small_inputs(self):
        legacy = toZXArray(["XY", "-ZI"])
        large = toZXArray(["XY", "-ZI"], force_large=True)

        self.assertFalse(legacy.is_large)
        self.assertTrue(large.is_large)
        self.assertEqual(legacy.to_strings(), large.to_strings())
        np.testing.assert_array_equal(legacy.z_bits(), large.z_bits())
        np.testing.assert_array_equal(legacy.x_bits(), large.x_bits())
        np.testing.assert_array_equal(legacy.binary(), large.binary())
        np.testing.assert_array_equal(legacy.signs(), large.signs())

    def test_true_large_rejects_legacy_conversion(self):
        large = toZXArray("X" * 64)
        self.assertTrue(large.is_large)
        with self.assertRaises(ValueError):
            large.legacy_array(copy=False)

    def test_raw_kernel_interop_matches_current_arrays(self):
        raw = toZX(["XX", "YY", "ZZ"])
        wrapped = toZXArray(["XX", "YY", "ZZ"])

        np.testing.assert_array_equal(row_reduce(wrapped.legacy_array()), row_reduce(raw))
        np.testing.assert_array_equal(centralizer(wrapped.legacy_array()), centralizer(raw))
        np.testing.assert_array_equal(differences(wrapped.legacy_array()), differences(raw))

    def test_commutation_helpers_work_on_both_backends(self):
        self.assertFalse(toZXArray("X").commutes(toZXArray("Z")))
        self.assertEqual(toZXArray("X").symplectic_inner_product(toZXArray("Z")), 1)

        large_x = toZXArray("X" + "I" * 63)
        large_z = toZXArray("Z" + "I" * 63)
        self.assertFalse(large_x.commutes(large_z))
        self.assertEqual(large_x.symplectic_inner_product(large_z), 1)


class TestZXArrayStorage(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_storage_accepts_legacy_zxarray_without_changing_load_type(self):
        wrapped = toZXArray(["XX", "YY"])
        target = self.tmp_path / "legacy_zxarray.ptstore"

        save_pauli_data(target, wrapped)
        loaded = load_pauli_data(target)

        self.assertIsInstance(loaded, np.ndarray)
        np.testing.assert_array_equal(loaded, wrapped.legacy_array())

    def test_storage_accepts_large_zxarray_without_changing_load_type(self):
        wrapped = toZXArray(["XX", "YY"], force_large=True)
        target = self.tmp_path / "large_zxarray.ptstore"

        save_pauli_data(target, wrapped)
        loaded = load_pauli_data(target)

        self.assertIsInstance(loaded, PauliIntCollection)
        self.assertEqual(loaded.to_strings(), wrapped.to_strings())


if __name__ == "__main__":
    unittest.main()
