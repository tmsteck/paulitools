import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import importlib


def _import_helper(base: str, fallback: str):
    try:
        return importlib.import_module(base)
    except ModuleNotFoundError:  # pragma: no cover - package install
        return importlib.import_module(fallback)


core_module = _import_helper("core", "paulitools.core")
large_module = _import_helper("large_pauli", "paulitools.large_pauli")
storage_module = _import_helper("storage", "paulitools.storage")

toZX = core_module.toZX
toZX_extended = core_module.toZX_extended
PauliIntCollection = large_module.PauliIntCollection
SerializationError = storage_module.SerializationError
append_pauli_data = storage_module.append_pauli_data
iter_pauli_records = storage_module.iter_pauli_records
load_pauli_data = storage_module.load_pauli_data
save_pauli_data = storage_module.save_pauli_data


class TempDirTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        return super().setUp()

    def tearDown(self) -> None:
        self._tmp.cleanup()
        return super().tearDown()


class TestLegacyStorage(TempDirTestCase):
    def test_round_trip_numpy_array(self):
        legacy = toZX(["XX", "YY", "ZZ"])
        target = self.tmp_path / "legacy.pauli"

        save_pauli_data(target, legacy)
        loaded = load_pauli_data(target)

        self.assertIsInstance(loaded, np.ndarray)
        np.testing.assert_array_equal(legacy, loaded)

    def test_append_numpy_array(self):
        base = toZX(["XX", "YY"])
        extra = toZX(["ZZ"])
        target = self.tmp_path / "append.pauli"

        save_pauli_data(target, base)
        append_pauli_data(target, extra)

        loaded = load_pauli_data(target)
        expected = np.concatenate([base[:1], base[1:], extra[1:]])
        np.testing.assert_array_equal(loaded, expected)

    def test_iter_records_legacy(self):
        base = toZX(["XX"])
        extra = toZX(["YY", "ZZ"])
        target = self.tmp_path / "iter.pauli"

        save_pauli_data(target, base)
        append_pauli_data(target, extra)

        batches = list(iter_pauli_records(target))
        self.assertEqual(len(batches), 2)
        np.testing.assert_array_equal(batches[0], base[1:])
        np.testing.assert_array_equal(batches[1], extra[1:])

    def test_append_mismatched_length_raises(self):
        base = toZX(["XX"])
        mismatched = toZX(["XYZ"])
        target = self.tmp_path / "mismatch.pauli"

        save_pauli_data(target, base)
        with self.assertRaises(SerializationError):
            append_pauli_data(target, mismatched)


class TestPauliIntStorage(TempDirTestCase):
    def test_round_trip_collection(self):
        collection = toZX_extended(["XYZI", "ZZXX"], force_large=True)
        self.assertIsInstance(collection, PauliIntCollection)

        target = self.tmp_path / "collection.pauli"
        save_pauli_data(target, collection)
        loaded = load_pauli_data(target)

        self.assertIsInstance(loaded, PauliIntCollection)
        self.assertEqual(loaded.n_qubits, collection.n_qubits)
        self.assertEqual(len(loaded.paulis), len(collection.paulis))
        self.assertEqual(loaded.to_strings(), collection.to_strings())

    def test_round_trip_single_pauli(self):
        collection = toZX_extended("-XYZZ", force_large=True)
        target = self.tmp_path / "single.pauli"
        save_pauli_data(target, collection)
        loaded = load_pauli_data(target)

        self.assertIsInstance(loaded, PauliIntCollection)
        self.assertEqual(len(loaded.paulis), 1)
        self.assertEqual(loaded.to_strings(), collection.to_strings())

    def test_append_collection(self):
        base = toZX_extended(["XXII", "YYII"], force_large=True)
        extra = toZX_extended(["ZZII"], force_large=True)
        target = self.tmp_path / "append_large.pauli"

        save_pauli_data(target, base)
        append_pauli_data(target, extra)

        loaded = load_pauli_data(target)
        self.assertEqual(len(loaded.paulis), 3)
        self.assertEqual(loaded.to_strings(), base.to_strings() + extra.to_strings())

    def test_iter_records_pauli(self):
        base = toZX_extended(["XXII"], force_large=True)
        extra = toZX_extended(["YYII", "ZZII"], force_large=True)
        target = self.tmp_path / "iter_pauli_large.pauli"

        save_pauli_data(target, base)
        append_pauli_data(target, extra)

        batches = list(iter_pauli_records(target))
        self.assertEqual(len(batches), 2)
        self.assertEqual(batches[0].to_strings(), base.to_strings())
        self.assertEqual(batches[1].to_strings(), extra.to_strings())

    def test_checksum_detects_corruption(self):
        collection = toZX_extended("XY" * 20, force_large=True)
        target = self.tmp_path / "corrupt.pauli"
        save_pauli_data(target, collection)

        data = bytearray(target.read_bytes())
        data[-1] ^= 0xFF  # Flip the final byte inside payload
        target.write_bytes(data)

        with self.assertRaises(SerializationError):
            load_pauli_data(target)


class TestMetadata(TempDirTestCase):
    def test_metadata_round_trip(self):
        legacy = toZX(["XX"])
        target = self.tmp_path / "meta.pauli"
        user_metadata = {"experiment": 42}

        save_pauli_data(target, legacy, user_metadata=user_metadata)
        data, metadata = load_pauli_data(target, include_metadata=True)

        np.testing.assert_array_equal(data, legacy)
        self.assertEqual(metadata["experiment"], 42)


if __name__ == "__main__":
    unittest.main()
