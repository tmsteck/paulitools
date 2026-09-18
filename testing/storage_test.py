import os
import json
import multiprocessing
import threading
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np

from paulitools import (
    PauliIntCollection,
    PauliInt,
    Pauli,
    ZXArray,
    SerializationError,
    append_pauli_data,
    iter_pauli_records,
    load_pauli_data,
    save_pauli_data,
    toZX,
    toZX_extended,
)
import paulitools.storage as storage


def _append_worker(path, worker, barrier, outcomes):
    """Spawn-safe worker: simultaneous first opens and repeated unique batches."""
    try:
        barrier.wait(timeout=20)
        for batch in range(3):
            value = 100 * worker + 2 * batch + 2
            append_pauli_data(path, np.array([10, value, value + 1], dtype=np.int64))
        outcomes.put(None)
    except Exception as exc:
        outcomes.put(repr(exc))


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


class TestObjectStorage(TempDirTestCase):
    def test_pauli_and_zxarray_load_preserve_backend_and_metadata(self):
        for force_large in (False, True):
            with self.subTest(force_large=force_large):
                pauli = Pauli("-XY", force_large=force_large)
                target = self.tmp_path / f"object_{force_large}.ptstore"
                save_pauli_data(target, pauli, user_metadata={"label": "signed"})
                raw = load_pauli_data(target)
                self.assertIsInstance(raw, PauliIntCollection if force_large else np.ndarray)
                wrapped, metadata = load_pauli_data(target, as_zxarray=True, include_metadata=True)
                self.assertIsInstance(wrapped, ZXArray)
                self.assertEqual(wrapped.is_large, force_large)
                self.assertEqual(wrapped.to_strings(), ["-XY"])
                self.assertEqual(metadata, {"label": "signed"})
                append_pauli_data(target, Pauli("ZZ", force_large=force_large))
                self.assertEqual(load_pauli_data(target, as_zxarray=True).to_strings(), ["-XY", "+ZZ"])

    def test_imaginary_phase_rejection_never_touches_destination(self):
        target = self.tmp_path / "preserved.ptstore"
        save_pauli_data(target, np.array([1, 4], dtype=np.int64))
        original = target.read_bytes()
        for value in (Pauli("+iX"), ZXArray.from_input("-iZ", force_large=True)):
            for append in (False, True):
                with self.subTest(value=repr(value), append=append):
                    with self.assertRaisesRegex(ValueError, "version 1.*i phases"):
                        save_pauli_data(target, value, append=append)
                    self.assertEqual(target.read_bytes(), original)
                    missing = self.tmp_path / "uncreated" / "archive.ptstore"
                    with self.assertRaises(ValueError):
                        save_pauli_data(missing, value, append=append)
                    self.assertFalse(missing.parent.exists())

    def test_zero_width_large_archives(self):
        target = self.tmp_path / "zero_width.ptstore"
        save_pauli_data(target, PauliIntCollection(0, []))
        empty = load_pauli_data(target, as_zxarray=True)
        self.assertTrue(empty.is_large)
        self.assertEqual(empty.n_qubits, 0)
        self.assertEqual(len(empty), 0)
        append_pauli_data(target, PauliIntCollection(0, [PauliInt.zeros(0), PauliInt.zeros(0, sign=1)]))
        result = load_pauli_data(target)
        self.assertEqual(result.n_qubits, 0)
        self.assertEqual([p.sign for p in result], [0, 1])
        self.assertEqual([p.z_chunks.shape for p in result], [(0,), (0,)])
        self.assertEqual(len(list(iter_pauli_records(target))), 1)


class TestArchiveValidation(TempDirTestCase):
    def test_invalid_packed_inputs_do_not_truncate(self):
        target = self.tmp_path / "validated.ptstore"
        save_pauli_data(target, np.array([1, 4], dtype=np.int64))
        original = target.read_bytes()
        for invalid in (
            np.array([1.5, 4.0]), np.array([1.0, 4.5]),
            np.array([1, 8]), np.array([32, 0]), np.array([-1, 0]),
            np.array([1, -1]), np.array([], dtype=np.int64),
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises((TypeError, ValueError)):
                    save_pauli_data(target, invalid)
                self.assertEqual(target.read_bytes(), original)

    def test_header_types_and_dimensions_are_validated(self):
        invalid_headers = [
            [], {"version": 1, "format": "legacy", "length": 1.5},
            {"version": 1, "format": "legacy", "length": 32},
            {"version": True, "format": "legacy", "length": 1},
            {"version": 1, "format": "pauliint", "n_qubits": 65, "chunk_count": 1},
            {"version": 1, "format": "pauliint", "n_qubits": -1, "chunk_count": 0},
            {"version": 1, "format": "legacy", "length": 1, "user_metadata": []},
        ]
        target = self.tmp_path / "header.ptstore"
        for header in invalid_headers:
            with self.subTest(header=header):
                payload = json.dumps(header).encode("utf-8")
                target.write_bytes(storage.MAGIC + len(payload).to_bytes(8, "little") + payload)
                original = target.read_bytes()
                with self.assertRaises(SerializationError):
                    load_pauli_data(target)
                with self.assertRaises(SerializationError):
                    append_pauli_data(target, np.array([1, 4], dtype=np.int64))
                self.assertEqual(target.read_bytes(), original)

    def test_checksum_valid_invalid_array_payload_is_rejected(self):
        target = self.tmp_path / "bad_values.ptstore"
        for array, count in [(np.array([4.5]), 1), (np.array([8], np.int64), 1), (np.array([4], np.int64), 2)]:
            with self.subTest(array=array, count=count):
                with target.open("wb") as fh:
                    storage._write_header(fh, {"version": 1, "format": "legacy", "length": 1})
                    storage._write_record(fh, "legacy_batch", count, {"values": storage._npy_bytes(array)})
                with self.assertRaises(SerializationError):
                    load_pauli_data(target)
                with self.assertRaises(SerializationError):
                    list(iter_pauli_records(target))

    def test_save_still_overwrites_existing_records(self):
        target = self.tmp_path / "overwrite.ptstore"
        save_pauli_data(target, np.array([1, 4], dtype=np.int64))
        append_pauli_data(target, np.array([1, 2], dtype=np.int64))
        save_pauli_data(target, np.array([2, 6], dtype=np.int64))
        np.testing.assert_array_equal(load_pauli_data(target), [2, 6])
        self.assertEqual(len(list(iter_pauli_records(target))), 1)

    def test_large_payload_sign_shape_and_padding_are_validated(self):
        target = self.tmp_path / "bad_chunks.ptstore"
        header = {"version": 1, "format": "pauliint", "n_qubits": 65, "chunk_count": 2}
        valid_signs = np.array([0], dtype=np.uint8)
        valid_chunks = np.zeros((1, 2), dtype=np.uint64)
        cases = [
            (np.array([2], np.uint8), valid_chunks, valid_chunks),
            (valid_signs, np.zeros((1, 1), np.uint64), valid_chunks),
            (valid_signs, np.array([[0, 2]], np.uint64), valid_chunks),
            (valid_signs, valid_chunks.astype(np.float64), valid_chunks),
        ]
        for signs, z, x in cases:
            with self.subTest(signs=signs, z=z):
                with target.open("wb") as fh:
                    storage._write_header(fh, header)
                    payloads = {name: storage._npy_bytes(value) for name, value in [("signs", signs), ("z_chunks", z), ("x_chunks", x)]}
                    storage._write_record(fh, "pauli_batch", 1, payloads)
                with self.assertRaises(SerializationError):
                    load_pauli_data(target)
                with self.assertRaises(SerializationError):
                    list(iter_pauli_records(target))


class TestConcurrentStorage(TempDirTestCase):
    def test_lock_covers_creation_and_complete_record_write(self):
        target = self.tmp_path / "threaded.ptstore"
        entered_first = threading.Event()
        release_first = threading.Event()
        second_attempted = threading.Event()
        guard = threading.Lock()
        calls = {"locks": 0, "writes": 0}
        original_lock = storage._exclusive_file_lock
        original_write = storage._write_batch_record

        @contextmanager
        def tracked_lock(fh):
            with guard:
                calls["locks"] += 1
                if calls["locks"] == 2:
                    second_attempted.set()
            with original_lock(fh):
                yield

        def gated_write(*args):
            with guard:
                calls["writes"] += 1
                first = calls["writes"] == 1
            if first:
                entered_first.set()
                if not release_first.wait(10):
                    raise RuntimeError("Test did not release first writer")
            return original_write(*args)

        with patch.object(storage, "_exclusive_file_lock", tracked_lock), patch.object(storage, "_write_batch_record", gated_write):
            with ThreadPoolExecutor(max_workers=2) as executor:
                first = executor.submit(append_pauli_data, target, np.array([1, 4], np.int64))
                try:
                    self.assertTrue(entered_first.wait(10))
                    second = executor.submit(append_pauli_data, target, np.array([1, 2], np.int64))
                    self.assertTrue(second_attempted.wait(10))
                    with self.assertRaises(TimeoutError):
                        second.result(timeout=0.1)
                    self.assertEqual(calls["writes"], 1)
                finally:
                    release_first.set()
                first.result(timeout=10)
                second.result(timeout=10)
        np.testing.assert_array_equal(load_pauli_data(target), [1, 4, 2])

    def test_process_writers_preserve_every_batch_on_new_and_existing_files(self):
        context = multiprocessing.get_context("spawn")
        for existing in (False, True):
            with self.subTest(existing=existing):
                target = self.tmp_path / f"multiprocess_{existing}.ptstore"
                if existing:
                    save_pauli_data(target, np.array([10, 1024], dtype=np.int64))
                barrier = context.Barrier(4)
                outcomes = context.Queue()
                workers = [context.Process(target=_append_worker, args=(str(target), worker, barrier, outcomes)) for worker in range(4)]
                try:
                    for worker in workers:
                        worker.start()
                    for worker in workers:
                        worker.join(timeout=30)
                        self.assertFalse(worker.is_alive(), "Concurrent append worker hung")
                        self.assertEqual(worker.exitcode, 0)
                    self.assertEqual([outcomes.get(timeout=5) for _ in workers], [None] * len(workers))
                finally:
                    for worker in workers:
                        if worker.is_alive():
                            worker.terminate()
                            worker.join(timeout=5)
                    outcomes.close()
                    outcomes.join_thread()
                expected = [100 * worker + 2 * batch + offset for worker in range(4) for batch in range(3) for offset in (2, 3)]
                if existing:
                    expected.append(1024)
                loaded = load_pauli_data(target)
                self.assertEqual(loaded[0], 10)
                self.assertEqual(sorted(loaded[1:]), sorted(expected))
                self.assertEqual(len(list(iter_pauli_records(target))), 12 + int(existing))


if __name__ == "__main__":
    unittest.main()
