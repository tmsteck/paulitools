"""The compatibility reader preserves bits while retaining archive checks."""

import numpy as np
import pytest

import paulitools.storage as storage
from paulitools.storage import load_legacy_payload, load_pauli_data, save_pauli_data, SerializationError


def _write_fixture(path, *, length=64, batches=(), metadata=None, header=None):
    if header is None:
        header = {"version": 1, "format": "legacy", "length": length}
        if metadata is not None:
            header["user_metadata"] = metadata
    with path.open("wb") as fh:
        storage._write_header(fh, header)
        for values in batches:
            storage._write_record(fh, "legacy_batch", values.size,
                                  {"values": storage._npy_bytes(values)})


def test_historical_signed_int64_bits_and_multiple_records_preserved(tmp_path):
    values = np.array([0, 1, np.iinfo(np.int64).max, np.iinfo(np.int64).min, -1], dtype=np.int64)
    path = tmp_path / "historical.ptstore"
    _write_fixture(path, batches=(values[:2], values[2:]), metadata={"encoding": "historical"})
    actual, metadata = load_legacy_payload(path, include_metadata=True)
    assert actual.dtype == np.int64
    np.testing.assert_array_equal(actual, np.concatenate(([64], values)))
    np.testing.assert_array_equal(actual[1:].view(np.uint64), values.view(np.uint64))
    assert metadata == {"encoding": "historical"}
    with pytest.raises(SerializationError, match="0..31"):
        load_pauli_data(path)
    with pytest.raises(ValueError):
        save_pauli_data(tmp_path / "invalid.ptstore", actual)
    assert not (tmp_path / "invalid.ptstore").exists()


@pytest.mark.parametrize("length", (0, 1, 64, np.iinfo(np.int64).max))
def test_empty_payload_and_header_range(tmp_path, length):
    path = tmp_path / "empty.ptstore"
    _write_fixture(path, length=int(length))
    data, metadata = load_legacy_payload(path, include_metadata=True)
    np.testing.assert_array_equal(data, np.array([length], dtype=np.int64))
    assert metadata == {}


def test_valid_current_archive_supported_without_relaxing_current_loader(tmp_path):
    path = tmp_path / "current.ptstore"
    packed = np.array([1, 0, 1, 4, 7], dtype=np.int64)
    save_pauli_data(path, packed)
    np.testing.assert_array_equal(load_legacy_payload(path), packed)
    np.testing.assert_array_equal(load_pauli_data(path), packed)
    _write_fixture(path, length=1, batches=(np.array([-1, 8], dtype=np.int64),))
    np.testing.assert_array_equal(load_legacy_payload(path), [1, -1, 8])
    with pytest.raises(SerializationError, match="invalid packed"):
        load_pauli_data(path)


def test_checksum_still_validated(tmp_path):
    path = tmp_path / "corrupt.ptstore"
    _write_fixture(path, batches=(np.array([-1], dtype=np.int64),))
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    with pytest.raises(SerializationError, match="Checksum mismatch"):
        load_legacy_payload(path)


@pytest.mark.parametrize("values", (
    np.array([1], dtype=np.uint64), np.array([1], dtype=np.int32),
    np.array([1.0]), np.array([[1]], dtype=np.int64),
))
def test_payload_dtype_and_shape_validated(tmp_path, values):
    path = tmp_path / "shape.ptstore"
    _write_fixture(path, batches=(values,))
    with pytest.raises(SerializationError, match="dtype, shape, or count"):
        load_legacy_payload(path)


@pytest.mark.parametrize("length", (-1, True, 1.0, 2 ** 63, "64", None))
def test_invalid_opaque_header_length_rejected(tmp_path, length):
    path = tmp_path / "header.ptstore"
    _write_fixture(path, length=length)
    with pytest.raises(SerializationError, match="nonnegative int64"):
        load_legacy_payload(path)


@pytest.mark.parametrize("header", (
    {"version": 2, "format": "legacy", "length": 64},
    {"version": True, "format": "legacy", "length": 64},
    {"version": 1, "format": "pauliint", "n_qubits": 64, "chunk_count": 1},
    {"version": 1, "format": "legacy", "length": 64, "user_metadata": []},
))
def test_version_format_metadata_validation_retained(tmp_path, header):
    path = tmp_path / "header.ptstore"
    _write_fixture(path, header=header)
    with pytest.raises(SerializationError):
        load_legacy_payload(path)


def test_record_type_count_and_payload_keys_validated(tmp_path):
    path = tmp_path / "record.ptstore"
    blob = storage._npy_bytes(np.array([-1], dtype=np.int64))
    for record_type, count, payloads in (
        ("pauli_batch", 1, {"values": blob}),
        ("legacy_batch", 2, {"values": blob}),
        ("legacy_batch", 1, {"values": blob, "extra": blob}),
    ):
        with path.open("wb") as fh:
            storage._write_header(fh, {"version": 1, "format": "legacy", "length": 64})
            storage._write_record(fh, record_type, count, payloads)
        with pytest.raises(SerializationError):
            load_legacy_payload(path)


def test_bad_magic_and_truncated_record_rejected(tmp_path):
    path = tmp_path / "broken.ptstore"
    path.write_bytes(b"not a ptstore")
    with pytest.raises(SerializationError, match="Unrecognised"):
        load_legacy_payload(path)
    _write_fixture(path, batches=(np.array([-1], dtype=np.int64),))
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(SerializationError, match="EOF"):
        load_legacy_payload(path)
