"""Append-friendly storage for Pauli data."""

from __future__ import annotations

import io
import json
import os
import hashlib
import operator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union

import numpy as np

from .large_pauli import (
    PauliInt,
    PauliIntCollection,
    is_pauliint,
    is_pauliint_collection,
)
from .zx_array import ZXArray, is_zxarray
from .pauli import Pauli

PauliLike = Union[np.ndarray, Pauli, PauliInt, PauliIntCollection, ZXArray, Iterable[PauliInt]]

MAGIC = b"PTSTORE1\n"
CURRENT_VERSION = 1


class SerializationError(RuntimeError):
    """Raised when a stored Pauli archive fails validation."""


@dataclass
class LegacyBatch:
    length: int
    values: np.ndarray

    @property
    def count(self) -> int:
        return int(self.values.size)


@dataclass
class PauliBatch:
    n_qubits: int
    signs: np.ndarray
    z_chunks: np.ndarray
    x_chunks: np.ndarray

    @property
    def count(self) -> int:
        return int(self.signs.size)

    @property
    def chunk_count(self) -> int:
        return int(self.z_chunks.shape[1]) if self.z_chunks.ndim == 2 else 0


def save_pauli_data(
    path: Union[str, os.PathLike],
    data: PauliLike,
    *,
    append: bool = False,
    user_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist *data* to ``path``.

    When ``append=True`` new operators are appended without rewriting existing
    records. Concurrent writers using this API serialize creation, header
    validation and complete batch writes with an operating-system file lock.
    An empty or absent file receives a new header. Ordinary saves overwrite.

    Version 1 stores real signs only. Pauli/ZXArray inputs with +/-i phases
    raise ValueError before creating directories or opening the destination.
    """

    path = Path(path)
    representation, batch = _normalise_input(data)
    new_header = _build_header(representation, batch, user_metadata)
    _validate_header_dict(new_header)
    # Check serializability before opening an existing destination for writes.
    json.dumps(new_header, sort_keys=True)
    path.parent.mkdir(parents=True, exist_ok=True)

    # O_APPEND alone cannot protect a multi-write record or first-header race.
    # Opening without truncation lets every writer acquire the same inode lock
    # before inspecting, initializing, appending, or replacing its contents.
    with path.open("a+b") as fh:
        with _exclusive_file_lock(fh):
            fh.seek(0, os.SEEK_END)
            if not append or fh.tell() == 0:
                fh.seek(0)
                fh.truncate(0)
                _write_header(fh, new_header)
            else:
                fh.seek(0)
                header = _read_header_from_stream(fh)
                _validate_header_matches_batch(header, representation, batch)
                fh.seek(0, os.SEEK_END)
            _write_batch_record(fh, representation, batch)


@contextmanager
def _exclusive_file_lock(fh):
    """Hold an OS-level writer lock through flush, including empty files.

    Unix flock locks the inode through this descriptor. Windows locks byte
    zero, including on an empty file; msvcrt retries acquisition for up to
    ten seconds and raises OSError if another writer still owns the lock.
    External writers must cooperate with the same locking convention.
    """
    if os.name == "posix":
        import fcntl

        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            try:
                fh.flush()
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    elif os.name == "nt":  # pragma: no cover - exercised on Windows runners
        import msvcrt

        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
        try:
            yield
        finally:
            try:
                fh.flush()
            finally:
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
    else:  # pragma: no cover - explicit failure instead of unsafe writes
        raise RuntimeError("Safe archive writes require Unix fcntl or Windows msvcrt file locking")


def append_pauli_data(path: Union[str, os.PathLike], data: PauliLike) -> None:
    """Convenience wrapper for ``save_pauli_data(..., append=True)``."""

    save_pauli_data(path, data, append=True)


def load_pauli_data(
    path: Union[str, os.PathLike],
    *,
    include_metadata: bool = False,
    as_zxarray: bool = False,
) -> Union[np.ndarray, PauliIntCollection, ZXArray, Tuple[Union[np.ndarray, PauliIntCollection, ZXArray], Dict[str, Any]]]:
    """Load Pauli data from *path*.

    With ``include_metadata=True`` a ``(data, metadata)`` tuple is returned.
    ``as_zxarray=True`` wraps the result while preserving its stored backend.
    Defaults retain the legacy raw ndarray/PauliIntCollection return types.
    Reads do not acquire a snapshot lock; finish concurrent writes first.
    """

    path = Path(path)
    with path.open("rb") as fh:
        header = _read_header_from_stream(fh)
        fmt = header["format"]
        if fmt == "legacy":
            result = _read_legacy_records(fh, header)
        elif fmt == "pauliint":
            result = _read_pauli_records(fh, header)
        else:  # pragma: no cover - guarded by validation
            raise SerializationError(f"Unknown format '{fmt}'")

    if as_zxarray:
        result = ZXArray.from_raw(result) if fmt == "legacy" else ZXArray.from_collection(result)
    if include_metadata:
        return result, header.get("user_metadata", {})
    return result


def load_legacy_payload(
    path: Union[str, os.PathLike],
    *,
    include_metadata: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
    """Read an opaque historical int64 payload from a legacy archive.

    This explicit compatibility reader preserves the original length-prefixed
    int64 vector, including negative payload values and every payload bit.
    The nonnegative header ``length`` may be any int64 value. It is returned
    as opaque metadata, without interpreting it as a Pauli qubit count.
    Magic, version, record types, checksums, payload dtypes/shapes and counts
    are still validated. ``pauliint`` archives are rejected.

    Use this only before a format-specific historical decoder. The returned
    vector is not necessarily valid input to packed Pauli kernels. Normal
    ``load_pauli_data`` and all writers retain strict Pauli validation.
    ``include_metadata=True`` returns ``(vector, user_metadata)``. As with
    ordinary loads, finish concurrent writes before reading the archive.
    """
    with Path(path).open("rb") as fh:
        header = _read_header_from_stream(fh, opaque_legacy=True)
        result = _read_legacy_records(fh, header, opaque=True)
    if include_metadata:
        return result, header.get("user_metadata", {})
    return result


def iter_pauli_records(path: Union[str, os.PathLike]) -> Iterator[Union[np.ndarray, PauliIntCollection]]:
    """Stream each stored batch without materialising the full dataset."""

    path = Path(path)
    with path.open("rb") as fh:
        header = _read_header_from_stream(fh)
        fmt = header["format"]
        for record, payloads in _record_iterator(fh):
            if fmt == "legacy":
                if record.get("type") != "legacy_batch":
                    raise SerializationError("Unexpected record type in legacy archive")
                yield _legacy_payload_values(header, record, payloads)
            else:
                if record.get("type") != "pauli_batch":
                    raise SerializationError("Unexpected record type in pauli archive")
                yield _collection_from_payload(header, record, payloads)


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _normalise_input(data: PauliLike) -> Tuple[str, Union[LegacyBatch, PauliBatch]]:
    if isinstance(data, Pauli):
        data = data.to_zxarray()
    if is_zxarray(data):
        if np.any(data.phases() & 1):
            raise ValueError("PTSTORE version 1 cannot store +/-i phases; only real Pauli signs are supported")
        if data.is_large:
            return "pauliint", _pauli_batch_from_collection(data.pauliint_collection(copy=False))
        # Raw aliases can have been mutated since wrapper construction.
        array = ZXArray.from_raw(data.legacy_array(copy=False)).legacy_array(copy=False)
        length = int(array[0])
        values = np.ascontiguousarray(array[1:], dtype=np.int64)
        return "legacy", LegacyBatch(length=length, values=values)

    if isinstance(data, np.ndarray):
        array = ZXArray.from_raw(data).legacy_array(copy=False)
        length = int(array[0])
        values = np.ascontiguousarray(array[1:], dtype=np.int64)
        return "legacy", LegacyBatch(length=length, values=values)

    if is_pauliint(data):
        return "pauliint", _pauli_batch_from_sequence([data])

    if is_pauliint_collection(data):
        return "pauliint", _pauli_batch_from_collection(data)

    if isinstance(data, Iterable) and not isinstance(data, (str, bytes, bytearray)):
        seq = list(data)
        if not seq:
            raise ValueError("Cannot serialise an empty iterable of PauliInt instances")
        if not all(is_pauliint(item) for item in seq):
            raise TypeError("Iterable must contain only PauliInt instances")
        return "pauliint", _pauli_batch_from_sequence(seq)  # type: ignore[arg-type]

    raise TypeError(
        "Unsupported data type; expected numpy array, Pauli, ZXArray, PauliInt, PauliIntCollection, or iterable of PauliInt"
    )


def _pauli_batch_from_collection(collection: PauliIntCollection) -> PauliBatch:
    n_qubits = operator.index(collection.n_qubits)
    if n_qubits < 0:
        raise ValueError("PauliIntCollection must have a nonnegative qubit count")
    if len(collection.paulis) > 0:
        batch = _pauli_batch_from_sequence(list(collection.paulis))
        if batch.n_qubits != n_qubits:
            raise ValueError("Collection width does not match its Pauli operators")
        return batch
    prototype = PauliInt.zeros(n_qubits)
    chunk_count = prototype.z_chunks.shape[0]
    return PauliBatch(
        n_qubits=n_qubits,
        signs=np.empty(0, dtype=np.uint8),
        z_chunks=np.empty((0, chunk_count), dtype=np.uint64),
        x_chunks=np.empty((0, chunk_count), dtype=np.uint64),
    )


def _pauli_batch_from_sequence(paulis: Iterable[PauliInt]) -> PauliBatch:
    pauli_list = list(paulis)
    if not pauli_list:
        raise ValueError("Pauli sequence must contain at least one operator")

    n_qubits = operator.index(pauli_list[0].n_qubits)
    if n_qubits < 0:
        raise ValueError("PauliInt instances must have a nonnegative qubit count")
    chunk_count = (n_qubits + 63) // 64

    signs = np.empty(len(pauli_list), dtype=np.uint8)
    z_chunks = np.empty((len(pauli_list), chunk_count), dtype=np.uint64)
    x_chunks = np.empty_like(z_chunks)

    for idx, pauli in enumerate(pauli_list):
        if operator.index(pauli.n_qubits) != n_qubits:
            raise ValueError("All PauliInt instances must share the same number of qubits")
        if pauli.z_chunks.shape != (chunk_count,) or pauli.x_chunks.shape != (chunk_count,):
            raise ValueError("All PauliInt instances must share the same chunk size")
        if pauli.sign not in (0, 1):
            raise ValueError("PauliInt signs must be 0 or 1")
        if pauli.z_chunks.dtype != np.uint64 or pauli.x_chunks.dtype != np.uint64:
            raise TypeError("PauliInt chunk arrays must use uint64")
        # Recheck padding after possible mutation through exposed chunk arrays.
        PauliInt(n_qubits, pauli.sign, pauli.z_chunks, pauli.x_chunks)
        signs[idx] = pauli.sign
        z_chunks[idx] = pauli.z_chunks
        x_chunks[idx] = pauli.x_chunks

    return PauliBatch(n_qubits=n_qubits, signs=signs, z_chunks=z_chunks, x_chunks=x_chunks)


# ---------------------------------------------------------------------------
# File header utilities
# ---------------------------------------------------------------------------

def _build_header(
    representation: str,
    batch: Union[LegacyBatch, PauliBatch],
    user_metadata: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    header: Dict[str, Any] = {"version": CURRENT_VERSION, "format": representation}
    if representation == "legacy":
        header["length"] = int(batch.length)  # type: ignore[union-attr]
    else:
        header["n_qubits"] = int(batch.n_qubits)  # type: ignore[union-attr]
        header["chunk_count"] = int(batch.chunk_count)  # type: ignore[union-attr]
    if user_metadata is not None:
        header["user_metadata"] = user_metadata
    return header


def _write_header(fh, header: Dict[str, Any]) -> None:
    fh.write(MAGIC)
    _write_json_block(fh, header)


def _read_header(path: Path) -> Dict[str, Any]:
    with path.open("rb") as fh:
        return _read_header_from_stream(fh)


def _read_header_from_stream(fh, *, opaque_legacy=False) -> Dict[str, Any]:
    magic = fh.read(len(MAGIC))
    if not magic:
        raise SerializationError("File is empty")
    if magic != MAGIC:
        raise SerializationError("Unrecognised file format")
    header = _read_json_block(fh)
    _validate_header_dict(header, opaque_legacy=opaque_legacy)
    return header


def _validate_header_dict(header: Dict[str, Any], *, opaque_legacy=False) -> None:
    if not isinstance(header, dict):
        raise SerializationError("Archive header must be a JSON object")
    if type(header.get("version")) is not int or header["version"] != CURRENT_VERSION:
        raise SerializationError("Unsupported archive version")
    if "user_metadata" in header and not isinstance(header["user_metadata"], dict):
        raise SerializationError("Archive user_metadata must be an object")
    fmt = header.get("format")
    if opaque_legacy and fmt != "legacy":
        raise SerializationError("load_legacy_payload requires a legacy archive")
    if fmt == "legacy":
        maximum = np.iinfo(np.int64).max if opaque_legacy else 31
        if type(header.get("length")) is not int or not 0 <= header["length"] <= maximum:
            if opaque_legacy:
                raise SerializationError("Opaque legacy archive length must be a nonnegative int64 integer")
            raise SerializationError("Legacy archive length must be an integer in 0..31")
    elif fmt == "pauliint":
        n_qubits, chunk_count = header.get("n_qubits"), header.get("chunk_count")
        if type(n_qubits) is not int or n_qubits < 0:
            raise SerializationError("Pauli archive n_qubits must be a nonnegative integer")
        if type(chunk_count) is not int or chunk_count != (n_qubits + 63) // 64:
            raise SerializationError("Pauli archive chunk_count is inconsistent with n_qubits")
    else:
        raise SerializationError("Unknown archive format")


def _validate_header_matches_batch(
    header: Dict[str, Any],
    representation: str,
    batch: Union[LegacyBatch, PauliBatch],
) -> None:
    if header["format"] != representation:
        raise SerializationError("Data does not match archive representation")
    if representation == "legacy":
        if int(header["length"]) != int(batch.length):  # type: ignore[union-attr]
            raise SerializationError("Pauli length mismatch during append")
    else:
        if int(header["n_qubits"]) != int(batch.n_qubits):  # type: ignore[union-attr]
            raise SerializationError("Qubit count mismatch during append")
        if int(header["chunk_count"]) != int(batch.chunk_count):  # type: ignore[union-attr]
            raise SerializationError("Chunk size mismatch during append")


# ---------------------------------------------------------------------------
# Record encoding/decoding
# ---------------------------------------------------------------------------

def _write_batch_record(fh, representation: str, batch: Union[LegacyBatch, PauliBatch]) -> None:
    if batch.count == 0:
        return
    if representation == "legacy":
        payloads = {"values": _npy_bytes(np.asarray(batch.values, dtype=np.int64))}  # type: ignore[arg-type]
        _write_record(fh, "legacy_batch", batch.count, payloads)
    else:
        payloads = {
            "signs": _npy_bytes(np.asarray(batch.signs, dtype=np.uint8)),  # type: ignore[arg-type]
            "z_chunks": _npy_bytes(np.asarray(batch.z_chunks, dtype=np.uint64)),  # type: ignore[arg-type]
            "x_chunks": _npy_bytes(np.asarray(batch.x_chunks, dtype=np.uint64)),  # type: ignore[arg-type]
        }
        _write_record(fh, "pauli_batch", batch.count, payloads)


def _write_record(fh, record_type: str, count: int, payloads: Dict[str, bytes]) -> None:
    payload_order = list(payloads.keys())
    record = {
        "type": record_type,
        "count": int(count),
        "payload_order": payload_order,
        "payloads": {
            name: {"size": len(blob), "checksum": _checksum(blob)} for name, blob in payloads.items()
        },
    }
    _write_json_block(fh, record)
    for name in payload_order:
        fh.write(payloads[name])


def _record_iterator(fh) -> Iterator[Tuple[Dict[str, Any], Dict[str, bytes]]]:
    while True:
        length_bytes = fh.read(8)
        if not length_bytes:
            return
        if len(length_bytes) != 8:
            raise SerializationError("Corrupted record header")
        json_length = int.from_bytes(length_bytes, "little")
        json_blob = fh.read(json_length)
        if len(json_blob) != json_length:
            raise SerializationError("Unexpected EOF while reading record descriptor")
        record = _decode_json_object(json_blob)
        if type(record.get("count")) is not int or record["count"] < 0:
            raise SerializationError("Record count must be a nonnegative integer")
        order, descriptions = record.get("payload_order"), record.get("payloads")
        if not isinstance(order, list) or not isinstance(descriptions, dict):
            raise SerializationError("Record payload metadata is malformed")
        if any(not isinstance(name, str) for name in order) or len(set(order)) != len(order):
            raise SerializationError("Record payload names must be unique strings")
        if set(order) != set(descriptions):
            raise SerializationError("Record payload order and descriptions do not match")
        payloads: Dict[str, bytes] = {}
        for name in order:
            info = descriptions.get(name)
            if not isinstance(info, dict):
                raise SerializationError(f"Record missing payload metadata for '{name}'")
            size = info.get("size")
            if type(size) is not int or size < 0 or not isinstance(info.get("checksum"), str):
                raise SerializationError(f"Invalid payload size or checksum for '{name}'")
            blob = fh.read(size)
            if len(blob) != size:
                raise SerializationError(f"Unexpected EOF while reading payload '{name}'")
            if info["checksum"] != _checksum(blob):
                raise SerializationError(f"Checksum mismatch for payload '{name}'")
            payloads[name] = blob
        yield record, payloads


def _read_legacy_records(fh, header: Dict[str, Any], *, opaque=False) -> np.ndarray:
    chunks: List[np.ndarray] = []
    for record, payloads in _record_iterator(fh):
        if record.get("type") != "legacy_batch":
            raise SerializationError("Encountered non-legacy record in legacy archive")
        if opaque:
            chunks.append(_legacy_payload_array(record, payloads))
        else:
            chunks.append(_legacy_payload_values(header, record, payloads))
    if chunks:
        values = np.concatenate(chunks)
    else:
        values = np.empty(0, dtype=np.int64)
    output = np.empty(values.size + 1, dtype=np.int64)
    output[0] = int(header["length"])
    output[1:] = values
    return output


def _read_pauli_records(fh, header: Dict[str, Any]) -> PauliIntCollection:
    sign_chunks: List[np.ndarray] = []
    z_chunks_list: List[np.ndarray] = []
    x_chunks_list: List[np.ndarray] = []

    for record, payloads in _record_iterator(fh):
        if record.get("type") != "pauli_batch":
            raise SerializationError("Encountered non-pauli record in pauli archive")
        signs, z_chunks, x_chunks = _pauli_payload_arrays(header, record, payloads)
        sign_chunks.append(signs)
        z_chunks_list.append(z_chunks)
        x_chunks_list.append(x_chunks)

    if sign_chunks:
        signs = np.concatenate(sign_chunks)
        z_chunks = np.concatenate(z_chunks_list)
        x_chunks = np.concatenate(x_chunks_list)
    else:
        signs = np.empty(0, dtype=np.uint8)
        chunk_count = int(header["chunk_count"])
        z_chunks = np.empty((0, chunk_count), dtype=np.uint64)
        x_chunks = np.empty((0, chunk_count), dtype=np.uint64)

    paulis = [
        PauliInt(
            n_qubits=int(header["n_qubits"]),
            sign=int(signs[idx]),
            z_chunks=z_chunks[idx].copy(),
            x_chunks=x_chunks[idx].copy(),
        )
        for idx in range(signs.shape[0])
    ]
    return PauliIntCollection(int(header["n_qubits"]), paulis)


def _collection_from_payload(header, record, payloads) -> PauliIntCollection:
    signs, z_chunks, x_chunks = _pauli_payload_arrays(header, record, payloads)
    paulis = [
        PauliInt(
            n_qubits=int(header["n_qubits"]),
            sign=int(signs[idx]),
            z_chunks=z_chunks[idx].copy(),
            x_chunks=x_chunks[idx].copy(),
        )
        for idx in range(signs.shape[0])
    ]
    return PauliIntCollection(int(header["n_qubits"]), paulis)


def _load_payload_array(payloads, name):
    try:
        array = np.load(io.BytesIO(payloads[name]), allow_pickle=False)
    except (KeyError, ValueError, OSError, EOFError) as exc:
        raise SerializationError(f"Invalid NumPy payload '{name}'") from exc
    if not isinstance(array, np.ndarray):
        raise SerializationError(f"Payload '{name}' must be a NumPy array")
    return array


def _legacy_payload_array(record, payloads):
    """Validate record structure without assigning meaning to its int64 bits."""
    if set(payloads) != {"values"}:
        raise SerializationError("Legacy record requires exactly a values payload")
    values = _load_payload_array(payloads, "values")
    if values.dtype.kind != "i" or values.dtype.itemsize != 8 or values.shape != (record["count"],):
        raise SerializationError("Legacy payload dtype, shape, or count is inconsistent")
    return values.astype(np.int64, copy=False)


def _legacy_payload_values(header, record, payloads):
    values = _legacy_payload_array(record, payloads)
    packed = np.empty(values.size + 1, dtype=np.int64)
    packed[0], packed[1:] = header["length"], values
    try:
        ZXArray.from_raw(packed)
    except (TypeError, ValueError) as exc:
        raise SerializationError("Legacy payload contains invalid packed Pauli values") from exc
    return values


def _pauli_payload_arrays(header, record, payloads):
    if set(payloads) != {"signs", "z_chunks", "x_chunks"}:
        raise SerializationError("Pauli record requires signs, z_chunks, and x_chunks payloads")
    signs = _load_payload_array(payloads, "signs")
    z_chunks = _load_payload_array(payloads, "z_chunks")
    x_chunks = _load_payload_array(payloads, "x_chunks")
    shape = (record["count"], header["chunk_count"])
    if (signs.dtype != np.uint8 or signs.shape != (record["count"],)
            or z_chunks.dtype.kind != "u" or z_chunks.dtype.itemsize != 8
            or x_chunks.dtype.kind != "u" or x_chunks.dtype.itemsize != 8
            or z_chunks.shape != shape or x_chunks.shape != shape):
        raise SerializationError("Pauli payload dtype, shape, or count is inconsistent")
    z_chunks = z_chunks.astype(np.uint64, copy=False)
    x_chunks = x_chunks.astype(np.uint64, copy=False)
    if np.any(signs > 1):
        raise SerializationError("Pauli sign payload must contain only 0/1 values")
    tail = header["n_qubits"] % 64
    if tail:
        unused = ~((np.uint64(1) << np.uint64(tail)) - np.uint64(1))
        if np.any(z_chunks[:, -1] & unused) or np.any(x_chunks[:, -1] & unused):
            raise SerializationError("Pauli payload has bits beyond its declared qubit width")
    return signs, z_chunks, x_chunks


def _write_json_block(fh, data: Dict[str, Any]) -> None:
    blob = json.dumps(data, sort_keys=True).encode("utf-8")
    fh.write(len(blob).to_bytes(8, "little"))
    fh.write(blob)


def _read_json_block(fh) -> Dict[str, Any]:
    size_bytes = fh.read(8)
    if len(size_bytes) != 8:
        raise SerializationError("Failed to read JSON block length")
    size = int.from_bytes(size_bytes, "little")
    blob = fh.read(size)
    if len(blob) != size:
        raise SerializationError("Unexpected EOF while reading JSON block")
    return _decode_json_object(blob)


def _decode_json_object(blob: bytes) -> Dict[str, Any]:
    try:
        value = json.loads(blob.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SerializationError("Invalid archive JSON") from exc
    if not isinstance(value, dict):
        raise SerializationError("Archive JSON blocks must be objects")
    return value


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def _checksum(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
