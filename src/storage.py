"""Append-friendly storage for Pauli data."""

from __future__ import annotations

import io
import json
import os
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union

import numpy as np

try:  # pragma: no cover - handled during package import
    from .large_pauli import (
        PauliInt,
        PauliIntCollection,
        is_pauliint,
        is_pauliint_collection,
    )
except ImportError:  # pragma: no cover - legacy import path
    from large_pauli import (  # type: ignore
        PauliInt,
        PauliIntCollection,
        is_pauliint,
        is_pauliint_collection,
    )

PauliLike = Union[np.ndarray, PauliInt, PauliIntCollection, Iterable[PauliInt]]

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
    records.  If the file does not exist yet, the call behaves like a fresh
    write regardless of ``append``.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    representation, batch = _normalise_input(data)

    if append and not path.exists():
        append = False

    if not append:
        header = _build_header(representation, batch, user_metadata)
        with path.open("wb") as fh:
            _write_header(fh, header)
            _write_batch_record(fh, representation, batch)
        return

    header = _read_header(path)
    _validate_header_matches_batch(header, representation, batch)
    if batch.count == 0:
        return
    with path.open("r+b") as fh:
        _read_header_from_stream(fh)
        fh.seek(0, os.SEEK_END)
        _write_batch_record(fh, header["format"], batch)


def append_pauli_data(path: Union[str, os.PathLike], data: PauliLike) -> None:
    """Convenience wrapper for ``save_pauli_data(..., append=True)``."""

    save_pauli_data(path, data, append=True)


def load_pauli_data(
    path: Union[str, os.PathLike],
    *,
    include_metadata: bool = False,
) -> Union[np.ndarray, PauliIntCollection, Tuple[Union[np.ndarray, PauliIntCollection], Dict[str, Any]]]:
    """Load Pauli data from *path*.

    With ``include_metadata=True`` a ``(data, metadata)`` tuple is returned.
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
                if record["type"] != "legacy_batch":
                    raise SerializationError("Unexpected record type in legacy archive")
                arr = np.load(io.BytesIO(payloads["values"]), allow_pickle=False)
                yield arr.astype(np.int64, copy=False)
            else:
                if record["type"] != "pauli_batch":
                    raise SerializationError("Unexpected record type in pauli archive")
                yield _collection_from_payload(header, payloads)


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _normalise_input(data: PauliLike) -> Tuple[str, Union[LegacyBatch, PauliBatch]]:
    if isinstance(data, np.ndarray):
        array = np.asarray(data, dtype=np.int64)
        if array.ndim != 1 or array.size == 0:
            raise TypeError("Legacy ZX arrays must be one-dimensional and non-empty")
        length = int(array[0])
        values = np.ascontiguousarray(array[1:], dtype=np.int64)
        return "legacy", LegacyBatch(length=length, values=values)

    if is_pauliint(data):
        return "pauliint", _pauli_batch_from_sequence([data])

    if is_pauliint_collection(data):
        return "pauliint", _pauli_batch_from_sequence(list(data.paulis))

    if isinstance(data, Iterable) and not isinstance(data, (str, bytes, bytearray)):
        seq = list(data)
        if not seq:
            raise ValueError("Cannot serialise an empty iterable of PauliInt instances")
        if not all(is_pauliint(item) for item in seq):
            raise TypeError("Iterable must contain only PauliInt instances")
        return "pauliint", _pauli_batch_from_sequence(seq)  # type: ignore[arg-type]

    raise TypeError(
        "Unsupported data type; expected numpy array, PauliInt, PauliIntCollection, or iterable of PauliInt"
    )


def _pauli_batch_from_sequence(paulis: Iterable[PauliInt]) -> PauliBatch:
    pauli_list = list(paulis)
    if not pauli_list:
        raise ValueError("Pauli sequence must contain at least one operator")

    n_qubits = pauli_list[0].n_qubits
    chunk_count = pauli_list[0].z_chunks.shape[0]

    signs = np.empty(len(pauli_list), dtype=np.uint8)
    z_chunks = np.empty((len(pauli_list), chunk_count), dtype=np.uint64)
    x_chunks = np.empty_like(z_chunks)

    for idx, pauli in enumerate(pauli_list):
        if pauli.n_qubits != n_qubits:
            raise ValueError("All PauliInt instances must share the same number of qubits")
        if pauli.z_chunks.shape != (chunk_count,) or pauli.x_chunks.shape != (chunk_count,):
            raise ValueError("All PauliInt instances must share the same chunk size")
        signs[idx] = np.uint8(pauli.sign & 1)
        z_chunks[idx] = np.asarray(pauli.z_chunks, dtype=np.uint64)
        x_chunks[idx] = np.asarray(pauli.x_chunks, dtype=np.uint64)

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
    if user_metadata:
        header["user_metadata"] = user_metadata
    return header


def _write_header(fh, header: Dict[str, Any]) -> None:
    fh.write(MAGIC)
    _write_json_block(fh, header)


def _read_header(path: Path) -> Dict[str, Any]:
    with path.open("rb") as fh:
        return _read_header_from_stream(fh)


def _read_header_from_stream(fh) -> Dict[str, Any]:
    magic = fh.read(len(MAGIC))
    if not magic:
        raise SerializationError("File is empty")
    if magic != MAGIC:
        raise SerializationError("Unrecognised file format")
    header = _read_json_block(fh)
    _validate_header_dict(header)
    return header


def _validate_header_dict(header: Dict[str, Any]) -> None:
    if header.get("version") != CURRENT_VERSION:
        raise SerializationError("Unsupported archive version")
    fmt = header.get("format")
    if fmt == "legacy":
        if "length" not in header:
            raise SerializationError("Legacy archive missing 'length'")
    elif fmt == "pauliint":
        if "n_qubits" not in header or "chunk_count" not in header:
            raise SerializationError("Pauli archive missing 'n_qubits' or 'chunk_count'")
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
        record = json.loads(json_blob.decode("utf-8"))
        payloads: Dict[str, bytes] = {}
        for name in record.get("payload_order", []):
            info = record["payloads"].get(name)
            if info is None:
                raise SerializationError(f"Record missing payload metadata for '{name}'")
            size = int(info["size"])
            blob = fh.read(size)
            if len(blob) != size:
                raise SerializationError(f"Unexpected EOF while reading payload '{name}'")
            if info["checksum"] != _checksum(blob):
                raise SerializationError(f"Checksum mismatch for payload '{name}'")
            payloads[name] = blob
        yield record, payloads


def _read_legacy_records(fh, header: Dict[str, Any]) -> np.ndarray:
    chunks: List[np.ndarray] = []
    for record, payloads in _record_iterator(fh):
        if record["type"] != "legacy_batch":
            raise SerializationError("Encountered non-legacy record in legacy archive")
        arr = np.load(io.BytesIO(payloads["values"]), allow_pickle=False)
        chunks.append(np.asarray(arr, dtype=np.int64))
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
        if record["type"] != "pauli_batch":
            raise SerializationError("Encountered non-pauli record in pauli archive")
        sign_chunks.append(np.load(io.BytesIO(payloads["signs"]), allow_pickle=False).astype(np.uint8, copy=False))
        z_chunks_list.append(np.load(io.BytesIO(payloads["z_chunks"]), allow_pickle=False).astype(np.uint64, copy=False))
        x_chunks_list.append(np.load(io.BytesIO(payloads["x_chunks"]), allow_pickle=False).astype(np.uint64, copy=False))

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


def _collection_from_payload(header: Dict[str, Any], payloads: Dict[str, bytes]) -> PauliIntCollection:
    signs = np.load(io.BytesIO(payloads["signs"]), allow_pickle=False).astype(np.uint8, copy=False)
    z_chunks = np.load(io.BytesIO(payloads["z_chunks"]), allow_pickle=False).astype(np.uint64, copy=False)
    x_chunks = np.load(io.BytesIO(payloads["x_chunks"]), allow_pickle=False).astype(np.uint64, copy=False)
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
    return json.loads(blob.decode("utf-8"))


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def _checksum(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
