"""Canonical, immutable, content-addressed artifacts.

The artifact value domain is an intentional subset of RFC 8785: null, booleans,
Unicode strings, arrays, string-keyed objects, and integers exactly representable
by IEEE-754 binary64. Binary floats are rejected. Schemas represent fractional
values as fixed-point integers or validated decimal strings until a complete,
separately tested ECMAScript number serializer is introduced. Within this value
domain the emitted bytes use RFC 8785 string escaping and UTF-16 key ordering.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias, cast
from uuid import uuid4

JsonScalar: TypeAlias = None | bool | int | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
MAX_SAFE_INTEGER = 9_007_199_254_740_991
_SCHEMA_VERSION = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
_HASH = re.compile(r"^[0-9a-f]{64}$")


class ArtifactError(RuntimeError):
    """Base class for fail-closed artifact errors."""


class CanonicalizationError(ArtifactError):
    """A value is outside the deliberately restricted canonical JSON domain."""


class ArtifactCorruption(ArtifactError):
    """Stored bytes do not match their immutable content identity."""


class UnknownSchemaMajor(ArtifactError):
    """An artifact uses a schema major this reader does not support."""


class ImmutableArtifactConflict(ArtifactError):
    """A content-addressed path already contains different bytes."""


class ArtifactDurabilityError(ArtifactError):
    """A filesystem durability barrier failed and publication is not trusted."""


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sync_directory(path: Path) -> None:
    try:
        _fsync_directory(path)
    except OSError as error:
        raise ArtifactDurabilityError(f"directory fsync failed for {path}") from error


def _validate_string(value: str) -> None:
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise CanonicalizationError("Unicode surrogate code points are not valid artifact strings")


def _string_bytes(value: str) -> bytes:
    _validate_string(value)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _canonical(value: object) -> bytes:
    if value is None:
        return b"null"
    if value is True:
        return b"true"
    if value is False:
        return b"false"
    if isinstance(value, int):
        if not -MAX_SAFE_INTEGER <= value <= MAX_SAFE_INTEGER:
            raise CanonicalizationError(
                "artifact numeric values must be safe integers; use fixed-point or a decimal string"
            )
        return str(value).encode("ascii")
    if isinstance(value, float):
        raise CanonicalizationError(
            "binary floating-point numeric values are unsupported; use fixed-point or a decimal string"
        )
    if isinstance(value, str):
        return _string_bytes(value)
    if isinstance(value, Mapping):
        items: list[tuple[str, object]] = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalizationError("canonical JSON object keys must be strings")
            _validate_string(key)
            items.append((key, item))
        items.sort(key=lambda item: item[0].encode("utf-16-be"))
        encoded = [(_string_bytes(key) + b":" + _canonical(item)) for key, item in items]
        return b"{" + b",".join(encoded) + b"}"
    if isinstance(value, list):
        return b"[" + b",".join(_canonical(item) for item in value) + b"]"
    raise CanonicalizationError(f"unsupported artifact value type: {type(value).__name__}")


def canonical_json_bytes(value: object) -> bytes:
    """Return deterministic UTF-8 bytes for the supported RFC 8785 value domain."""

    return _canonical(value)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _schema_major(version: str) -> int:
    match = _SCHEMA_VERSION.fullmatch(version)
    if match is None:
        raise ArtifactCorruption(f"invalid schema version: {version!r}")
    return int(match.group(1))


@dataclass(frozen=True, slots=True)
class Artifact:
    content_hash: str
    schema_name: str
    schema_version: str
    _payload_bytes: bytes
    path: Path
    raw_bytes: bytes
    digest_bytes: bytes

    @property
    def payload(self) -> dict[str, JsonValue]:
        """Return an isolated view decoded from immutable canonical payload bytes."""

        decoded = json.loads(self._payload_bytes.decode("utf-8"))
        if not isinstance(decoded, dict):
            raise ArtifactCorruption("artifact payload snapshot is not an object")
        return cast(dict[str, JsonValue], decoded)


class ArtifactStore:
    """Local fixture store with atomic immutable publication by content hash."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.artifact_dir = self.root / "artifacts"
        self.blob_dir = self.root / "blobs"
        self.durable_mkdir(self.artifact_dir)
        self.durable_mkdir(self.blob_dir)

    def put(self, schema_name: str, schema_version: str, payload: Mapping[str, object]) -> Artifact:
        _validate_string(schema_name)
        _schema_major(schema_version)
        payload_bytes = canonical_json_bytes(dict(payload))
        normalized_payload = cast(
            dict[str, JsonValue],
            json.loads(payload_bytes.decode("utf-8")),
        )
        body: dict[str, object] = {
            "payload": normalized_payload,
            "schema_name": schema_name,
            "schema_version": schema_version,
        }
        digest_bytes = canonical_json_bytes(body)
        content_hash = sha256_hex(digest_bytes)
        envelope = dict(body)
        envelope["content_hash"] = content_hash
        raw_bytes = canonical_json_bytes(envelope)
        path = self.artifact_dir / f"{content_hash}.json"
        self._publish(path, raw_bytes)
        return Artifact(
            content_hash=content_hash,
            schema_name=schema_name,
            schema_version=schema_version,
            _payload_bytes=payload_bytes,
            path=path,
            raw_bytes=raw_bytes,
            digest_bytes=digest_bytes,
        )

    def put_blob(self, data: bytes) -> tuple[str, int]:
        """Publish exact boundary bytes under their SHA-256 identity."""

        content_hash = sha256_hex(data)
        self._publish(self.blob_dir / f"{content_hash}.blob", data)
        return content_hash, len(data)

    def read_blob(self, content_hash: str, *, expected_size: int) -> bytes:
        if _HASH.fullmatch(content_hash) is None:
            raise ArtifactCorruption("invalid blob content hash")
        try:
            data = (self.blob_dir / f"{content_hash}.blob").read_bytes()
        except OSError as error:
            raise ArtifactCorruption(f"blob payload is unavailable: {content_hash}") from error
        if len(data) != expected_size:
            raise ArtifactCorruption("blob size does not match its manifest")
        if sha256_hex(data) != content_hash:
            raise ArtifactCorruption("blob content hash does not match payload")
        return data

    def read(
        self,
        content_hash: str,
        *,
        expected_schema_name: str | None = None,
        supported_major: int = 1,
    ) -> Artifact:
        if _HASH.fullmatch(content_hash) is None:
            raise ArtifactCorruption("invalid artifact content hash")
        path = self.artifact_dir / f"{content_hash}.json"
        try:
            raw_bytes = path.read_bytes()
        except OSError as error:
            raise ArtifactCorruption(f"artifact payload is unavailable: {content_hash}") from error
        try:
            decoded = json.loads(raw_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ArtifactCorruption("artifact is not valid UTF-8 JSON") from error
        if not isinstance(decoded, dict) or set(decoded) != {
            "content_hash",
            "payload",
            "schema_name",
            "schema_version",
        }:
            raise ArtifactCorruption("artifact envelope fields are invalid")
        stored_hash = decoded["content_hash"]
        schema_name = decoded["schema_name"]
        schema_version = decoded["schema_version"]
        payload = decoded["payload"]
        if not all(isinstance(item, str) for item in (stored_hash, schema_name, schema_version)):
            raise ArtifactCorruption("artifact identity fields are invalid")
        if not isinstance(payload, dict):
            raise ArtifactCorruption("artifact payload must be an object")
        body = {
            "payload": payload,
            "schema_name": schema_name,
            "schema_version": schema_version,
        }
        try:
            digest_bytes = canonical_json_bytes(body)
            canonical_envelope = canonical_json_bytes(decoded)
        except CanonicalizationError as error:
            raise ArtifactCorruption("artifact contains non-canonical values") from error
        calculated_hash = sha256_hex(digest_bytes)
        if raw_bytes != canonical_envelope:
            raise ArtifactCorruption("artifact bytes are not canonical JSON")
        if stored_hash != content_hash or calculated_hash != content_hash:
            raise ArtifactCorruption("artifact content hash does not match payload")
        if expected_schema_name is not None and schema_name != expected_schema_name:
            raise ArtifactCorruption(f"expected schema {expected_schema_name!r}, found {schema_name!r}")
        if _schema_major(schema_version) != supported_major:
            raise UnknownSchemaMajor(
                f"unsupported schema major for {schema_name}: {schema_version}; expected {supported_major}.x"
            )
        return Artifact(
            content_hash=content_hash,
            schema_name=schema_name,
            schema_version=schema_version,
            _payload_bytes=canonical_json_bytes(payload),
            path=path,
            raw_bytes=raw_bytes,
            digest_bytes=digest_bytes,
        )

    @staticmethod
    def _publish(path: Path, data: bytes) -> None:
        ArtifactStore.durable_mkdir(path.parent)
        temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        temporary_exists = True
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                try:
                    existing = path.read_bytes()
                except OSError as error:
                    raise ImmutableArtifactConflict(f"cannot verify existing artifact {path}") from error
                if existing != data:
                    raise ImmutableArtifactConflict(f"immutable artifact conflict at {path}") from None
            else:
                _sync_directory(path.parent)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                temporary_exists = False
            except OSError as error:
                raise ArtifactDurabilityError(f"temporary publication cleanup failed in {path.parent}") from error
            if temporary_exists:
                _sync_directory(path.parent)

    @staticmethod
    def durable_mkdir(path: str | Path) -> None:
        """Create every missing path component with durable directory entries."""

        target = Path(path)
        missing: list[Path] = []
        cursor = target
        while not cursor.exists():
            missing.append(cursor)
            parent = cursor.parent
            if parent == cursor:
                break
            cursor = parent
        if cursor.exists() and not cursor.is_dir():
            raise ArtifactDurabilityError(f"directory parent is not a directory: {cursor}")
        for directory in reversed(missing):
            try:
                directory.mkdir()
            except FileExistsError:
                if not directory.is_dir():
                    raise ArtifactDurabilityError(f"durable directory path is not a directory: {directory}") from None
                continue
            except OSError as error:
                raise ArtifactDurabilityError(f"durable directory creation failed for {directory}") from error
            _sync_directory(directory)
            _sync_directory(directory.parent)

    @staticmethod
    def durable_touch(path: str | Path) -> None:
        """Create a durable empty coordination file if it is absent."""

        target = Path(path)
        ArtifactStore.durable_mkdir(target.parent)
        if target.exists():
            if not target.is_file():
                raise ArtifactDurabilityError(f"coordination path is not a file: {target}")
            return
        try:
            descriptor = os.open(target, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError as error:
            raise ArtifactDurabilityError(f"coordination file creation failed for {target}") from error
        _sync_directory(target.parent)
