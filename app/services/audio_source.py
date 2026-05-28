from __future__ import annotations

import base64
from dataclasses import dataclass
from hashlib import sha256
from mimetypes import guess_extension
from os import getenv
from pathlib import Path
from urllib.error import URLError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

from app.schemas.analyze import AnalyzeEvidenceRequest

DEFAULT_MAX_AUDIO_SOURCE_BYTES = 25 * 1024 * 1024
DEFAULT_FETCH_TIMEOUT_SECONDS = 10


@dataclass(frozen=True)
class AudioSource:
    data: bytes
    filename: str
    content_type: str
    source_kind: str


class AudioSourceError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def resolve_audio_source(payload: AnalyzeEvidenceRequest) -> AudioSource | None:
    reference = payload.storage_reference

    if not reference:
        return None

    if reference.startswith("data:"):
        return _resolve_data_url(payload, reference)

    parsed = urlparse(reference)

    if parsed.scheme == "file":
        return _resolve_file_reference(payload, parsed.path)

    if parsed.scheme in {"http", "https"}:
        return _resolve_url_reference(payload, reference, parsed.scheme, parsed.hostname)

    raise AudioSourceError("unsupported_storage_reference")


def _resolve_data_url(
    payload: AnalyzeEvidenceRequest,
    reference: str,
) -> AudioSource:
    try:
        header, encoded = reference.split(",", 1)
    except ValueError as error:
        raise AudioSourceError("invalid_data_url") from error

    if ";base64" not in header:
        raise AudioSourceError("unsupported_data_url_encoding")

    content_type = header.removeprefix("data:").split(";", 1)[0]
    if content_type and not content_type.lower().startswith("audio/"):
        raise AudioSourceError("data_url_not_audio")

    try:
        data = base64.b64decode(encoded, validate=True)
    except ValueError as error:
        raise AudioSourceError("invalid_base64_audio") from error

    _validate_audio_bytes(payload, data)

    return AudioSource(
        data=data,
        filename=_build_filename(payload, payload.mime_type),
        content_type=payload.mime_type,
        source_kind="data_url",
    )


def _resolve_file_reference(
    payload: AnalyzeEvidenceRequest,
    path: str,
) -> AudioSource:
    if not _env_bool("AI_SERVICE_ALLOW_FILE_REFERENCES"):
        raise AudioSourceError("file_references_disabled")

    file_path = Path(unquote(path)).expanduser().resolve()

    if not file_path.is_file():
        raise AudioSourceError("audio_file_not_found")

    data = file_path.read_bytes()
    _validate_audio_bytes(payload, data)

    return AudioSource(
        data=data,
        filename=file_path.name or _build_filename(payload, payload.mime_type),
        content_type=payload.mime_type,
        source_kind="file",
    )


def _resolve_url_reference(
    payload: AnalyzeEvidenceRequest,
    reference: str,
    scheme: str,
    hostname: str | None,
) -> AudioSource:
    if scheme == "http" and not _env_bool("AI_SERVICE_ALLOW_INSECURE_AUDIO_REFERENCES"):
        raise AudioSourceError("insecure_audio_reference")

    _validate_allowed_host(hostname)

    request = Request(
        reference,
        headers={
            "Accept": payload.mime_type,
            "User-Agent": "cicla-vera-ai-service/0.1",
        },
    )

    max_bytes = _get_max_audio_source_bytes()

    try:
        with urlopen(request, timeout=_get_fetch_timeout_seconds()) as response:
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > max_bytes:
                raise AudioSourceError("audio_source_too_large")

            data = response.read(max_bytes + 1)
    except AudioSourceError:
        raise
    except (OSError, URLError, ValueError) as error:
        raise AudioSourceError("audio_source_fetch_failed") from error

    _validate_audio_bytes(payload, data)

    parsed = urlparse(reference)
    filename = Path(unquote(parsed.path)).name or _build_filename(
        payload,
        payload.mime_type,
    )

    return AudioSource(
        data=data,
        filename=filename,
        content_type=payload.mime_type,
        source_kind="url",
    )


def _validate_audio_bytes(payload: AnalyzeEvidenceRequest, data: bytes) -> None:
    if not data:
        raise AudioSourceError("audio_source_empty")

    if len(data) > _get_max_audio_source_bytes():
        raise AudioSourceError("audio_source_too_large")

    if len(data) != payload.size:
        raise AudioSourceError("audio_size_mismatch")

    digest = sha256(data).hexdigest()
    if digest.lower() != payload.content_hash.lower():
        raise AudioSourceError("content_hash_mismatch")


def _validate_allowed_host(hostname: str | None) -> None:
    allowed_hosts = {
        item.strip().lower()
        for item in getenv("AI_SERVICE_ALLOWED_AUDIO_HOSTS", "").split(",")
        if item.strip()
    }

    if allowed_hosts and (not hostname or hostname.lower() not in allowed_hosts):
        raise AudioSourceError("audio_source_host_not_allowed")


def _build_filename(payload: AnalyzeEvidenceRequest, content_type: str) -> str:
    extension = guess_extension(content_type) or ".audio"
    return f"{payload.evidence_record_id}{extension}"


def _get_max_audio_source_bytes() -> int:
    return _env_int("AI_SERVICE_MAX_AUDIO_SOURCE_BYTES", DEFAULT_MAX_AUDIO_SOURCE_BYTES)


def _get_fetch_timeout_seconds() -> int:
    return _env_int("AI_SERVICE_AUDIO_FETCH_TIMEOUT_SECONDS", DEFAULT_FETCH_TIMEOUT_SECONDS)


def _env_int(name: str, default: int) -> int:
    raw = getenv(name)

    if not raw:
        return default

    try:
        value = int(raw)
    except ValueError:
        return default

    return value if value > 0 else default


def _env_bool(name: str) -> bool:
    return getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}
