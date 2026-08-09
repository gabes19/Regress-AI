"""Store and resolve uploaded datasets without trusting client file paths."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
import uuid

from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename


DATASET_ID_PATTERN = re.compile(r"[0-9a-f]{32}")


class DatasetError(ValueError):
    """Base exception for invalid or unavailable datasets."""


class DatasetNotFoundError(DatasetError):
    """Raised when a valid dataset ID has no stored dataset."""


@dataclass(frozen=True)
class StoredDataset:
    """Trusted server-side dataset reference."""

    dataset_id: str
    original_filename: str
    storage_path: Path


def validate_dataset_id(dataset_id: str | None) -> str:
    """Accept only lowercase 32-hex IDs, never paths or filenames."""
    if not DATASET_ID_PATTERN.fullmatch(dataset_id or ""):
        raise DatasetNotFoundError("Dataset not found.")
    return str(dataset_id)


def _dataset_paths(
    dataset_id: str,
    upload_folder: str | Path,
) -> tuple[Path, Path]:
    trusted_id = validate_dataset_id(dataset_id)
    upload_root = Path(upload_folder).resolve()
    csv_path = (upload_root / f"{trusted_id}.csv").resolve()
    metadata_path = (upload_root / f"{trusted_id}.json").resolve()

    if csv_path.parent != upload_root or metadata_path.parent != upload_root:
        raise DatasetNotFoundError("Dataset not found.")

    return csv_path, metadata_path


def _safe_original_filename(filename: str | None) -> str:
    safe_name = secure_filename(filename or "")
    if not safe_name:
        raise DatasetError("Please choose a CSV file.")
    if Path(safe_name).suffix.casefold() != ".csv":
        raise DatasetError("Please upload a CSV file.")
    return safe_name


def _write_metadata(
    metadata_path: Path,
    dataset_id: str,
    original_filename: str,
) -> None:
    metadata = {
        "dataset_id": dataset_id,
        "original_filename": original_filename,
    }
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, separators=(",", ":"))


def store_uploaded_dataset(
    uploaded_file: FileStorage,
    upload_folder: str | Path,
) -> StoredDataset:
    """Save an uploaded CSV under an opaque server-generated ID."""
    original_filename = _safe_original_filename(uploaded_file.filename)
    dataset_id = uuid.uuid4().hex
    csv_path, metadata_path = _dataset_paths(dataset_id, upload_folder)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        uploaded_file.save(csv_path)
        _write_metadata(
            metadata_path,
            dataset_id,
            original_filename,
        )
    except Exception:
        csv_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
        raise

    return StoredDataset(
        dataset_id=dataset_id,
        original_filename=original_filename,
        storage_path=csv_path,
    )


def store_existing_dataset(
    source_path: str | Path,
    upload_folder: str | Path,
    original_filename: str | None = None,
) -> StoredDataset:
    """Copy a trusted bundled dataset into the ID-based upload lifecycle."""
    source = Path(source_path).resolve()
    if not source.is_file():
        raise DatasetNotFoundError("Sample dataset not found.")

    display_name = _safe_original_filename(
        original_filename or source.name
    )
    dataset_id = uuid.uuid4().hex
    csv_path, metadata_path = _dataset_paths(dataset_id, upload_folder)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        shutil.copyfile(source, csv_path)
        _write_metadata(metadata_path, dataset_id, display_name)
    except Exception:
        csv_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
        raise

    return StoredDataset(
        dataset_id=dataset_id,
        original_filename=display_name,
        storage_path=csv_path,
    )


def load_dataset(
    dataset_id: str | None,
    upload_folder: str | Path,
) -> StoredDataset:
    """Resolve a dataset ID to a trusted path and display filename."""
    trusted_id = validate_dataset_id(dataset_id)
    csv_path, metadata_path = _dataset_paths(trusted_id, upload_folder)

    if not csv_path.is_file() or not metadata_path.is_file():
        raise DatasetNotFoundError("Dataset not found.")

    try:
        with metadata_path.open("r", encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
        original_filename = _safe_original_filename(
            metadata.get("original_filename")
        )
    except (OSError, json.JSONDecodeError, AttributeError, DatasetError) as error:
        raise DatasetNotFoundError("Dataset metadata is invalid.") from error

    if metadata.get("dataset_id") != trusted_id:
        raise DatasetNotFoundError("Dataset metadata is invalid.")

    return StoredDataset(
        dataset_id=trusted_id,
        original_filename=original_filename,
        storage_path=csv_path,
    )


def delete_dataset(
    dataset_id: str,
    upload_folder: str | Path,
) -> None:
    """Remove a newly stored dataset when intake validation fails."""
    csv_path, metadata_path = _dataset_paths(dataset_id, upload_folder)
    csv_path.unlink(missing_ok=True)
    metadata_path.unlink(missing_ok=True)
