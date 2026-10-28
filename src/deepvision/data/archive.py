"""Safe extraction of the Intel Image Classification archive.

The archive is verified against a SHA-256 checksum and every entry is validated
before anything is written. Files are extracted into a temporary directory next
to the destination, which is renamed into place only after extraction succeeds.
The nested Kaggle layout is normalized on the way out:

    seg_train/seg_train/<class>/<file>  ->  train/<class>/<file>
    seg_test/seg_test/<class>/<file>    ->  test/<class>/<file>
    seg_pred/seg_pred/<file>            ->  pred/<file>

Usage:
    python -m deepvision.data.archive --archive <zip> --output <dir>
"""

import argparse
import hashlib
import re
import shutil
import stat
import sys
import tempfile
import zipfile
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO

EXPECTED_SHA256 = "3921405607c3e9d67a3459b6c629894c3a723aa1647f9310b6793da306e89337"
CLASSES = ("buildings", "forest", "glacier", "mountain", "sea", "street")
ALLOWED_SUFFIXES = frozenset({".jpg"})
LABELED_SPLITS = {"seg_train": "train", "seg_test": "test"}
UNLABELED_SPLITS = {"seg_pred": "pred"}

DEFAULT_ARCHIVE = Path("data/raw/intel-image-classification.zip")
DEFAULT_OUTPUT = Path("data/interim/intel")

_CHUNK_SIZE = 1 << 20
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_DRIVE_RE = re.compile(r"^[A-Za-z]:")
# Characters that are invalid in Windows file names; ":" would also address NTFS data streams.
_FORBIDDEN_CHARS = frozenset('<>:"|?*\\')
_WINDOWS_RESERVED = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)


class ArchiveError(Exception):
    """Raised when an archive fails verification or cannot be extracted safely."""


@dataclass(frozen=True)
class SafetyLimits:
    """Upper bounds that reject archives far outside the expected dataset shape."""

    max_entries: int = 30_000
    max_file_bytes: int = 5 * 2**20
    max_total_bytes: int = 2**30
    max_compression_ratio: float = 100.0


@dataclass(frozen=True)
class ExtractionResult:
    output_dir: Path
    sha256: str
    counts: dict[str, int]
    total_files: int
    total_bytes: int


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of a file."""
    with Path(path).open("rb") as stream:
        return _sha256_stream(stream)


def _sha256_stream(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    while chunk := stream.read(_CHUNK_SIZE):
        digest.update(chunk)
    return digest.hexdigest()


def _normalize_checksum(expected: str) -> str:
    value = expected.strip().lower()
    if not _SHA256_RE.fullmatch(value):
        raise ArchiveError(f"expected SHA-256 is not 64 hex characters: {expected!r}")
    return value


def safe_parts(name: str) -> tuple[str, ...]:
    """Split an entry name into path components, rejecting anything unsafe or non-portable."""
    if name.startswith("/") or _DRIVE_RE.match(name):
        raise ArchiveError(f"absolute path in entry {name!r}")
    parts = tuple(name.removesuffix("/").split("/"))
    for part in parts:
        if part in ("", ".", ".."):
            raise ArchiveError(f"unsafe path component {part!r} in entry {name!r}")
        if _FORBIDDEN_CHARS.intersection(part) or any(ord(c) < 32 for c in part):
            raise ArchiveError(f"forbidden character in entry {name!r}")
        if part.endswith((" ", ".")) or part.split(".")[0].upper() in _WINDOWS_RESERVED:
            raise ArchiveError(f"name is not portable to Windows in entry {name!r}")
    return parts


def map_entry(
    name: str,
    classes: tuple[str, ...] = CLASSES,
    allowed_suffixes: frozenset[str] = ALLOWED_SUFFIXES,
) -> PurePosixPath:
    """Map an archive file entry to its normalized path relative to the output directory."""
    parts = safe_parts(name)
    if PurePosixPath(parts[-1]).suffix.lower() not in allowed_suffixes:
        raise ArchiveError(f"unexpected file type in entry {name!r}")
    match parts:
        case (split, inner, label, filename) if split in LABELED_SPLITS and inner == split:
            if label not in classes:
                raise ArchiveError(f"unknown class {label!r} in entry {name!r}")
            return PurePosixPath(LABELED_SPLITS[split], label, filename)
        case (split, inner, filename) if split in UNLABELED_SPLITS and inner == split:
            return PurePosixPath(UNLABELED_SPLITS[split], filename)
    raise ArchiveError(f"entry {name!r} does not match the expected layout")


def plan_extraction(
    zf: zipfile.ZipFile,
    classes: tuple[str, ...] = CLASSES,
    limits: SafetyLimits | None = None,
    allowed_suffixes: frozenset[str] = ALLOWED_SUFFIXES,
) -> list[tuple[zipfile.ZipInfo, PurePosixPath]]:
    """Validate every entry and return (entry, target) pairs sorted by target path."""
    limits = limits or SafetyLimits()
    infos = zf.infolist()
    if len(infos) > limits.max_entries:
        raise ArchiveError(f"archive has too many entries: {len(infos)} > {limits.max_entries}")

    plan: list[tuple[zipfile.ZipInfo, PurePosixPath]] = []
    seen: dict[str, str] = {}
    total_bytes = 0
    for info in infos:
        name = info.filename
        if info.flag_bits & 0x1:
            raise ArchiveError(f"encrypted entry {name!r}")
        file_type = stat.S_IFMT(info.external_attr >> 16)
        if stat.S_ISLNK(file_type):
            raise ArchiveError(f"symbolic link entry {name!r}")
        if file_type and file_type not in (stat.S_IFREG, stat.S_IFDIR):
            raise ArchiveError(f"entry {name!r} is not a regular file")
        if info.is_dir():
            # Directories are created implicitly from file paths; only check they are safe.
            safe_parts(name)
            continue

        target = map_entry(name, classes, allowed_suffixes)
        # Case-insensitive so entries cannot collide on Windows or macOS file systems.
        key = target.as_posix().casefold()
        if key in seen:
            raise ArchiveError(f"duplicate path: {name!r} collides with {seen[key]!r}")
        seen[key] = name

        if info.file_size > limits.max_file_bytes:
            raise ArchiveError(
                f"entry {name!r} exceeds the per-file limit: "
                f"{info.file_size} > {limits.max_file_bytes} bytes"
            )
        total_bytes += info.file_size
        if total_bytes > limits.max_total_bytes:
            raise ArchiveError(
                f"archive exceeds the total size limit of {limits.max_total_bytes} bytes"
            )
        ratio = info.file_size / max(info.compress_size, 1)
        if ratio > limits.max_compression_ratio:
            raise ArchiveError(
                f"entry {name!r} has a suspicious compression ratio: "
                f"{ratio:.1f} > {limits.max_compression_ratio}"
            )
        plan.append((info, target))

    _check_coverage([target for _, target in plan], classes)
    return sorted(plan, key=lambda item: item[1].as_posix())


def _check_coverage(targets: list[PurePosixPath], classes: tuple[str, ...]) -> None:
    present = {target.parts[:2] for target in targets if len(target.parts) == 3}
    missing = [
        f"{split}/{label}"
        for split in LABELED_SPLITS.values()
        for label in classes
        if (split, label) not in present
    ]
    missing += [
        split
        for split in UNLABELED_SPLITS.values()
        if not any(len(t.parts) == 2 and t.parts[0] == split for t in targets)
    ]
    if missing:
        raise ArchiveError(f"archive is missing expected directories: {', '.join(missing)}")


def extract_archive(
    archive: Path,
    output_dir: Path,
    *,
    expected_sha256: str = EXPECTED_SHA256,
    classes: tuple[str, ...] = CLASSES,
    allowed_suffixes: frozenset[str] = ALLOWED_SUFFIXES,
    limits: SafetyLimits | None = None,
) -> ExtractionResult:
    """Verify, validate, and atomically extract the archive into a new output directory."""
    archive, output_dir = Path(archive), Path(output_dir)
    limits = limits or SafetyLimits()
    expected = _normalize_checksum(expected_sha256)
    if not archive.is_file():
        raise ArchiveError(f"archive not found: {archive}")
    _ensure_absent(output_dir)

    # Hash and read through one handle so the verified bytes are the extracted bytes.
    with archive.open("rb") as stream:
        actual = _sha256_stream(stream)
        if actual != expected:
            raise ArchiveError(f"SHA-256 mismatch for {archive}: expected {expected}, got {actual}")
        stream.seek(0)
        try:
            with zipfile.ZipFile(stream) as zf:
                plan = plan_extraction(zf, classes, limits, allowed_suffixes)
                total_bytes = _extract_atomically(zf, plan, output_dir, limits)
        except (zipfile.BadZipFile, zlib.error, EOFError) as exc:
            raise ArchiveError(f"corrupt archive {archive}: {exc}") from exc

    counts = Counter(target.parent.as_posix() for _, target in plan)
    return ExtractionResult(
        output_dir=output_dir,
        sha256=actual,
        counts=dict(sorted(counts.items())),
        total_files=len(plan),
        total_bytes=total_bytes,
    )


def _ensure_absent(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise ArchiveError(f"output already exists, refusing to overwrite: {path}")


def _extract_atomically(
    zf: zipfile.ZipFile,
    plan: list[tuple[zipfile.ZipInfo, PurePosixPath]],
    output_dir: Path,
    limits: SafetyLimits,
) -> int:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    # Same parent directory, so the final rename stays on one file system.
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", suffix=".partial", dir=output_dir.parent)
    )
    try:
        root = staging.resolve()
        total_bytes = 0
        for info, target in plan:
            dest = staging.joinpath(*target.parts)
            if not dest.resolve().is_relative_to(root):
                raise ArchiveError(f"entry {info.filename!r} resolves outside the output directory")
            dest.parent.mkdir(parents=True, exist_ok=True)
            total_bytes += _copy_member(zf, info, dest, limits)
        staging.chmod(0o755)
        _ensure_absent(output_dir)
        staging.rename(output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return total_bytes


def _copy_member(
    zf: zipfile.ZipFile, info: zipfile.ZipInfo, dest: Path, limits: SafetyLimits
) -> int:
    # Count bytes while copying instead of trusting the sizes declared in the headers.
    written = 0
    with zf.open(info) as src, dest.open("xb") as dst:
        while chunk := src.read(_CHUNK_SIZE):
            written += len(chunk)
            if written > info.file_size or written > limits.max_file_bytes:
                raise ArchiveError(f"entry {info.filename!r} is larger than its declared size")
            dst.write(chunk)
    return written


def build_parser() -> argparse.ArgumentParser:
    defaults = SafetyLimits()
    parser = argparse.ArgumentParser(
        prog="python -m deepvision.data.archive",
        description="Verify and safely extract the Intel Image Classification archive.",
    )
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sha256", default=EXPECTED_SHA256, help="expected archive SHA-256")
    parser.add_argument("--max-entries", type=int, default=defaults.max_entries)
    parser.add_argument("--max-file-bytes", type=int, default=defaults.max_file_bytes)
    parser.add_argument("--max-total-bytes", type=int, default=defaults.max_total_bytes)
    parser.add_argument(
        "--max-compression-ratio", type=float, default=defaults.max_compression_ratio
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    limits = SafetyLimits(
        max_entries=args.max_entries,
        max_file_bytes=args.max_file_bytes,
        max_total_bytes=args.max_total_bytes,
        max_compression_ratio=args.max_compression_ratio,
    )
    try:
        result = extract_archive(
            args.archive, args.output, expected_sha256=args.sha256, limits=limits
        )
    except (ArchiveError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"verified {args.archive} (sha256 {result.sha256})")
    for directory, count in result.counts.items():
        print(f"  {directory}: {count}")
    print(
        f"extracted {result.total_files} files ({result.total_bytes} bytes) to {result.output_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
