import hashlib
import inspect
import stat
import warnings
import zipfile
from pathlib import Path

import pytest

from deepvision.data.archive import (
    CLASSES,
    EXPECTED_SHA256,
    ArchiveError,
    SafetyLimits,
    extract_archive,
    main,
    sha256_file,
)

FILE_ATTR = (stat.S_IFREG | 0o644) << 16
DIR_ATTR = ((stat.S_IFDIR | 0o755) << 16) | 0x10
LINK_ATTR = (stat.S_IFLNK | 0o777) << 16


def valid_entries() -> list[tuple[str, bytes]]:
    entries = [
        (f"{split}/{split}/{label}/{label}_1.jpg", f"{split}-{label}".encode())
        for split in ("seg_train", "seg_test")
        for label in CLASSES
    ]
    entries += [("seg_pred/seg_pred/10.jpg", b"pred-10"), ("seg_pred/seg_pred/11.jpg", b"pred-11")]
    return entries


def add_entry(
    zf: zipfile.ZipFile,
    name: str,
    data: bytes,
    *,
    attr: int = FILE_ATTR,
    compress_type: int = zipfile.ZIP_STORED,
) -> None:
    info = zipfile.ZipInfo("placeholder")
    # Assigned after construction so ZipInfo does not normalize the raw name.
    info.filename = name
    info.external_attr = attr
    info.compress_type = compress_type
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # duplicate-name warning
        zf.writestr(info, data)


def write_zip(path: Path, entries: list[tuple[str, bytes]], **kwargs) -> str:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in entries:
            add_entry(zf, name, data, **kwargs)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(path: Path) -> tuple[str, int, int]:
    info = path.stat()
    return hashlib.sha256(path.read_bytes()).hexdigest(), info.st_size, info.st_mtime_ns


def assert_nothing_published(output: Path) -> None:
    assert not output.exists()
    if output.parent.exists():
        assert not [p.name for p in output.parent.iterdir() if p.name.endswith(".partial")]


@pytest.fixture
def output(tmp_path: Path) -> Path:
    return tmp_path / "interim" / "intel"


@pytest.fixture
def valid_zip(tmp_path: Path) -> tuple[Path, str]:
    path = tmp_path / "archive.zip"
    return path, write_zip(path, valid_entries())


def test_extracts_and_normalizes_layout(valid_zip, output):
    path, sha = valid_zip

    result = extract_archive(path, output, expected_sha256=sha)

    assert result.output_dir == output
    assert result.sha256 == sha
    assert result.total_files == 14
    assert result.counts == {
        "pred": 2,
        **{f"test/{label}": 1 for label in CLASSES},
        **{f"train/{label}": 1 for label in CLASSES},
    }
    assert sorted(p.name for p in output.iterdir()) == ["pred", "test", "train"]
    assert (output / "train" / "forest" / "forest_1.jpg").read_bytes() == b"seg_train-forest"
    assert (output / "pred" / "10.jpg").read_bytes() == b"pred-10"
    assert result.total_bytes == sum(len(data) for _, data in valid_entries())
    assert [p.name for p in output.parent.iterdir()] == ["intel"]  # no staging directory left


def test_directory_entries_are_allowed(tmp_path, output):
    path = tmp_path / "archive.zip"
    with zipfile.ZipFile(path, "w") as zf:
        add_entry(zf, "seg_train/", b"", attr=DIR_ATTR)
        add_entry(zf, "seg_train/seg_train/", b"", attr=DIR_ATTR)
        for name, data in valid_entries():
            add_entry(zf, name, data)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()

    assert extract_archive(path, output, expected_sha256=sha).total_files == 14


def test_sha256_file_matches_hashlib(valid_zip):
    path, sha = valid_zip
    assert sha256_file(path) == sha


def test_default_checksum_is_dataset_checksum():
    default = inspect.signature(extract_archive).parameters["expected_sha256"].default
    assert default == EXPECTED_SHA256
    assert EXPECTED_SHA256 == "3921405607c3e9d67a3459b6c629894c3a723aa1647f9310b6793da306e89337"


def test_checksum_mismatch_extracts_nothing(valid_zip, output):
    path, _ = valid_zip

    with pytest.raises(ArchiveError, match="SHA-256 mismatch"):
        extract_archive(path, output, expected_sha256="0" * 64)
    assert_nothing_published(output)


def test_malformed_checksum_is_rejected(valid_zip, output):
    path, _ = valid_zip
    with pytest.raises(ArchiveError, match="64 hex"):
        extract_archive(path, output, expected_sha256="abc123")


def test_missing_archive_is_rejected(tmp_path, output):
    with pytest.raises(ArchiveError, match="not found"):
        extract_archive(tmp_path / "missing.zip", output, expected_sha256=EXPECTED_SHA256)


@pytest.mark.parametrize(
    "name",
    [
        "../evil.jpg",
        "seg_train/seg_train/forest/../../../../evil.jpg",
        "/tmp/evil.jpg",
        "C:/Windows/evil.jpg",
        "..\\..\\evil.jpg",
        "seg_train/seg_train/forest//a.jpg",
        "seg_train/seg_train/forest/./a.jpg",
        "seg_train/seg_train/forest/a.jpg:stream.jpg",
        "seg_train/seg_train/forest/CON.jpg",
        "seg_train/seg_train/forest/a.jpg ",
        "../evil/",
    ],
)
def test_malicious_paths_are_rejected(tmp_path, output, name):
    path = tmp_path / "archive.zip"
    sha = write_zip(path, [*valid_entries(), (name, b"x")])

    with pytest.raises(ArchiveError):
        extract_archive(path, output, expected_sha256=sha)
    assert_nothing_published(output)
    assert not (tmp_path / "evil.jpg").exists()


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("seg_train/forest/a.jpg", "expected layout"),
        ("seg_train/seg_test/forest/a.jpg", "expected layout"),
        ("seg_val/seg_val/forest/a.jpg", "expected layout"),
        ("seg_pred/seg_pred/forest/a.jpg", "expected layout"),
        ("seg_train/seg_train/desert/a.jpg", "unknown class"),
        ("seg_train/seg_train/Forest/a.jpg", "unknown class"),
        ("readme.jpg", "expected layout"),
        ("seg_train/seg_train/forest/a.png", "file type"),
        ("seg_train/seg_train/forest/a.jpg.exe", "file type"),
        ("seg_train/seg_train/forest/notes.txt", "file type"),
    ],
)
def test_invalid_layout_or_extension_is_rejected(tmp_path, output, name, message):
    path = tmp_path / "archive.zip"
    sha = write_zip(path, [*valid_entries(), (name, b"x")])

    with pytest.raises(ArchiveError, match=message):
        extract_archive(path, output, expected_sha256=sha)
    assert_nothing_published(output)


@pytest.mark.parametrize("dropped", ["seg_test/seg_test/glacier/", "seg_pred/"])
def test_missing_class_or_split_is_rejected(tmp_path, output, dropped):
    path = tmp_path / "archive.zip"
    sha = write_zip(path, [e for e in valid_entries() if not e[0].startswith(dropped)])

    with pytest.raises(ArchiveError, match="missing expected directories"):
        extract_archive(path, output, expected_sha256=sha)


@pytest.mark.parametrize(
    "duplicate",
    ["seg_train/seg_train/forest/forest_1.jpg", "seg_train/seg_train/forest/FOREST_1.JPG"],
)
def test_duplicate_paths_are_rejected(tmp_path, output, duplicate):
    path = tmp_path / "archive.zip"
    sha = write_zip(path, [*valid_entries(), (duplicate, b"other")])

    with pytest.raises(ArchiveError, match="duplicate path"):
        extract_archive(path, output, expected_sha256=sha)


def test_symlink_entry_is_rejected(tmp_path, output):
    path = tmp_path / "archive.zip"
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in valid_entries():
            add_entry(zf, name, data)
        add_entry(zf, "seg_pred/seg_pred/link.jpg", b"../../outside", attr=LINK_ATTR)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()

    with pytest.raises(ArchiveError, match="symbolic link"):
        extract_archive(path, output, expected_sha256=sha)


def test_encrypted_entry_is_rejected(tmp_path, output):
    path = tmp_path / "archive.zip"
    write_zip(path, valid_entries())
    # zipfile cannot write encrypted entries, so set the "encrypted" flag in the first
    # central directory record directly (general purpose flags at offset 8).
    raw = bytearray(path.read_bytes())
    offset = raw.index(b"PK\x01\x02") + 8
    raw[offset] |= 0x1
    path.write_bytes(raw)
    sha = hashlib.sha256(raw).hexdigest()

    with pytest.raises(ArchiveError, match="encrypted"):
        extract_archive(path, output, expected_sha256=sha)


@pytest.mark.parametrize(
    ("limits", "message"),
    [
        (SafetyLimits(max_entries=13), "too many entries"),
        (SafetyLimits(max_file_bytes=10), "per-file limit"),
        (SafetyLimits(max_total_bytes=100), "total size limit"),
    ],
)
def test_size_limits_are_enforced(valid_zip, output, limits, message):
    path, sha = valid_zip

    with pytest.raises(ArchiveError, match=message):
        extract_archive(path, output, expected_sha256=sha, limits=limits)
    assert_nothing_published(output)


def test_compression_ratio_limit_is_enforced(tmp_path, output):
    path = tmp_path / "archive.zip"
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in valid_entries():
            add_entry(zf, name, data)
        add_entry(
            zf, "seg_pred/seg_pred/bomb.jpg", b"\0" * 1_000_000, compress_type=zipfile.ZIP_DEFLATED
        )
    sha = hashlib.sha256(path.read_bytes()).hexdigest()

    with pytest.raises(ArchiveError, match="compression ratio"):
        extract_archive(path, output, expected_sha256=sha)


def test_existing_output_is_never_overwritten(valid_zip, output):
    path, sha = valid_zip
    output.mkdir(parents=True)
    sentinel = output / "keep.txt"
    sentinel.write_text("existing")

    with pytest.raises(ArchiveError, match="refusing to overwrite"):
        extract_archive(path, output, expected_sha256=sha)
    assert sorted(p.name for p in output.iterdir()) == ["keep.txt"]
    assert sentinel.read_text() == "existing"


def test_partial_extraction_is_cleaned_up_on_failure(tmp_path, output):
    path = tmp_path / "archive.zip"
    entries = [*valid_entries(), ("seg_test/seg_test/street/zz.jpg", b"CORRUPT-ME")]
    write_zip(path, entries)
    # Damage the stored bytes of one entry that sorts after others, so some files are
    # written before its CRC check fails during extraction.
    raw = path.read_bytes().replace(b"CORRUPT-ME", b"CORRUPTED!")
    path.write_bytes(raw)
    sha = hashlib.sha256(raw).hexdigest()

    with pytest.raises(ArchiveError, match="corrupt archive"):
        extract_archive(path, output, expected_sha256=sha)
    assert_nothing_published(output)
    assert list(output.parent.iterdir()) == []


def test_source_archive_is_not_modified(valid_zip, output, tmp_path):
    path, sha = valid_zip
    before = fingerprint(path)

    extract_archive(path, output, expected_sha256=sha)
    with pytest.raises(ArchiveError):
        extract_archive(path, tmp_path / "other", expected_sha256="0" * 64)

    assert fingerprint(path) == before


def test_cli_extracts_with_custom_paths(valid_zip, output, capsys):
    path, sha = valid_zip

    code = main(["--archive", str(path), "--output", str(output), "--sha256", sha])

    assert code == 0
    assert (output / "test" / "sea" / "sea_1.jpg").is_file()
    assert "extracted 14 files" in capsys.readouterr().out


def test_cli_reports_errors_without_extracting(valid_zip, output, capsys):
    path, _ = valid_zip

    code = main(["--archive", str(path), "--output", str(output), "--sha256", "f" * 64])

    assert code == 1
    assert "error: SHA-256 mismatch" in capsys.readouterr().err
    assert_nothing_published(output)


def test_cli_passes_safety_limits(valid_zip, output, capsys):
    path, sha = valid_zip

    args = ["--archive", str(path), "--output", str(output), "--sha256", sha]
    code = main([*args, "--max-entries", "3"])

    assert code == 1
    assert "too many entries" in capsys.readouterr().err
