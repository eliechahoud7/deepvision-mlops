"""Image-level audit of the extracted Intel Image Classification dataset.

Every image is fully decoded and fingerprinted. The audit reports corrupt files,
unexpected formats, sizes and colour modes, class counts, exact duplicates (by
file bytes and by decoded pixels), and near-duplicate candidates found with
dHash and pHash. Nothing is modified or deleted: duplicates and candidates are
written to reports for human review.

Expected input layout, as produced by deepvision.data.archive:

    <input>/train/<class>/<file>
    <input>/test/<class>/<file>
    <input>/pred/<file>

Outputs:

    <output-dir>/summary.json                    numeric summary (DVC metrics)
    <output-dir>/image_issues.csv                corrupt and unexpected images
    <output-dir>/exact_duplicates.csv            one row per member of each duplicate group
    <output-dir>/near_duplicate_candidates.csv   candidate pairs for review
    <manifest>                                   one row per image

Usage:
    python -m deepvision.data.audit --input <dir> --output-dir <dir> --manifest <csv>
"""

import argparse
import csv
import hashlib
import io
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
from PIL import Image

LABELED_SPLITS = ("train", "test")
UNLABELED_SPLITS = ("pred",)
HISTOGRAM_MAX_DISTANCE = 12

RELATION_FIELDS = ["splits", "labels", "cross_split", "train_test", "label_conflict"]
ISSUE_FIELDS = ["relpath", "split", "label", "issue", "detail"]
GROUP_FIELDS = ["kind", "hash", "group_size", *RELATION_FIELDS, "relpath", "split", "label"]
CANDIDATE_FIELDS = [
    "relpath_a", "split_a", "label_a", "relpath_b", "split_b", "label_b",
    "dhash_distance", "phash_distance", "thumb_mad", *RELATION_FIELDS,
]  # fmt: skip

_RESAMPLE = Image.Resampling.LANCZOS
_THUMB_SIZE = 16
_PHASH_SIZE = 32
_PHASH_LOW = 8


class AuditError(Exception):
    """Raised when the input does not match the expected dataset layout."""


@dataclass(frozen=True)
class Expected:
    """Image properties every image is expected to have; others are reported as issues."""

    width: int = 150
    height: int = 150
    mode: str = "RGB"
    format: str = "JPEG"


@dataclass(frozen=True)
class NearDuplicateSettings:
    """A pair is a candidate only if both hash distances are within their thresholds."""

    dhash_max_distance: int = 6
    phash_max_distance: int = 6
    block_size: int = 1024


@dataclass
class ImageRecord:
    relpath: str
    split: str
    label: str
    file_bytes: int
    sha256: str
    ok: bool = False
    error: str = ""
    format: str = ""
    mode: str = ""
    width: int = 0
    height: int = 0
    pixel_sha256: str = ""
    dhash: str = ""
    phash: str = ""


def discover_images(root: Path) -> list[tuple[Path, str, str]]:
    """Return (path, split, label) for every file, sorted by relative path."""
    root = Path(root)
    if not root.is_dir():
        raise AuditError(f"input directory not found: {root}")
    found = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        parts = path.relative_to(root).parts
        match parts:
            case (split, label, _) if split in LABELED_SPLITS:
                found.append((path, split, label))
            case (split, _) if split in UNLABELED_SPLITS:
                found.append((path, split, ""))
            case _:
                raise AuditError(f"file outside the expected layout: {'/'.join(parts)}")
    if not found:
        raise AuditError(f"no images found under {root}")
    return sorted(found, key=lambda item: item[0].relative_to(root).as_posix())


def _bits_to_hex(bits: np.ndarray) -> str:
    return np.packbits(bits.flatten()).tobytes().hex()


def dhash(gray: Image.Image) -> str:
    """64-bit difference hash: is each pixel brighter than its right neighbour at 9x8?"""
    small = np.asarray(gray.resize((9, 8), _RESAMPLE), dtype=np.int16)
    return _bits_to_hex(small[:, 1:] > small[:, :-1])


def _dct_matrix(n: int) -> np.ndarray:
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    matrix = np.cos(np.pi * (2 * i + 1) * k / (2 * n)) * np.sqrt(2 / n)
    matrix[0] /= np.sqrt(2)
    return matrix


_DCT = _dct_matrix(_PHASH_SIZE)


def phash(gray: Image.Image) -> str:
    """64-bit perceptual hash: low-frequency DCT coefficients above their median."""
    pixels = np.asarray(gray.resize((_PHASH_SIZE, _PHASH_SIZE), _RESAMPLE), dtype=np.float64)
    low = (_DCT @ pixels @ _DCT.T)[:_PHASH_LOW, :_PHASH_LOW]
    return _bits_to_hex(low > np.median(low))


def inspect_image(path: Path, relpath: str, split: str, label: str):
    """Fully verify and decode one image; return its record and a small greyscale thumbnail."""
    data = Path(path).read_bytes()
    record = ImageRecord(relpath, split, label, len(data), hashlib.sha256(data).hexdigest())
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
        # verify() does not decode pixel data, so reopen and load fully.
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            record.format, record.mode = im.format or "", im.mode
            record.width, record.height = im.size
            rgb = im.convert("RGB")
    except Exception as exc:  # any decoder failure marks the image as corrupt
        record.error = f"{type(exc).__name__}: {exc}"
        return record, None

    pixel_digest = hashlib.sha256(f"{rgb.width}x{rgb.height}:".encode())
    pixel_digest.update(rgb.tobytes())
    gray = rgb.convert("L")
    record.ok = True
    record.pixel_sha256 = pixel_digest.hexdigest()
    record.dhash = dhash(gray)
    record.phash = phash(gray)
    thumb = np.asarray(gray.resize((_THUMB_SIZE, _THUMB_SIZE), _RESAMPLE), dtype=np.uint8)
    return record, thumb


def hash_array(hex_hashes: Iterable[str]) -> np.ndarray:
    return np.array([int(h, 16) for h in hex_hashes], dtype=np.uint64)


def near_duplicate_pairs(
    dhashes: np.ndarray, phashes: np.ndarray, settings: NearDuplicateSettings
) -> tuple[list[tuple[int, int, int, int]], np.ndarray, np.ndarray]:
    """Compare all pairs in square blocks to bound memory.

    Returns (i, j, dhash_distance, phash_distance) for i < j within both thresholds,
    plus histograms of all pairwise dHash and pHash distances (index = distance).
    """
    n, block = len(dhashes), settings.block_size
    if block < 1:
        raise ValueError("block_size must be positive")
    pairs: list[tuple[int, int, int, int]] = []
    hist_d = np.zeros(65, dtype=np.int64)
    hist_p = np.zeros(65, dtype=np.int64)
    for r0 in range(0, n, block):
        rows_d = dhashes[r0 : r0 + block, None]
        rows_p = phashes[r0 : r0 + block, None]
        for c0 in range(r0, n, block):
            dist_d = np.bitwise_count(rows_d ^ dhashes[None, c0 : c0 + block])
            dist_p = np.bitwise_count(rows_p ^ phashes[None, c0 : c0 + block])
            if c0 == r0:
                valid = np.triu(np.ones(dist_d.shape, dtype=bool), k=1)
            else:
                valid = np.ones(dist_d.shape, dtype=bool)
            hist_d += np.bincount(dist_d[valid], minlength=65)
            hist_p += np.bincount(dist_p[valid], minlength=65)
            hit = (
                valid
                & (dist_d <= settings.dhash_max_distance)
                & (dist_p <= settings.phash_max_distance)
            )
            rows, cols = np.nonzero(hit)
            pairs.extend(
                zip(
                    (rows + r0).tolist(),
                    (cols + c0).tolist(),
                    dist_d[hit].tolist(),
                    dist_p[hit].tolist(),
                    strict=True,
                )
            )
    return pairs, hist_d, hist_p


def _relation(members: Iterable[ImageRecord]) -> dict:
    members = list(members)
    splits = sorted({m.split for m in members})
    labels = sorted({m.label for m in members if m.label})
    return {
        "splits": "+".join(splits),
        "labels": "+".join(labels),
        "cross_split": len(splits) > 1,
        "train_test": {"train", "test"} <= set(splits),
        "label_conflict": len(labels) > 1,
    }


def exact_duplicate_groups(records: list[ImageRecord]) -> list[dict]:
    """Group identical files, and identical decoded pixels across different files."""
    by_file: dict[str, list[ImageRecord]] = defaultdict(list)
    by_pixel: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        by_file[record.sha256].append(record)
        if record.ok:
            by_pixel[record.pixel_sha256].append(record)

    groups = []
    for kind, index in (("file_sha256", by_file), ("pixel_sha256", by_pixel)):
        for digest, members in sorted(index.items()):
            if len(members) < 2:
                continue
            # Pixel groups made only of byte-identical files are already file groups.
            if kind == "pixel_sha256" and len({m.sha256 for m in members}) < 2:
                continue
            groups.append({"kind": kind, "hash": digest, "members": members, **_relation(members)})
    return groups


def image_issues(records: list[ImageRecord], expected: Expected) -> list[dict]:
    issues = []
    for r in records:
        base = {"relpath": r.relpath, "split": r.split, "label": r.label}
        if not r.ok:
            issues.append({**base, "issue": "corrupt", "detail": r.error})
            continue
        if (r.width, r.height) != (expected.width, expected.height):
            issues.append({**base, "issue": "unexpected_size", "detail": f"{r.width}x{r.height}"})
        if r.mode != expected.mode:
            issues.append({**base, "issue": "unexpected_mode", "detail": r.mode})
        if r.format != expected.format:
            issues.append({**base, "issue": "unexpected_format", "detail": r.format})
    return issues


def _count_by(items: Iterable[dict], key: str) -> dict[str, int]:
    return dict(sorted(Counter(item[key] for item in items).items()))


def _group_stats(groups: list[dict]) -> dict:
    return {
        "groups": len(groups),
        "images": sum(len(g["members"]) for g in groups),
        "cross_split_groups": sum(g["cross_split"] for g in groups),
        "train_test_groups": sum(g["train_test"] for g in groups),
        "label_conflict_groups": sum(g["label_conflict"] for g in groups),
        "groups_by_splits": _count_by(groups, "splits"),
    }


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_audit(
    input_dir: Path,
    output_dir: Path,
    manifest_path: Path,
    *,
    expected: Expected | None = None,
    settings: NearDuplicateSettings | None = None,
) -> dict:
    """Audit every image under input_dir, write all reports, and return the summary."""
    expected = expected or Expected()
    settings = settings or NearDuplicateSettings()
    input_dir, output_dir, manifest_path = Path(input_dir), Path(output_dir), Path(manifest_path)

    records: list[ImageRecord] = []
    thumbs: list[np.ndarray] = []
    for path, split, label in discover_images(input_dir):
        record, thumb = inspect_image(path, path.relative_to(input_dir).as_posix(), split, label)
        records.append(record)
        if thumb is not None:
            thumbs.append(thumb)

    decoded = [r for r in records if r.ok]
    raw_pairs, hist_d, hist_p = near_duplicate_pairs(
        hash_array(r.dhash for r in decoded), hash_array(r.phash for r in decoded), settings
    )
    candidates = []
    for i, j, dist_d, dist_p in raw_pairs:
        a, b = decoded[i], decoded[j]
        if a.sha256 == b.sha256 or a.pixel_sha256 == b.pixel_sha256:
            continue  # already reported as exact duplicates
        mad = float(np.abs(thumbs[i].astype(np.int16) - thumbs[j]).mean())
        candidates.append(
            {
                "relpath_a": a.relpath,
                "split_a": a.split,
                "label_a": a.label,
                "relpath_b": b.relpath,
                "split_b": b.split,
                "label_b": b.label,
                "dhash_distance": dist_d,
                "phash_distance": dist_p,
                "thumb_mad": f"{mad:.3f}",
                **_relation((a, b)),
            }
        )
    candidates.sort(
        key=lambda c: (
            c["dhash_distance"] + c["phash_distance"],
            float(c["thumb_mad"]),
            c["relpath_a"],
            c["relpath_b"],
        )
    )

    groups = exact_duplicate_groups(records)
    issues = image_issues(records, expected)

    _write_csv(manifest_path, [f.name for f in fields(ImageRecord)], map(asdict, records))
    _write_csv(output_dir / "image_issues.csv", ISSUE_FIELDS, issues)
    group_rows = (
        {
            "kind": g["kind"],
            "hash": g["hash"],
            "group_size": len(g["members"]),
            **{k: g[k] for k in RELATION_FIELDS},
            "relpath": m.relpath,
            "split": m.split,
            "label": m.label,
        }
        for g in groups
        for m in g["members"]
    )
    _write_csv(output_dir / "exact_duplicates.csv", GROUP_FIELDS, group_rows)
    _write_csv(output_dir / "near_duplicate_candidates.csv", CANDIDATE_FIELDS, candidates)

    class_counts = {
        split: dict(sorted(Counter(r.label for r in records if r.split == split).items()))
        for split in LABELED_SPLITS
    }
    histogram_range = range(HISTOGRAM_MAX_DISTANCE + 1)
    summary = {
        "images": {
            "total": len(records),
            "decoded": len(decoded),
            "corrupt": len(records) - len(decoded),
        },
        "split_counts": dict(sorted(Counter(r.split for r in records).items())),
        "class_counts": class_counts,
        "properties": {
            "formats": dict(sorted(Counter(r.format for r in decoded).items())),
            "modes": dict(sorted(Counter(r.mode for r in decoded).items())),
            "sizes": dict(sorted(Counter(f"{r.width}x{r.height}" for r in decoded).items())),
        },
        "issues": _count_by(issues, "issue"),
        "exact_duplicates": {
            kind: _group_stats([g for g in groups if g["kind"] == kind])
            for kind in ("file_sha256", "pixel_sha256")
        },
        "near_duplicates": {
            "dhash_max_distance": settings.dhash_max_distance,
            "phash_max_distance": settings.phash_max_distance,
            "candidate_pairs": len(candidates),
            "cross_split_pairs": sum(c["cross_split"] for c in candidates),
            "train_test_pairs": sum(c["train_test"] for c in candidates),
            "label_conflict_pairs": sum(c["label_conflict"] for c in candidates),
            "pairs_by_splits": _count_by(candidates, "splits"),
        },
        "pair_distance_histogram": {
            "dhash": {str(d): int(hist_d[d]) for d in histogram_range},
            "phash": {str(d): int(hist_p[d]) for d in histogram_range},
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    expected, near = Expected(), NearDuplicateSettings()
    parser = argparse.ArgumentParser(
        prog="python -m deepvision.data.audit",
        description="Audit extracted images for corruption, properties, and duplicates.",
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-width", type=int, default=expected.width)
    parser.add_argument("--expected-height", type=int, default=expected.height)
    parser.add_argument("--expected-mode", default=expected.mode)
    parser.add_argument("--expected-format", default=expected.format)
    parser.add_argument("--dhash-max-distance", type=int, default=near.dhash_max_distance)
    parser.add_argument("--phash-max-distance", type=int, default=near.phash_max_distance)
    parser.add_argument("--block-size", type=int, default=near.block_size)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    expected = Expected(
        args.expected_width, args.expected_height, args.expected_mode, args.expected_format
    )
    settings = NearDuplicateSettings(
        args.dhash_max_distance, args.phash_max_distance, args.block_size
    )
    try:
        summary = run_audit(
            args.input, args.output_dir, args.manifest, expected=expected, settings=settings
        )
    except (AuditError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    images, near = summary["images"], summary["near_duplicates"]
    print(
        f"audited {images['total']} images: "
        f"{images['decoded']} decoded, {images['corrupt']} corrupt"
    )
    print(f"issues: {summary['issues'] or 'none'}")
    for kind, stats in summary["exact_duplicates"].items():
        print(
            f"exact duplicates ({kind}): {stats['groups']} groups, "
            f"{stats['train_test_groups']} train/test, "
            f"{stats['label_conflict_groups']} label conflicts"
        )
    print(
        f"near-duplicate candidates: {near['candidate_pairs']} pairs, "
        f"{near['train_test_pairs']} train/test, {near['label_conflict_pairs']} label conflicts"
    )
    print(f"reports written to {args.output_dir} and {args.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
