import csv
import hashlib
import io
import itertools
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from deepvision.data.audit import (
    AuditError,
    NearDuplicateSettings,
    dhash,
    discover_images,
    hash_array,
    main,
    near_duplicate_pairs,
    phash,
    run_audit,
)

REPORTS = [
    "summary.json",
    "image_issues.csv",
    "exact_duplicates.csv",
    "near_duplicate_candidates.csv",
]


def smooth_image(seed: int, size: tuple[int, int] = (150, 150), mode: str = "RGB") -> Image.Image:
    """A distinct, natural-looking image: random 8x8 colours upsampled smoothly."""
    rng = np.random.default_rng(seed)
    small = Image.fromarray(rng.integers(0, 256, (8, 8, 3), dtype=np.uint8))
    return small.resize(size, Image.Resampling.BICUBIC).convert(mode)


def brighten(image: Image.Image, amount: int) -> Image.Image:
    pixels = np.asarray(image, dtype=np.int16) + amount
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))


def save(image: Image.Image, path: Path, fmt: str = "JPEG", **kwargs) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format=fmt, **({"quality": 95} if fmt == "JPEG" else {}) | kwargs)
    return path


def copy_image(root: Path, source: str, target: str) -> None:
    shutil.copyfile(root / source, root / target)


def jpeg_bytes(seed: int) -> bytes:
    buffer = io.BytesIO()
    smooth_image(seed).save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    root = tmp_path / "intel"
    for seed, rel in enumerate(
        [
            "train/forest/f1.jpg",
            "train/forest/f2.jpg",
            "train/sea/s1.jpg",
            "test/forest/t1.jpg",
            "test/sea/t2.jpg",
            "pred/p1.jpg",
        ],
        start=1,
    ):
        save(smooth_image(seed), root / rel)
    return root


def audit(root: Path, tmp_path: Path, **kwargs) -> tuple[dict, Path]:
    out = tmp_path / "reports"
    summary = run_audit(root, out, tmp_path / "processed" / "manifest.csv", **kwargs)
    return summary, out


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def tree_fingerprint(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_clean_dataset_has_no_findings(dataset, tmp_path):
    summary, out = audit(dataset, tmp_path)

    assert summary["images"] == {"total": 6, "decoded": 6, "corrupt": 0}
    assert summary["split_counts"] == {"pred": 1, "test": 2, "train": 3}
    assert summary["class_counts"] == {
        "train": {"forest": 2, "sea": 1},
        "test": {"forest": 1, "sea": 1},
    }
    assert summary["properties"] == {
        "formats": {"JPEG": 6},
        "modes": {"RGB": 6},
        "sizes": {"150x150": 6},
    }
    assert summary["issues"] == {}
    assert summary["exact_duplicates"]["file_sha256"]["groups"] == 0
    assert summary["exact_duplicates"]["pixel_sha256"]["groups"] == 0
    assert summary["near_duplicates"]["candidate_pairs"] == 0
    assert sum(summary["pair_distance_histogram"]["dhash"].values()) <= 15  # 6 choose 2
    for name in REPORTS:
        assert (out / name).is_file()

    manifest = read_csv(tmp_path / "processed" / "manifest.csv")
    assert [row["relpath"] for row in manifest] == sorted(row["relpath"] for row in manifest)
    first = manifest[0]
    assert first["relpath"] == "pred/p1.jpg"
    assert (first["split"], first["label"], first["ok"]) == ("pred", "", "True")
    assert (first["width"], first["height"], first["mode"], first["format"]) == (
        "150",
        "150",
        "RGB",
        "JPEG",
    )
    assert len(first["sha256"]) == 64 and len(first["dhash"]) == len(first["phash"]) == 16


def test_corrupt_images_are_reported_not_removed(dataset, tmp_path):
    (dataset / "train" / "forest" / "garbage.jpg").write_bytes(b"not an image at all")
    valid = jpeg_bytes(99)
    (dataset / "test" / "sea" / "truncated.jpg").write_bytes(valid[: len(valid) // 2])

    summary, out = audit(dataset, tmp_path)

    assert summary["images"] == {"total": 8, "decoded": 6, "corrupt": 2}
    assert summary["issues"] == {"corrupt": 2}
    corrupt = {row["relpath"]: row for row in read_csv(out / "image_issues.csv")}
    assert set(corrupt) == {"test/sea/truncated.jpg", "train/forest/garbage.jpg"}
    assert all(row["issue"] == "corrupt" and row["detail"] for row in corrupt.values())
    manifest = {row["relpath"]: row for row in read_csv(tmp_path / "processed" / "manifest.csv")}
    assert manifest["test/sea/truncated.jpg"]["ok"] == "False"
    assert manifest["test/sea/truncated.jpg"]["dhash"] == ""
    assert (dataset / "train" / "forest" / "garbage.jpg").exists()


def test_unexpected_size_mode_and_format_are_reported(dataset, tmp_path):
    save(smooth_image(20, size=(150, 113)), dataset / "train" / "sea" / "short.jpg")
    save(smooth_image(21, mode="L"), dataset / "train" / "sea" / "gray.jpg")
    save(smooth_image(22), dataset / "test" / "forest" / "png.jpg", fmt="PNG")

    summary, out = audit(dataset, tmp_path)

    assert summary["issues"] == {"unexpected_format": 1, "unexpected_mode": 1, "unexpected_size": 1}
    details = {(r["relpath"], r["issue"]): r["detail"] for r in read_csv(out / "image_issues.csv")}
    assert details == {
        ("test/forest/png.jpg", "unexpected_format"): "PNG",
        ("train/sea/gray.jpg", "unexpected_mode"): "L",
        ("train/sea/short.jpg", "unexpected_size"): "150x113",
    }


def test_exact_duplicate_across_train_and_test_is_leakage(dataset, tmp_path):
    copy_image(dataset, "train/forest/f1.jpg", "test/forest/dup.jpg")

    summary, out = audit(dataset, tmp_path)

    stats = summary["exact_duplicates"]["file_sha256"]
    assert (stats["groups"], stats["images"]) == (1, 2)
    assert (stats["cross_split_groups"], stats["train_test_groups"]) == (1, 1)
    assert stats["label_conflict_groups"] == 0
    assert stats["groups_by_splits"] == {"test+train": 1}
    rows = read_csv(out / "exact_duplicates.csv")
    assert [r["relpath"] for r in rows] == ["test/forest/dup.jpg", "train/forest/f1.jpg"]
    assert {r["train_test"] for r in rows} == {"True"}
    assert summary["near_duplicates"]["candidate_pairs"] == 0  # not repeated as a candidate


def test_identical_image_with_two_labels_is_a_label_conflict(dataset, tmp_path):
    copy_image(dataset, "train/forest/f1.jpg", "train/sea/s9.jpg")

    summary, out = audit(dataset, tmp_path)

    stats = summary["exact_duplicates"]["file_sha256"]
    assert (stats["label_conflict_groups"], stats["cross_split_groups"]) == (1, 0)
    rows = read_csv(out / "exact_duplicates.csv")
    assert {r["labels"] for r in rows} == {"forest+sea"}


def test_pixel_identical_files_with_different_bytes_are_grouped(dataset, tmp_path):
    image = smooth_image(30)
    a = save(image, dataset / "train" / "forest" / "a.jpg", fmt="PNG", compress_level=1)
    b = save(image, dataset / "pred" / "b.jpg", fmt="PNG", compress_level=9)
    assert a.read_bytes() != b.read_bytes()

    summary, out = audit(dataset, tmp_path)

    assert summary["exact_duplicates"]["file_sha256"]["groups"] == 0
    pixel = summary["exact_duplicates"]["pixel_sha256"]
    assert (pixel["groups"], pixel["cross_split_groups"], pixel["train_test_groups"]) == (1, 1, 0)
    assert pixel["groups_by_splits"] == {"pred+train": 1}
    rows = read_csv(out / "exact_duplicates.csv")
    assert {r["kind"] for r in rows} == {"pixel_sha256"}


def test_near_duplicate_across_splits_is_a_candidate(dataset, tmp_path):
    base = smooth_image(1)  # same picture as train/forest/f1.jpg
    save(brighten(base, 4), dataset / "test" / "forest" / "near.jpg")

    summary, out = audit(dataset, tmp_path)

    near = summary["near_duplicates"]
    assert (near["candidate_pairs"], near["train_test_pairs"]) == (1, 1)
    assert near["pairs_by_splits"] == {"test+train": 1}
    [pair] = read_csv(out / "near_duplicate_candidates.csv")
    assert (pair["relpath_a"], pair["relpath_b"]) == ("test/forest/near.jpg", "train/forest/f1.jpg")
    assert int(pair["dhash_distance"]) <= 6 and int(pair["phash_distance"]) <= 6
    assert float(pair["thumb_mad"]) < 10
    assert summary["exact_duplicates"]["file_sha256"]["groups"] == 0
    assert (dataset / "test" / "forest" / "near.jpg").exists()


def test_perceptual_hashes_separate_similar_from_different_images():
    base = smooth_image(5).convert("L")
    similar = brighten(smooth_image(5), 4).convert("L")
    different = smooth_image(6).convert("L")

    def distance(h1: str, h2: str) -> int:
        return (int(h1, 16) ^ int(h2, 16)).bit_count()

    for hash_fn in (dhash, phash):
        assert len(hash_fn(base)) == 16
        assert distance(hash_fn(base), hash_fn(similar)) <= 4
        assert distance(hash_fn(base), hash_fn(different)) >= 12


def test_block_comparison_matches_brute_force():
    rng = np.random.default_rng(0)
    base = rng.integers(0, 2**63, 40, dtype=np.int64).astype(np.uint64)
    flips = np.uint64(1) << rng.integers(0, 64, 40).astype(np.uint64)
    dh = np.concatenate([base, base ^ flips])  # each base hash has a 1-bit neighbour
    ph = np.concatenate([base, base ^ (flips | np.uint64(1))])
    settings = NearDuplicateSettings(dhash_max_distance=2, phash_max_distance=3, block_size=1000)

    expected_pairs = []
    expected_hist = np.zeros(65, dtype=np.int64)
    for i, j in itertools.combinations(range(len(dh)), 2):
        dd = (int(dh[i]) ^ int(dh[j])).bit_count()
        pd = (int(ph[i]) ^ int(ph[j])).bit_count()
        expected_hist[dd] += 1
        if dd <= 2 and pd <= 3:
            expected_pairs.append((i, j, dd, pd))

    for block in (1, 7, 80, 1000):
        pairs, hist_d, _ = near_duplicate_pairs(
            dh, ph, NearDuplicateSettings(2, 3, block_size=block)
        )
        assert sorted(pairs) == expected_pairs
        assert np.array_equal(hist_d, expected_hist)
    assert len(near_duplicate_pairs(dh, ph, settings)[0]) >= 40


def test_both_hash_thresholds_must_hold():
    dh = hash_array(["0000000000000000", "0000000000000007"])  # distance 3
    ph = hash_array(["0000000000000000", "00000000000000ff"])  # distance 8

    def count(d_max: int, p_max: int) -> int:
        return len(near_duplicate_pairs(dh, ph, NearDuplicateSettings(d_max, p_max, 1))[0])

    assert count(3, 8) == 1
    assert count(2, 8) == 0
    assert count(3, 7) == 0
    with pytest.raises(ValueError, match="block_size"):
        near_duplicate_pairs(dh, ph, NearDuplicateSettings(3, 8, 0))


def test_reports_are_deterministic(dataset, tmp_path):
    copy_image(dataset, "train/forest/f1.jpg", "test/forest/dup.jpg")
    save(brighten(smooth_image(2), 3), dataset / "pred" / "near.jpg")

    run_audit(dataset, tmp_path / "r1", tmp_path / "m1.csv")
    run_audit(dataset, tmp_path / "r2", tmp_path / "m2.csv")

    for name in REPORTS:
        assert (tmp_path / "r1" / name).read_bytes() == (tmp_path / "r2" / name).read_bytes()
    assert (tmp_path / "m1.csv").read_bytes() == (tmp_path / "m2.csv").read_bytes()
    assert str(tmp_path) not in (tmp_path / "r1" / "summary.json").read_text()


def test_input_images_are_never_modified(dataset, tmp_path):
    copy_image(dataset, "train/forest/f1.jpg", "test/forest/dup.jpg")
    before = tree_fingerprint(dataset)

    audit(dataset, tmp_path)

    assert tree_fingerprint(dataset) == before


@pytest.mark.parametrize("bad", ["stray.jpg", "train/stray.jpg", "pred/sub/p.jpg", "val/x/y.jpg"])
def test_files_outside_layout_are_rejected(dataset, tmp_path, bad):
    save(smooth_image(50), dataset / bad)
    with pytest.raises(AuditError, match="expected layout"):
        discover_images(dataset)


def test_missing_or_empty_input_is_rejected(tmp_path):
    with pytest.raises(AuditError, match="not found"):
        discover_images(tmp_path / "missing")
    (tmp_path / "empty").mkdir()
    with pytest.raises(AuditError, match="no images"):
        discover_images(tmp_path / "empty")


def test_cli_writes_reports(dataset, tmp_path, capsys):
    out, manifest = tmp_path / "reports", tmp_path / "manifest.csv"

    code = main(
        ["--input", str(dataset), "--output-dir", str(out), "--manifest", str(manifest)]
        + ["--dhash-max-distance", "5", "--phash-max-distance", "4", "--block-size", "2"]
    )

    assert code == 0
    assert "audited 6 images: 6 decoded, 0 corrupt" in capsys.readouterr().out
    summary = json.loads((out / "summary.json").read_text())
    assert summary["near_duplicates"]["dhash_max_distance"] == 5
    assert summary["near_duplicates"]["phash_max_distance"] == 4
    assert manifest.is_file()


def test_cli_reports_errors(tmp_path, capsys):
    code = main(
        ["--input", str(tmp_path / "missing"), "--output-dir", str(tmp_path / "r")]
        + ["--manifest", str(tmp_path / "m.csv")]
    )
    assert code == 1
    assert "error: input directory not found" in capsys.readouterr().err
