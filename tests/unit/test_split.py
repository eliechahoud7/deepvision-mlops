import csv
import json
import random
from pathlib import Path

import pytest

from deepvision.data.split import (
    OUTPUT_FIELDS,
    SplitError,
    SplitSettings,
    assign_splits,
    build_groups,
    leakage_checks,
    main,
    run_split,
)


def record(relpath: str, ok: bool = True) -> dict:
    parts = relpath.split("/")
    label = parts[1] if parts[0] in ("train", "test") else ""
    return {"relpath": relpath, "split": parts[0], "label": label, "ok": str(ok), "sha256": "x"}


def exact(kind_hash: str, *paths: str) -> list[dict]:
    return [{"kind": "file_sha256", "hash": kind_hash, "relpath": p} for p in paths]


def near(a: str, b: str) -> dict:
    return {"relpath_a": a, "relpath_b": b}


def split_of(rows: list[dict]) -> dict[str, dict]:
    return {r["relpath"]: r for r in rows}


def singletons(split: str, label: str, n: int, start: int = 0) -> list[dict]:
    return [record(f"{split}/{label}/{i:04d}.jpg") for i in range(start, start + n)]


def run(records, exact_rows=(), near_rows=(), settings=None) -> dict[str, dict]:
    groups = build_groups(records, list(exact_rows), list(near_rows))
    return split_of(assign_splits(records, groups, settings or SplitSettings()))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.fixture
def audit_files(tmp_path: Path) -> dict[str, Path]:
    records = (
        singletons("train", "forest", 40)
        + singletons("train", "sea", 40)
        + singletons("test", "forest", 5)
        + singletons("test", "sea", 5)
        + [record("pred/0001.jpg"), record("pred/0002.jpg")]
    )
    exact_rows = exact("h1", "train/forest/0000.jpg", "train/sea/0000.jpg") + exact(
        "h2", "test/forest/0004.jpg", "train/forest/0001.jpg"
    )
    near_rows = [
        near("test/forest/0000.jpg", "train/forest/0002.jpg"),
        near("pred/0001.jpg", "train/sea/0005.jpg"),
        near("train/sea/0006.jpg", "train/sea/0007.jpg"),
    ]
    return {
        "manifest": write_csv(tmp_path / "manifest.csv", records, list(records[0])),
        "exact": write_csv(tmp_path / "exact.csv", exact_rows, ["kind", "hash", "relpath"]),
        "near": write_csv(tmp_path / "near.csv", near_rows, ["relpath_a", "relpath_b"]),
    }


def test_groups_are_transitive_with_smallest_path_as_id():
    records = [record(p) for p in ("train/a/3.jpg", "train/a/1.jpg", "pred/9.jpg", "train/a/5.jpg")]
    groups = build_groups(
        records, exact("h", "train/a/3.jpg", "pred/9.jpg"), [near("pred/9.jpg", "train/a/1.jpg")]
    )
    assert groups == {
        "pred/9.jpg": ["pred/9.jpg", "train/a/1.jpg", "train/a/3.jpg"],
        "train/a/5.jpg": ["train/a/5.jpg"],
    }


def test_links_to_unknown_images_are_rejected():
    with pytest.raises(SplitError, match="unknown image"):
        build_groups([record("train/a/1.jpg")], [], [near("train/a/1.jpg", "train/a/2.jpg")])


def test_label_conflicts_are_excluded_without_relabelling():
    records = singletons("train", "forest", 10) + [
        record("train/sea/0001.jpg"),
        record("test/forest/0099.jpg"),
    ]
    rows = run(
        records,
        exact("c1", "train/forest/0000.jpg", "train/sea/0001.jpg")
        + exact("c2", "test/forest/0099.jpg", "train/forest/0001.jpg"),
    )
    # Both members of a train-only conflict are excluded and keep their labels.
    assert rows["train/forest/0000.jpg"]["split"] == "excluded"
    assert rows["train/forest/0000.jpg"]["label"] == "forest"
    assert rows["train/sea/0001.jpg"]["label"] == "sea"
    assert rows["train/sea/0001.jpg"]["reasons"] == "label_conflict"
    # Exact duplicate between test and train: same label, so not a conflict; train side goes.
    assert rows["test/forest/0099.jpg"]["split"] == "test"
    assert rows["train/forest/0001.jpg"]["reasons"] == "linked_to_test"


def test_conflicting_test_image_is_test_excluded():
    records = [record("test/street/0001.jpg"), record("train/buildings/0001.jpg")]
    rows = run(records, exact("c", "test/street/0001.jpg", "train/buildings/0001.jpg"))

    assert rows["test/street/0001.jpg"]["split"] == "test_excluded"
    assert rows["test/street/0001.jpg"]["reasons"] == "label_conflict"
    assert rows["train/buildings/0001.jpg"]["split"] == "excluded"
    assert rows["train/buildings/0001.jpg"]["reasons"] == "label_conflict;linked_to_test"


def test_training_images_linked_to_test_are_excluded_even_through_pred():
    records = singletons("train", "sea", 10) + [
        record("test/sea/0100.jpg"),
        record("pred/0001.jpg"),
    ]
    rows = run(
        records,
        near_rows=[
            near("train/sea/0003.jpg", "pred/0001.jpg"),
            near("pred/0001.jpg", "test/sea/0100.jpg"),
        ],
    )

    assert rows["train/sea/0003.jpg"]["split"] == "excluded"
    assert rows["train/sea/0003.jpg"]["reasons"] == "linked_to_test"
    assert rows["test/sea/0100.jpg"]["split"] == "test"
    assert rows["pred/0001.jpg"]["split"] == "pred"
    assert rows["pred/0001.jpg"]["reasons"] == "linked_to_train;linked_to_test"
    assert rows["train/sea/0003.jpg"]["linked_to"] == "pred/0001.jpg;test/sea/0100.jpg"


def test_linked_groups_never_cross_train_and_val():
    rng = random.Random(0)
    records = singletons("train", "forest", 300) + singletons("train", "sea", 300)
    paths = [r["relpath"] for r in records]
    near_rows = []
    for label in ("forest", "sea"):
        same = [p for p in paths if f"/{label}/" in p]
        for _ in range(120):
            a, b = rng.sample(same, 2)
            near_rows.append(near(a, b))
    groups = build_groups(records, [], near_rows)
    rows = assign_splits(records, groups, SplitSettings())

    assert leakage_checks(rows) == {"groups_spanning_train_val": 0, "groups_spanning_fit_test": 0}
    for members in groups.values():
        assert len({split_of(rows)[m]["split"] for m in members}) == 1
    assert max(len(m) for m in groups.values()) > 2  # transitive chains were exercised


def test_validation_fraction_is_per_class():
    records = singletons("train", "forest", 200) + singletons("train", "sea", 100)
    rows = run(records)
    val = [r for r in rows.values() if r["split"] == "val"]
    assert sum(r["label"] == "forest" for r in val) == 30
    assert sum(r["label"] == "sea" for r in val) == 15


def test_every_image_is_kept_with_its_original_label(audit_files, tmp_path):
    run_split(
        audit_files["manifest"],
        audit_files["exact"],
        audit_files["near"],
        tmp_path / "splits.csv",
        tmp_path / "summary.json",
    )
    source = {r["relpath"]: r for r in csv.DictReader(audit_files["manifest"].open())}
    output = list(csv.DictReader((tmp_path / "splits.csv").open()))

    assert list(output[0]) == OUTPUT_FIELDS
    assert [r["relpath"] for r in output] == sorted(source)
    assert all(r["label"] == source[r["relpath"]]["label"] for r in output)
    assert all(r["source_split"] == source[r["relpath"]]["split"] for r in output)
    excluded = [r for r in output if r["split"] in ("excluded", "test_excluded")]
    assert excluded and all(r["reasons"] for r in excluded)


def test_summary_counts_and_exclusion_reasons(audit_files, tmp_path):
    summary = run_split(
        audit_files["manifest"],
        audit_files["exact"],
        audit_files["near"],
        tmp_path / "splits.csv",
        tmp_path / "summary.json",
    )

    assert summary == json.loads((tmp_path / "summary.json").read_text())
    assert summary["exclusions"] == {
        "train": {"label_conflict": 2, "linked_to_test": 2},
        "test": {},
    }
    assert summary["test_sets"] == {"clean": 10, "official": 10}
    assert summary["split_counts"]["excluded"] == 4
    assert summary["split_counts"]["train"] + summary["split_counts"]["val"] == 76
    assert summary["pred"] == {"total": 2, "unseen": 1, "linked_to_train": 1, "linked_to_test": 0}
    assert summary["leakage_checks"] == {
        "groups_spanning_train_val": 0,
        "groups_spanning_fit_test": 0,
    }
    assert summary["groups"]["multi_image"] == 5


def test_corrupt_images_are_excluded():
    records = singletons("train", "sea", 5) + [record("test/sea/0009.jpg", ok=False)]
    records[0]["ok"] = "False"
    rows = run(records)
    assert (rows["train/sea/0000.jpg"]["split"], rows["train/sea/0000.jpg"]["reasons"]) == (
        "excluded",
        "corrupt",
    )
    assert rows["test/sea/0009.jpg"]["split"] == "test_excluded"


def test_split_is_reproducible_and_seed_dependent(audit_files, tmp_path):
    def split_bytes(name: str, seed: int) -> bytes:
        out = tmp_path / name
        run_split(
            audit_files["manifest"],
            audit_files["exact"],
            audit_files["near"],
            out / "splits.csv",
            out / "summary.json",
            SplitSettings(seed=seed),
        )
        return (out / "splits.csv").read_bytes()

    assert split_bytes("a", 42) == split_bytes("b", 42)
    assert split_bytes("a", 42) != split_bytes("c", 7)


def test_input_order_does_not_change_the_split():
    records = singletons("train", "forest", 50) + singletons("train", "sea", 50)
    near_rows = [near("train/forest/0001.jpg", "train/forest/0002.jpg")]
    shuffled = records[::-1]

    assert run(records, near_rows=near_rows) == run(shuffled, near_rows=near_rows)


@pytest.mark.parametrize("fraction", [0, 1, -0.1, 1.5])
def test_invalid_validation_fraction_is_rejected(fraction):
    with pytest.raises(SplitError, match="val_fraction"):
        run(singletons("train", "sea", 5), settings=SplitSettings(val_fraction=fraction))


def test_cli_writes_outputs(audit_files, tmp_path, capsys):
    args = [
        "--manifest", str(audit_files["manifest"]),
        "--exact-duplicates", str(audit_files["exact"]),
        "--near-duplicates", str(audit_files["near"]),
        "--output", str(tmp_path / "out" / "splits.csv"),
        "--summary", str(tmp_path / "out" / "summary.json"),
        "--val-fraction", "0.2", "--seed", "3",
    ]  # fmt: skip

    assert main(args) == 0
    assert "leakage checks" in capsys.readouterr().out
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary["settings"] == {"val_fraction": 0.2, "seed": 3}


def test_cli_reports_errors(tmp_path, capsys):
    args = ["--manifest", str(tmp_path / "missing.csv"), "--exact-duplicates", "x"]
    args += ["--near-duplicates", "y", "--output", "o.csv", "--summary", "s.json"]
    assert main(args) == 1
    assert "error:" in capsys.readouterr().err
