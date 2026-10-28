"""Leakage-aware train/validation/test split for the Intel Image Classification dataset.

Images linked by exact duplicates or near-duplicate candidates from the audit are
merged transitively into groups. The policy:

* Images and labels are never changed, and every audited image appears in the output.
* Groups whose labelled members disagree are excluded from training, and their test
  members are marked ``test_excluded``.
* Training images grouped with any test image are excluded, so leakage is always
  resolved on the training side and test images are never moved.
* Validation takes a fixed fraction of each class as whole groups, chosen with a
  seeded shuffle, so no group spans train and validation.
* Prediction images keep the ``pred`` split and record whether they are linked to
  training or test images.

Output splits: train, val, test (clean), test_excluded, excluded (training images
removed from use), pred. The official test set is test + test_excluded.

Usage:
    python -m deepvision.data.split --manifest <csv> --exact-duplicates <csv>
        --near-duplicates <csv> --output <csv> --summary <json>
"""

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

SPLITS = ("train", "val", "test", "test_excluded", "excluded", "pred")
OUTPUT_FIELDS = [
    "relpath", "source_split", "label", "split", "group_id", "group_size",
    "reasons", "linked_to", "sha256",
]  # fmt: skip


class SplitError(Exception):
    """Raised when inputs are inconsistent or the split would leak between sets."""


@dataclass(frozen=True)
class SplitSettings:
    val_fraction: float = 0.15
    seed: int = 42


def read_csv(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class _UnionFind:
    """Disjoint sets whose root is always the lexicographically smallest member."""

    def __init__(self, items: Iterable[str]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, a: str, b: str) -> None:
        for item in (a, b):
            if item not in self.parent:
                raise SplitError(f"duplicate report references an unknown image: {item}")
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            low, high = sorted((root_a, root_b))
            self.parent[high] = low


def build_groups(
    records: list[dict], exact_rows: list[dict], near_rows: list[dict]
) -> dict[str, list[str]]:
    """Merge exact-duplicate groups and near-duplicate pairs transitively.

    Returns {group_id: sorted members}; the group id is the smallest member path.
    """
    union_find = _UnionFind(r["relpath"] for r in records)
    exact_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in exact_rows:
        exact_groups[(row["kind"], row["hash"])].append(row["relpath"])
    for members in exact_groups.values():
        for other in members[1:]:
            union_find.union(members[0], other)
    for row in near_rows:
        union_find.union(row["relpath_a"], row["relpath_b"])

    groups: dict[str, list[str]] = defaultdict(list)
    for relpath in sorted(union_find.parent):
        groups[union_find.find(relpath)].append(relpath)
    return dict(groups)


def assign_splits(
    records: list[dict], groups: dict[str, list[str]], settings: SplitSettings
) -> list[dict]:
    """Apply the split policy and return one output row per image, sorted by path."""
    if not 0 < settings.val_fraction < 1:
        raise SplitError(f"val_fraction must be between 0 and 1: {settings.val_fraction}")
    by_path = {r["relpath"]: r for r in records}
    rows: dict[str, dict] = {}
    trainable: dict[str, list[tuple[str, list[str]]]] = defaultdict(list)

    for group_id, members in sorted(groups.items()):
        group = [by_path[m] for m in members]
        sources = {r["split"] for r in group}
        conflict = len({r["label"] for r in group if r["label"]}) > 1
        usable: list[str] = []
        for record in group:
            source, reasons = record["split"], []
            if record["ok"] != "True":
                reasons.append("corrupt")
            if source == "pred":
                split = "pred"
                reasons += [f"linked_to_{s}" for s in ("train", "test") if s in sources]
            elif source == "test":
                if conflict:
                    reasons.append("label_conflict")
                split = "test_excluded" if reasons else "test"
            elif source == "train":
                if conflict:
                    reasons.append("label_conflict")
                if "test" in sources:
                    reasons.append("linked_to_test")
                split = "excluded" if reasons else ""
                if not reasons:
                    usable.append(record["relpath"])
            else:
                raise SplitError(f"unknown source split {source!r} for {record['relpath']}")
            rows[record["relpath"]] = {
                "relpath": record["relpath"],
                "source_split": source,
                "label": record["label"],
                "split": split,
                "group_id": group_id,
                "group_size": len(members),
                "reasons": ";".join(reasons),
                "linked_to": ";".join(m for m in members if m != record["relpath"]),
                "sha256": record["sha256"],
            }
        if usable:
            # Conflicting groups are never usable, so a usable group has one label.
            trainable[by_path[usable[0]]["label"]].append((group_id, usable))

    rng = random.Random(settings.seed)
    for label in sorted(trainable):
        label_groups = sorted(trainable[label])
        rng.shuffle(label_groups)
        target = round(sum(len(m) for _, m in label_groups) * settings.val_fraction)
        taken = 0
        for _, members in label_groups:
            split = "val" if taken + len(members) <= target else "train"
            if split == "val":
                taken += len(members)
            for relpath in members:
                rows[relpath]["split"] = split

    return [rows[relpath] for relpath in sorted(rows)]


def leakage_checks(rows: list[dict]) -> dict[str, int]:
    """Count groups that span train/val or train-or-val/test; both must be zero."""
    splits_by_group: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        splits_by_group[row["group_id"]].add(row["split"])
    fit, held_out = {"train", "val"}, {"test", "test_excluded"}
    return {
        "groups_spanning_train_val": sum(fit <= s for s in splits_by_group.values()),
        "groups_spanning_fit_test": sum(
            bool(s & fit) and bool(s & held_out) for s in splits_by_group.values()
        ),
    }


def summarize(rows: list[dict], groups: dict[str, list[str]], settings: SplitSettings) -> dict:
    def class_counts(splits: set[str]) -> dict[str, int]:
        return dict(sorted(Counter(r["label"] for r in rows if r["split"] in splits).items()))

    def reason_counts(source: str) -> dict[str, int]:
        reasons = Counter(
            reason
            for r in rows
            if r["source_split"] == source and r["split"] in ("excluded", "test_excluded")
            for reason in r["reasons"].split(";")
        )
        return dict(sorted(reasons.items()))

    train, val = class_counts({"train"}), class_counts({"val"})
    pred = [r for r in rows if r["split"] == "pred"]
    sizes = [len(m) for m in groups.values()]
    return {
        "settings": {"val_fraction": settings.val_fraction, "seed": settings.seed},
        "images": {"total": len(rows)},
        "split_counts": {s: sum(r["split"] == s for r in rows) for s in SPLITS},
        "test_sets": {
            "clean": sum(r["split"] == "test" for r in rows),
            "official": sum(r["source_split"] == "test" for r in rows),
        },
        "class_counts": {
            "train": train,
            "val": val,
            "test_clean": class_counts({"test"}),
            "test_official": class_counts({"test", "test_excluded"}),
        },
        "val_fraction_by_class": {
            label: round(val.get(label, 0) / (train.get(label, 0) + val.get(label, 0)), 4)
            for label in sorted(set(train) | set(val))
        },
        "exclusions": {"train": reason_counts("train"), "test": reason_counts("test")},
        "pred": {
            "total": len(pred),
            "unseen": sum(not r["reasons"] for r in pred),
            "linked_to_train": sum("linked_to_train" in r["reasons"] for r in pred),
            "linked_to_test": sum("linked_to_test" in r["reasons"] for r in pred),
        },
        "groups": {
            "total": len(groups),
            "multi_image": sum(size > 1 for size in sizes),
            "largest": max(sizes, default=0),
        },
        "leakage_checks": leakage_checks(rows),
    }


def run_split(
    manifest_path: Path,
    exact_path: Path,
    near_path: Path,
    output_path: Path,
    summary_path: Path,
    settings: SplitSettings | None = None,
) -> dict:
    """Build the split from audit reports, write the manifest and summary, return the summary."""
    settings = settings or SplitSettings()
    records = read_csv(manifest_path)
    if not records:
        raise SplitError(f"audit manifest is empty: {manifest_path}")
    groups = build_groups(records, read_csv(exact_path), read_csv(near_path))
    rows = assign_splits(records, groups, settings)
    summary = summarize(rows, groups, settings)
    if any(summary["leakage_checks"].values()):
        raise SplitError(f"split would leak between sets: {summary['leakage_checks']}")

    output_path, summary_path = Path(output_path), Path(summary_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    defaults = SplitSettings()
    parser = argparse.ArgumentParser(
        prog="python -m deepvision.data.split",
        description="Create a leakage-aware train/val/test split from the dataset audit.",
    )
    parser.add_argument("--manifest", type=Path, required=True, help="audit manifest CSV")
    parser.add_argument("--exact-duplicates", type=Path, required=True)
    parser.add_argument("--near-duplicates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="split manifest CSV")
    parser.add_argument("--summary", type=Path, required=True, help="summary JSON")
    parser.add_argument("--val-fraction", type=float, default=defaults.val_fraction)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = SplitSettings(val_fraction=args.val_fraction, seed=args.seed)
    try:
        summary = run_split(
            args.manifest,
            args.exact_duplicates,
            args.near_duplicates,
            args.output,
            args.summary,
            settings,
        )
    except (SplitError, OSError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    counts = summary["split_counts"]
    print(", ".join(f"{split} {counts[split]}" for split in SPLITS))
    print(
        f"test: clean {summary['test_sets']['clean']}, official {summary['test_sets']['official']}"
    )
    print(f"excluded: {summary['exclusions']}")
    print(f"leakage checks: {summary['leakage_checks']}")
    print(f"split manifest written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
