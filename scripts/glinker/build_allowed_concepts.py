#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts.glinker.rf2_graph import ROOTS, assign_l1_type, descendants_by_root


def load_train_concepts(train_annotations: Path) -> set[str]:
    out: set[str] = set()
    with train_annotations.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            cid = (row.get("concept_id") or "").strip()
            if cid:
                out.add(cid)
    return out


def build_allowed_concepts_df(
    *,
    snomed_dir: Path,
    train_annotations: Path,
) -> pd.DataFrame:
    by_root = descendants_by_root(snomed_dir, roots=ROOTS)
    train_concepts = load_train_concepts(train_annotations)

    allowed_ids: set[str] = set()
    for descendants in by_root.values():
        allowed_ids.update(descendants)
    allowed_ids.update(train_concepts)

    rows = []
    for concept_id in sorted(allowed_ids, key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x)):
        root_hits = sorted([root for root, descendants in by_root.items() if concept_id in descendants])
        l1_type = assign_l1_type(concept_id, by_root, roots=ROOTS)
        in_train = concept_id in train_concepts
        is_descendant = len(root_hits) > 0
        train_override = in_train and not is_descendant

        rows.append(
            {
                "concept_id": concept_id,
                "l1_type": l1_type or "unknown",
                "root_hits": ",".join(root_hits),
                "is_descendant": bool(is_descendant),
                "in_train": bool(in_train),
                "train_override": bool(train_override),
            }
        )

    return pd.DataFrame(rows)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Build the allowed concept universe for this task: descendants of "
            "Clinical finding / Procedure / Body structure, plus train concept safety override."
        )
    )
    ap.add_argument(
        "--snomed-dir",
        default="data/SnomedCT_InternationalRF2_PRODUCTION_20260101T120000Z",
    )
    ap.add_argument("--train-annotations", default="data/train_annotations.csv")
    ap.add_argument(
        "--out-parquet",
        default="data/interim/glinker/allowed_concepts.parquet",
    )
    ap.add_argument(
        "--out-csv",
        default="data/interim/glinker/allowed_concepts.csv",
    )
    ap.add_argument(
        "--strict-parquet",
        action="store_true",
        help="Fail if parquet cannot be written (requires pyarrow/fastparquet).",
    )
    args = ap.parse_args(argv)

    df = build_allowed_concepts_df(
        snomed_dir=Path(args.snomed_dir),
        train_annotations=Path(args.train_annotations),
    )

    out_parquet = Path(args.out_parquet)
    out_csv = Path(args.out_csv)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    parquet_written = True
    try:
        df.to_parquet(out_parquet, index=False)
    except Exception as e:
        parquet_written = False
        if args.strict_parquet:
            raise RuntimeError(
                f"Failed to write parquet at {out_parquet}. Install parquet support (pyarrow)."
            ) from e
        print(
            f"warning: parquet write failed ({e.__class__.__name__}). "
            f"CSV was still written to {out_csv}."
        )

    df.to_csv(out_csv, index=False)

    print(f"allowed concepts: {len(df):,}")
    print(f"train overrides: {int(df['train_override'].sum()):,}")
    if parquet_written:
        print(f"wrote: {out_parquet}")
    else:
        print("wrote: parquet skipped")
    print(f"wrote: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
