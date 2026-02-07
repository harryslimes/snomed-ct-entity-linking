#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts.glinker.io import detect_delimiter, is_active_invalid_reason
from scripts.glinker.rf2_graph import concept_snapshot_path


def _extract_release_id(path: Path) -> str:
    for token in re.findall(r"\d{8}", path.name):
        return token
    return "unknown"


def load_train_concepts(train_annotations: Path) -> set[str]:
    out: set[str] = set()
    with train_annotations.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            cid = (row.get("concept_id") or "").strip()
            if cid:
                out.add(cid)
    return out


def load_rf2_active_concepts(concept_path: Path) -> set[str]:
    out: set[str] = set()
    with concept_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp, delimiter="\t")
        for row in reader:
            if (row.get("active") or "").strip() != "1":
                continue
            cid = (row.get("id") or "").strip()
            if cid:
                out.add(cid)
    return out


def load_athena_active_snomed_codes(athena_concept_path: Path) -> set[str]:
    out: set[str] = set()
    delim = detect_delimiter(athena_concept_path)
    with athena_concept_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp, delimiter=delim)
        for row in reader:
            if (row.get("vocabulary_id") or "").strip() != "SNOMED":
                continue
            if not is_active_invalid_reason(row.get("invalid_reason")):
                continue
            code = (row.get("concept_code") or "").strip()
            if code:
                out.add(code)
    return out


def build_manifest(
    *,
    snomed_dir: Path,
    athena_dir: Path,
    train_annotations: Path,
    min_train_coverage: float,
) -> dict:
    rf2_concept_path = concept_snapshot_path(snomed_dir)
    athena_concept_path = athena_dir / "CONCEPT.csv"
    if not athena_concept_path.exists():
        raise FileNotFoundError(f"Missing Athena concept file: {athena_concept_path}")

    train_concepts = load_train_concepts(train_annotations)
    rf2_concepts = load_rf2_active_concepts(rf2_concept_path)
    athena_concepts = load_athena_active_snomed_codes(athena_concept_path)

    in_rf2 = train_concepts & rf2_concepts
    in_athena = train_concepts & athena_concepts
    in_both = train_concepts & rf2_concepts & athena_concepts
    total = max(1, len(train_concepts))

    rf2_cov = len(in_rf2) / total
    athena_cov = len(in_athena) / total
    both_cov = len(in_both) / total
    passes = rf2_cov >= min_train_coverage and athena_cov >= min_train_coverage

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "snomed_dir": str(snomed_dir),
            "rf2_concept_path": str(rf2_concept_path),
            "rf2_release_id": _extract_release_id(rf2_concept_path),
            "athena_concept_path": str(athena_concept_path),
            "train_annotations": str(train_annotations),
        },
        "counts": {
            "train_unique_concepts": len(train_concepts),
            "rf2_active_concepts": len(rf2_concepts),
            "athena_active_snomed_concepts": len(athena_concepts),
            "train_concepts_in_rf2": len(in_rf2),
            "train_concepts_in_athena": len(in_athena),
            "train_concepts_in_both": len(in_both),
            "train_concepts_missing_rf2": len(train_concepts - rf2_concepts),
            "train_concepts_missing_athena": len(train_concepts - athena_concepts),
        },
        "coverage": {
            "rf2": round(rf2_cov, 6),
            "athena": round(athena_cov, 6),
            "both": round(both_cov, 6),
            "minimum_required": min_train_coverage,
        },
        "status": {
            "pass": passes,
            "reason": (
                "coverage threshold met"
                if passes
                else "coverage below threshold; check terminology release mismatch"
            ),
        },
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Validate that training annotations are covered by active SNOMED concepts "
            "from both RF2 and Athena exports, then write a reproducibility manifest."
        )
    )
    ap.add_argument(
        "--snomed-dir",
        default="data/SnomedCT_InternationalRF2_PRODUCTION_20260101T120000Z",
    )
    ap.add_argument("--athena-dir", default="data/athena")
    ap.add_argument("--train-annotations", default="data/train_annotations.csv")
    ap.add_argument(
        "--out",
        default="outputs/glinker/manifests/data_manifest.json",
    )
    ap.add_argument("--min-train-coverage", type=float, default=0.99)
    args = ap.parse_args(argv)

    snomed_dir = Path(args.snomed_dir)
    athena_dir = Path(args.athena_dir)
    train_annotations = Path(args.train_annotations)
    out = Path(args.out)

    manifest = build_manifest(
        snomed_dir=snomed_dir,
        athena_dir=athena_dir,
        train_annotations=train_annotations,
        min_train_coverage=float(args.min_train_coverage),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(
        "train coverage:"
        f" rf2={manifest['coverage']['rf2']:.4f}"
        f" athena={manifest['coverage']['athena']:.4f}"
        f" both={manifest['coverage']['both']:.4f}"
    )
    print(f"manifest: {out}")
    return 0 if manifest["status"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
