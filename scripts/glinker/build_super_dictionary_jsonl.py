#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts.glinker.io import has_alnum, normalize_alias


_SOURCE_RANK = {
    "train_span|annotation_span": 0,
    "athena|concept_synonym": 1,
    "athena|concept_name": 2,
    "snomed_rf2|SYN": 3,
    "snomed_rf2|FSN": 4,
}


def load_allowed_map(path: Path) -> dict[str, str]:
    if not path.exists() and path.suffix.lower() == ".parquet":
        csv_fallback = path.with_suffix(".csv")
        if csv_fallback.exists():
            path = csv_fallback
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if "concept_id" not in df.columns or "l1_type" not in df.columns:
        raise ValueError(f"{path} must contain columns: concept_id,l1_type")
    out = {
        str(cid): str(l1_type)
        for cid, l1_type in df[["concept_id", "l1_type"]].itertuples(index=False, name=None)
    }
    return out


def _rank_for(source: str, source_detail: str) -> int:
    key = f"{source}|{source_detail}"
    return _SOURCE_RANK.get(key, 99)


def export_super_dictionary_jsonl(
    *,
    super_dict_tsv: Path,
    allowed_map: dict[str, str],
    out_jsonl: Path,
    max_aliases_per_concept: int = 256,
    min_alias_len: int = 2,
) -> dict[str, int]:
    aliases_by_concept: dict[str, dict[str, tuple[int, str]]] = defaultdict(dict)
    sources_by_concept: dict[str, set[str]] = defaultdict(set)

    with super_dict_tsv.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp, delimiter="\t")
        for row in reader:
            concept_id = (row.get("snomed_concept_id") or "").strip()
            if concept_id not in allowed_map:
                continue

            alias = normalize_alias(row.get("term") or "")
            if len(alias) < min_alias_len or not has_alnum(alias):
                continue

            source = (row.get("source") or "").strip()
            source_detail = (row.get("source_detail") or "").strip()
            rank = _rank_for(source, source_detail)
            prev = aliases_by_concept[concept_id].get(alias)
            if prev is None or rank < prev[0]:
                aliases_by_concept[concept_id][alias] = (rank, source)
            sources_by_concept[concept_id].add(source)

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    concept_count = 0
    alias_count = 0
    with out_jsonl.open("w", encoding="utf-8") as fp:
        for concept_id in sorted(aliases_by_concept.keys(), key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x)):
            ranked_aliases = sorted(
                aliases_by_concept[concept_id].items(),
                key=lambda kv: (kv[1][0], len(kv[0]), kv[0]),
            )
            names = [alias for alias, _ in ranked_aliases[: max(1, int(max_aliases_per_concept))]]
            if not names:
                continue
            record = {
                "id": concept_id,
                "names": names,
                "hierarchy": allowed_map[concept_id],
                "l1_type": allowed_map[concept_id],
                "sources": sorted(sources_by_concept[concept_id]),
            }
            fp.write(json.dumps(record, ensure_ascii=True) + "\n")
            concept_count += 1
            alias_count += len(names)

    return {
        "concept_count": concept_count,
        "alias_count": alias_count,
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Build scoped super dictionary JSONL for GLinker from super_dictionary_full.tsv "
            "and allowed concept universe."
        )
    )
    ap.add_argument("--super-dict-tsv", default="data/interim/super_dictionary_full.tsv")
    ap.add_argument("--allowed-concepts", default="data/interim/glinker/allowed_concepts.parquet")
    ap.add_argument("--out-jsonl", default="data/interim/glinker/super_dictionary_scoped.jsonl")
    ap.add_argument("--max-aliases-per-concept", type=int, default=256)
    ap.add_argument("--min-alias-len", type=int, default=2)
    args = ap.parse_args(argv)

    allowed_map = load_allowed_map(Path(args.allowed_concepts))
    stats = export_super_dictionary_jsonl(
        super_dict_tsv=Path(args.super_dict_tsv),
        allowed_map=allowed_map,
        out_jsonl=Path(args.out_jsonl),
        max_aliases_per_concept=int(args.max_aliases_per_concept),
        min_alias_len=int(args.min_alias_len),
    )
    print(f"concepts: {stats['concept_count']:,}")
    print(f"aliases: {stats['alias_count']:,}")
    print(f"wrote: {args.out_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
