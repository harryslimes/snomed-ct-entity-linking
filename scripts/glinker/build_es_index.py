#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts.glinker.io import has_alnum, normalize_alias


DEFAULT_INDEX_BODY = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "analysis": {
            "normalizer": {
                "lowercase_norm": {
                    "type": "custom",
                    "char_filter": [],
                    "filter": ["lowercase", "asciifolding"],
                }
            }
        },
    },
    "mappings": {
        "dynamic": "strict",
        "properties": {
            "concept_id": {"type": "keyword"},
            "l1_type": {"type": "keyword"},
            "alias": {"type": "text"},
            "alias_exact": {"type": "keyword", "normalizer": "lowercase_norm"},
            "alias_len": {"type": "integer"},
            "priority": {"type": "integer"},
            "source_count": {"type": "integer"},
        },
    },
}


@dataclass(frozen=True)
class AliasEntry:
    alias: str
    concept_id: str
    l1_type: str
    priority: int
    source_count: int


def iter_alias_entries(super_jsonl: Path) -> Iterable[AliasEntry]:
    with super_jsonl.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            concept_id = str(obj.get("id") or "").strip()
            l1_type = str(obj.get("l1_type") or obj.get("hierarchy") or "").strip()
            if not concept_id:
                continue

            names = obj.get("names") or []
            sources = obj.get("sources") or []
            source_count = len(sources)
            seen_aliases: set[str] = set()
            for priority, alias_raw in enumerate(names):
                alias = normalize_alias(str(alias_raw))
                if not alias or not has_alnum(alias):
                    continue
                if alias in seen_aliases:
                    continue
                seen_aliases.add(alias)
                yield AliasEntry(
                    alias=alias,
                    concept_id=concept_id,
                    l1_type=l1_type,
                    priority=int(priority),
                    source_count=source_count,
                )


def write_exact_dictionary_tsv(
    entries: Iterable[AliasEntry],
    out_tsv: Path,
    *,
    max_candidates_per_alias: int = 0,
) -> dict[str, int]:
    alias_to_rows: dict[str, list[AliasEntry]] = defaultdict(list)
    row_count = 0
    concept_ids: set[str] = set()
    for e in entries:
        alias_to_rows[e.alias].append(e)
        concept_ids.add(e.concept_id)
        row_count += 1

    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with out_tsv.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp, delimiter="\t")
        writer.writerow(["alias", "concept_id", "l1_type", "priority"])
        for alias in sorted(alias_to_rows):
            rows = sorted(alias_to_rows[alias], key=lambda x: (x.priority, x.concept_id))
            if max_candidates_per_alias > 0:
                rows = rows[: max_candidates_per_alias]
            for e in rows:
                writer.writerow([e.alias, e.concept_id, e.l1_type, e.priority])
                kept += 1

    return {
        "alias_doc_rows": row_count,
        "unique_aliases": len(alias_to_rows),
        "unique_concepts": len(concept_ids),
        "rows_written": kept,
    }


def _doc_id(entry: AliasEntry) -> str:
    digest = hashlib.sha1(f"{entry.concept_id}\t{entry.alias}".encode("utf-8")).hexdigest()
    return digest


def _load_index_body(path: Path | None) -> dict:
    if path is None:
        return dict(DEFAULT_INDEX_BODY)
    return json.loads(path.read_text(encoding="utf-8"))


def _es_request(session: requests.Session, method: str, url: str, **kwargs):
    r = session.request(method, url, **kwargs)
    if r.status_code >= 400:
        raise RuntimeError(f"ES {method} {url} failed ({r.status_code}): {r.text[:500]}")
    return r


def _prepare_index(
    *,
    session: requests.Session,
    base_url: str,
    index_name: str,
    index_body: dict,
    recreate: bool,
) -> None:
    base = base_url.rstrip("/")
    index_url = f"{base}/{index_name}"
    exists = session.head(index_url, timeout=30).status_code == 200
    if exists and recreate:
        _es_request(session, "DELETE", index_url, timeout=60)
        exists = False
    if not exists:
        _es_request(session, "PUT", index_url, json=index_body, timeout=60)


def _bulk_index(
    *,
    session: requests.Session,
    base_url: str,
    index_name: str,
    entries: Iterable[AliasEntry],
    bulk_size: int,
) -> int:
    base = base_url.rstrip("/")
    bulk_url = f"{base}/_bulk"
    n_indexed = 0
    payload_lines: list[str] = []
    for e in entries:
        meta = {"index": {"_index": index_name, "_id": _doc_id(e)}}
        doc = {
            "concept_id": e.concept_id,
            "l1_type": e.l1_type,
            "alias": e.alias,
            "alias_exact": e.alias,
            "alias_len": len(e.alias),
            "priority": e.priority,
            "source_count": e.source_count,
        }
        payload_lines.append(json.dumps(meta, ensure_ascii=True))
        payload_lines.append(json.dumps(doc, ensure_ascii=True))
        if len(payload_lines) >= 2 * bulk_size:
            _send_bulk(session, bulk_url, payload_lines)
            n_indexed += len(payload_lines) // 2
            payload_lines = []

    if payload_lines:
        _send_bulk(session, bulk_url, payload_lines)
        n_indexed += len(payload_lines) // 2
    _es_request(session, "POST", f"{base}/{index_name}/_refresh", timeout=60)
    return n_indexed


def _send_bulk(session: requests.Session, bulk_url: str, lines: list[str]) -> None:
    body = "\n".join(lines) + "\n"
    r = _es_request(
        session,
        "POST",
        bulk_url,
        data=body.encode("utf-8"),
        headers={"Content-Type": "application/x-ndjson"},
        timeout=120,
    )
    payload = r.json()
    if payload.get("errors"):
        failures = []
        for item in payload.get("items", []):
            index_obj = item.get("index", {})
            if int(index_obj.get("status", 200)) >= 300:
                failures.append(index_obj)
                if len(failures) >= 3:
                    break
        raise RuntimeError(f"Bulk indexing had errors. Sample failures: {failures}")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Build L2 exact dictionary and optional Elasticsearch fuzzy index "
            "from super_dictionary_scoped.jsonl."
        )
    )
    ap.add_argument(
        "--super-dict-jsonl",
        default="data/interim/glinker/super_dictionary_scoped.jsonl",
    )
    ap.add_argument(
        "--exact-out-tsv",
        default="data/interim/glinker/l2_exact_dictionary.tsv",
    )
    ap.add_argument("--max-candidates-per-alias", type=int, default=0)
    ap.add_argument("--es-url", default="http://127.0.0.1:9200")
    ap.add_argument("--es-index-name", default="snomed_super_dict_v1")
    ap.add_argument("--es-api-key", default="")
    ap.add_argument("--es-index-body", default="configs/es_index.json")
    ap.add_argument("--bulk-size", type=int, default=5000)
    ap.add_argument("--no-es", action="store_true")
    ap.add_argument("--recreate-index", action="store_true")
    args = ap.parse_args(argv)

    super_jsonl = Path(args.super_dict_jsonl)
    entries = list(iter_alias_entries(super_jsonl))
    stats = write_exact_dictionary_tsv(
        entries,
        Path(args.exact_out_tsv),
        max_candidates_per_alias=int(args.max_candidates_per_alias),
    )
    print(
        "exact dictionary:"
        f" aliases={stats['unique_aliases']:,}"
        f" concepts={stats['unique_concepts']:,}"
        f" rows={stats['rows_written']:,}"
    )
    print(f"exact output: {args.exact_out_tsv}")

    if args.no_es:
        print("skipped ES indexing (--no-es)")
        return 0

    index_body_path = Path(args.es_index_body)
    index_body = _load_index_body(index_body_path if index_body_path.exists() else None)

    session = requests.Session()
    if args.es_api_key:
        session.headers.update({"Authorization": f"ApiKey {args.es_api_key}"})
    session.headers.update({"Content-Type": "application/json"})
    try:
        _prepare_index(
            session=session,
            base_url=args.es_url,
            index_name=args.es_index_name,
            index_body=index_body,
            recreate=bool(args.recreate_index),
        )
        n = _bulk_index(
            session=session,
            base_url=args.es_url,
            index_name=args.es_index_name,
            entries=entries,
            bulk_size=int(args.bulk_size),
        )
    finally:
        session.close()

    print(f"indexed ES docs: {n:,}")
    print(f"es index: {args.es_index_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

