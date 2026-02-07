#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path


def detect_delimiter(path: Path) -> str:
    with path.open("rb") as fp:
        sample = fp.read(4096)
    return "\t" if sample.count(b"\t") >= sample.count(b",") else ","


def normalize_alias(text: str) -> str:
    return " ".join(str(text).strip().split()).lower()


def has_alnum(text: str) -> bool:
    return any(ch.isalnum() for ch in text)


def is_active_invalid_reason(value: str | None) -> bool:
    if value is None:
        return True
    value = value.strip()
    return value == "" or value.upper() == "NULL"


def maybe_int_sort_key(value: str) -> tuple[int, str]:
    if value.isdigit():
        return (0, f"{int(value):020d}")
    return (1, value)


def iter_dict_rows(path: Path, *, delimiter: str | None = None):
    delim = delimiter or detect_delimiter(path)
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp, delimiter=delim)
        for row in reader:
            yield row

