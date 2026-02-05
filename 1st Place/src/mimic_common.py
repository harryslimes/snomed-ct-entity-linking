# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 09:10:31 2024

@author: Yonatan
"""

import os
import re
from collections import Counter

import pandas as pd
from tqdm import tqdm

common_headers = [
    "Allergies",
    "History of Present Illness",
    "Family History",
    "Name",
    "Major Surgical or Invasive Procedure",
    "Admission Date",
    "Discharge Disposition",
    "Past Medical History",
    "Attending",
    "Service",
    "Date of Birth",
    "Discharge Instructions",
    "Discharge Condition",
    "Chief Complaint",
    "Physical Exam",
    "Pertinent Results",
    "Discharge Medications",
    "Social History",
    "Followup Instructions",
    "Medications on Admission",
    "Discharge Diagnosis",
]

internal_blacklist = [
    "other",
    "negative",
    "follow up",
    "mild",
    "normal",
    "inr",
    "changes",
    "change",
    "iv",
]

pattern_cache = {}
_WORD_RE = re.compile(r"[A-Za-z0-9]+")
# get_pattern() does NOT escape these, so treat mentions containing them as unsafe for
# token-based prefiltering (they can change matching semantics).
_UNSAFE_MENTION_RE = re.compile(r"[.^$?|\\\\]")
# Stopwords allowed *between* mention tokens when stopword-transparency is enabled.
# Keep this list conservative; broad stopword sets tend to increase false positives.
_DEFAULT_STOPWORDS = {"of", "the", "a", "an"}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def _env_stopwords() -> set[str]:
    raw = os.environ.get("KIRI_STOPWORDS")
    if raw is None or str(raw).strip() == "":
        return set(_DEFAULT_STOPWORDS)
    parts = re.split(r"[,\s]+", str(raw).strip())
    out = {p.strip().lower() for p in parts if p.strip()}
    return out or set(_DEFAULT_STOPWORDS)


def _escape_token_like_get_pattern(token: str) -> str:
    p = token
    for c in "+(){}[]*":
        p = p.replace(c, f"\\{c}")
    p = p.replace("-", "[- ]")
    p = p.replace("/", "[/ ]")
    return p


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        v = int(str(raw).strip())
    except Exception:
        return default
    return v


def _env_token_set(name: str, default: set[str]) -> set[str]:
    raw = os.environ.get(name)
    if raw is None:
        return set(default)
    if str(raw).strip() == "":
        return set()
    parts = re.split(r"[,\s]+", str(raw).strip())
    out = {p.strip().lower() for p in parts if p.strip()}
    return out or set(default)


class IndexedDict:
    """
    Lightweight prefilter to reduce the number of regex scans per note.

    Preserves matching semantics by only prefiltering "safe" mentions using the
    first two alphanumeric tokens. Mentions containing regex metacharacters that
    get_pattern() doesn't escape are always evaluated.
    """

    def __init__(self, d: dict, prefilter: str = "bigram"):
        if prefilter not in {"bigram", "unigram"}:
            raise ValueError(f"Unknown prefilter mode: {prefilter!r}")
        self._items = list(d.items())  # preserve original dict iteration order
        self._bigram_index: dict[tuple[str, str], list[int]] = {}
        self._unigram_index: dict[str, list[int]] = {}
        self._always: list[int] = []

        for idx, ((section, source_text), cid) in enumerate(self._items):
            s = str(source_text)
            if _UNSAFE_MENTION_RE.search(s):
                self._always.append(idx)
                continue

            toks = _WORD_RE.findall(s.lower())
            if not toks:
                self._always.append(idx)
                continue

            if prefilter == "unigram":
                self._unigram_index.setdefault(toks[0], []).append(idx)
                self._unigram_index.setdefault(toks[0] + "s", []).append(idx)
            elif len(toks) >= 2:
                self._bigram_index.setdefault((toks[0], toks[1]), []).append(idx)
                # get_pattern() appends "s*" to the full pattern, so for 2-token mentions
                # the 2nd token may appear with an extra trailing "s" in text (plural).
                if len(toks) == 2:
                    self._bigram_index.setdefault((toks[0], toks[1] + "s"), []).append(idx)
            else:
                self._unigram_index.setdefault(toks[0], []).append(idx)
                # Same "s*" behavior for single-token mentions (e.g. wheeze -> wheezes).
                self._unigram_index.setdefault(toks[0] + "s", []).append(idx)

    def iter_items_for_text(self, text: str):
        toks = _WORD_RE.findall(text.lower())
        if not toks:
            for idx in self._always:
                yield self._items[idx]
            return

        def _strip_trailing_s(token: str) -> str:
            out = token.rstrip("s")
            return out if out else token

        toks_base = [_strip_trailing_s(t) for t in toks]
        bigrams = set(zip(toks, toks[1:], strict=False))
        bigrams_base = set(zip(toks_base, toks_base[1:], strict=False))
        unigrams = set(toks)
        unigrams_base = set(toks_base)

        idxs = set(self._always)
        for bg in bigrams:
            for idx in self._bigram_index.get(bg, []):
                idxs.add(idx)
        for bg in bigrams_base:
            for idx in self._bigram_index.get(bg, []):
                idxs.add(idx)
        for u in unigrams:
            for idx in self._unigram_index.get(u, []):
                idxs.add(idx)
        for u in unigrams_base:
            for idx in self._unigram_index.get(u, []):
                idxs.add(idx)

        for idx in sorted(idxs):
            yield self._items[idx]


def get_pattern(s):
    stopword_transparent = _env_bool("KIRI_STOPWORD_TRANSPARENT", False)
    stopwords = _env_stopwords() if stopword_transparent else set()
    min_tokens = _env_int("KIRI_STOPWORD_MIN_TOKENS", 3) if stopword_transparent else 0
    allow_2 = (
        _env_token_set("KIRI_STOPWORD_ALLOW_2TOKENS", {"fracture", "fx"})
        if stopword_transparent
        else set()
    )
    cache_key = (
        str(s),
        stopword_transparent,
        tuple(sorted(stopwords)) if stopword_transparent else (),
        int(min_tokens) if stopword_transparent else 0,
        tuple(sorted(allow_2)) if stopword_transparent else (),
    )
    if cache_key in pattern_cache:
        return pattern_cache[cache_key]

    if stopword_transparent:
        p = " ".join(str(s).split())
        tokens = [t for t in p.split(" ") if t]
        if len(tokens) < min_tokens:
            # Allow selected high-precision 2-token patterns (e.g. "fracture femur").
            if not (len(tokens) == 2 and (tokens[0].lower() in allow_2 or tokens[1].lower() in allow_2)):
                stopword_transparent = False
                stopwords = set()

        # Safer semantics: only allow stopwords to appear *between* content tokens in text.
        # Do not drop stopwords that are part of the mention itself, since that can collapse
        # keys and explode matches (e.g., "patient discharge" -> "discharge").
        if stopword_transparent and any(t.lower() in stopwords for t in tokens):
            stopword_transparent = False
            stopwords = set()
        else:
            filtered = tokens
            # Safety: never collapse a multi-token mention down to a single token (or zero).
            if stopword_transparent and len(filtered) < 2:
                stopword_transparent = False
                stopwords = set()
                filtered = []

        if stopword_transparent:
            # Allow arbitrary (possibly repeated) stopwords between content tokens.
            stopword_alt = "|".join(re.escape(w) for w in sorted(stopwords))
            sep = r"[\s,;:/-]+"
            between = rf"(?:{sep}(?:{stopword_alt})\b)*{sep}" if stopword_alt else sep

            pattern_str = _escape_token_like_get_pattern(filtered[0])
            for tok in filtered[1:]:
                pattern_str += between + _escape_token_like_get_pattern(tok)
            pattern_str += "s*"

            try:
                r = re.compile(pattern_str)
            except Exception:
                print(pattern_str)
                return None
            pattern_cache[cache_key] = r
            return r

    p = " ".join(str(s).split())
    for c in "+(){}[]*":
        p = p.replace(c, f"\\{c}")
    p = p.replace(" ", "\\s+")
    p = p.replace("-", "[- ]")
    p = p.replace("/", "[/ ]")
    p = p + "s*"

    try:
        r = re.compile(p)
    except Exception:
        print(p)
        return None
    pattern_cache[cache_key] = r
    return r


def is_in_header(text, pos):
    next_nl = text[pos:].index("\n") + pos
    if text[:next_nl].strip().endswith(":"):
        return True
    return False


def get_header_by_pos(pos, headers_position, pos_header, legal_headers):
    prev_pos = [p for p in headers_position if p <= pos]
    if len(prev_pos) == 0:
        return None
    header_pos = max(prev_pos)
    h = pos_header[header_pos]

    # Normalize case and optional trailing ":" against known headers.
    h_clean = str(h).strip()
    h_no_colon = h_clean[:-1] if h_clean.endswith(":") else h_clean
    legal_map = {str(lh).rstrip(":").strip().casefold(): str(lh).rstrip(":").strip() for lh in legal_headers}
    key = h_no_colon.strip().casefold()
    if key in legal_map:
        return legal_map[key] + ":"
    if h_clean not in legal_headers and h_no_colon not in legal_headers:
        return "other"
    if h[-1] != ":":
        return h + ":"
    return h


def get_sections(text, headers):
    pos_header = {text.find(h + ":"): h + ":" for h in headers if h + ":" in text}
    add_break_lines(pos_header, text)
    positions = list(pos_header.keys())
    positions.sort()
    positions.append(len(text))
    return positions, pos_header


def add_break_lines(pos_header, text):
    prev_line_is_header = False
    lines = text.split("\n")
    prev_pos = None
    for i, l in enumerate(lines):
        l = l.strip()
        if len(l) == 0:
            continue
        pos = sum([len(line) + 1 for line in lines[:i]])
        assert text[pos : pos + len(lines[i])] == lines[i]
        if l.endswith(":") and i < len(lines) - 1 and len(lines[i + 1].strip()) == 0:
            pos_header[pos] = l
            prev_line_is_header = True
        elif (
            all([c in "-=_:" for c in l])
            and not prev_line_is_header
            and len(lines[i - 1].strip()) > 0
        ):
            pos_header[prev_pos] = lines[i - 1].strip()
        else:
            prev_line_is_header = False
        prev_pos = pos


def annotate_with_dict(text, d, headers, note_id, keep_overlaps=False):
    rows = []
    h_positions, pos_header = get_sections(text, headers)
    if isinstance(d, IndexedDict):
        items_iter = d.iter_items_for_text(text)
    else:
        items_iter = d.items()
    add_row = rows.append
    for (section, source_text), cid in items_iter:

        p = get_pattern(source_text)
        if p is None:
            continue
        for match in p.finditer(text):
            i = match.start()
            j = match.end()
            if i < 100:
                continue
            if text[i - 1].isalnum() or text[j].isalnum():
                continue
            if is_in_header(text, i) and not keep_overlaps:
                continue
            h = get_header_by_pos(i, h_positions, pos_header, headers)
            if h is None:
                return
            if "medication" in h.lower() or "service" in h.lower() or "date of birth" in h.lower():
                continue

            if h == section or h in section or section == "any":
                if type(cid) == Counter:
                    for k in cid:
                        add_row([note_id, i, j, k, section, source_text])
                else:
                    add_row([note_id, i, j, cid, section, source_text])

    ann = pd.DataFrame(
        rows,
        columns=["note_id", "start", "end", "concept_id", "section", "dict_entry"],
    )
    if keep_overlaps:
        return ann
    return remove_overlaps(ann)


def shorter_span(i, j, l):
    if l.iloc[i] < l.iloc[j]:
        return i
    return j


def remove_overlaps(df, verbose=False):
    log = []
    df = df.sort_values("start").reset_index(drop=True)
    length = df["end"] - df["start"]
    length = length.astype(float)
    section_any = [type(s) == tuple or s == "any" for s in df["section"]]
    length[
        section_any
    ] -= 0.1  # if the section is "other" then we prefer the same span with a section-based annotation
    to_remove = set()
    n = len(df)
    for i in range(n):
        if df.index[i] in to_remove:
            continue
        for j in range(i + 1, n):
            if df["start"].iloc[j] >= df["end"].iloc[i]:
                break
            remove_index = shorter_span(i, j, length)
            if verbose:
                log.append(
                    f'overalpping segments {df.iloc[i]["start"]}-{df.iloc[i]["end"]} and {df.iloc[j]["start"]}-{df.iloc[j]["end"]}'
                )
            to_remove.add(df.index[remove_index])
            if remove_index == i:
                break

    df2 = df.drop(to_remove)
    for i in to_remove:
        s, e = df.loc[i, ["start", "end"]].values
        overlaps = ((df2["start"] <= s) & (df2["end"] > s)) | (
            (df2["start"] <= e) & (df2["end"] > e)
        )
        if overlaps.sum() == 0:
            if verbose:
                log.append(f'returning segment {df.loc[i]["start"]}-{df.loc[i]["end"]}')
            df2.loc[i] = df.loc[i]

    if verbose:
        return df2, log
    return df2


def spans_overlap(s1, s2):
    return s1["start"] <= s2["start"] < s1["end"]


def check_for_overlaps(pred):
    for ni in tqdm(pred["note_id"].unique()):
        df = pred.query(f'note_id == "{ni}"')
        df = df.sort_values("start")
        for i in range(len(df)):
            for j in range(i + 1, len(df)):
                if spans_overlap(df.iloc[i], df.iloc[j]):
                    return df, i, j
    print("No overlaps")
    return None
