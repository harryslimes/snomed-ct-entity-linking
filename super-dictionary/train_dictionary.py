#!/usr/bin/env python3
"""Train a filtered section-aware dictionary from training annotations.

Replicates the key components of the KIRI training pipeline:
1. Build dictionary from training annotation spans (section, mention) -> concept_id
2. Score entries against training data for precision
3. Remove low-precision entries ("bad keys")
4. Extract uppercase-only mentions into a separate dictionary
5. Add SNOMED synonyms from the super dictionary
6. Add linguistic variants (abbreviation expansion, fracture permutations)
7. Add word replacements and permutations
8. Limit "any"-section keys to allowed sections per concept type

Output: a pickle file containing the trained dict and uppercase dict, used by
test_old_challenge_split.py --trained-dict.
"""
from __future__ import annotations

import argparse
import csv
import os
import pickle
import re
import sys
import time
from collections import Counter, defaultdict
from itertools import permutations as iter_permutations
from pathlib import Path
from typing import Optional

import pandas as pd

from engine import COMMON_HEADERS

# ---------------------------------------------------------------------------
# Configuration (mirrors KIRI defaults)
# ---------------------------------------------------------------------------
CORRECT_FRAC_FOR_DICT = 0.2   # precision threshold for section-specific keys
CORRECT_FRAC_FOR_ANY = 0.3    # precision threshold for "any"-section keys
BLACKLIST_THRESH = 2000        # word-frequency threshold for dynamic blacklist
SNOMED_MIN_LEN = 2             # min token count for SNOMED synonyms
SNOMED_MAX_LEN = 5             # max token count for SNOMED synonyms

INTERNAL_BLACKLIST = [
    "other", "negative", "follow up", "mild", "normal", "inr",
    "changes", "change", "iv",
]

_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_UNSAFE_MENTION_RE = re.compile(r"[.^$?|\\\\]")

# ---------------------------------------------------------------------------
# Pattern building (same as KIRI's get_pattern)
# ---------------------------------------------------------------------------
_pattern_cache: dict[str, re.Pattern | None] = {}


def get_pattern(s: str) -> re.Pattern | None:
    if s in _pattern_cache:
        return _pattern_cache[s]
    p = " ".join(str(s).split())
    for c in "+(){}[]*":
        p = p.replace(c, f"\\{c}")
    p = p.replace(" ", r"\s+")
    p = p.replace("-", "[- ]")
    p = p.replace("/", "[/ ]")
    p = p + "s*"
    try:
        r = re.compile(p)
    except Exception:
        _pattern_cache[s] = None
        return None
    _pattern_cache[s] = r
    return r


# ---------------------------------------------------------------------------
# Section detection (same as KIRI)
# ---------------------------------------------------------------------------
def get_sections(text: str, headers: list[str]) -> tuple[list[int], dict[int, str]]:
    pos_header = {}
    for h in headers:
        key = h + ":"
        idx = text.find(key)
        if idx >= 0:
            pos_header[idx] = key

    # Also detect break-line-style headers
    lines = text.split("\n")
    prev_pos = None
    prev_line_is_header = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        pos = sum(len(l) + 1 for l in lines[:i])
        if stripped.endswith(":") and i < len(lines) - 1 and not lines[i + 1].strip():
            pos_header[pos] = stripped
            prev_line_is_header = True
        elif (
            all(c in "-=_:" for c in stripped)
            and not prev_line_is_header
            and prev_pos is not None
            and i > 0
            and lines[i - 1].strip()
        ):
            pos_header[prev_pos] = lines[i - 1].strip()
        else:
            prev_line_is_header = False
        prev_pos = pos

    positions = sorted(pos_header.keys())
    positions.append(len(text))
    return positions, pos_header


def get_header_by_pos(
    pos: int,
    h_positions: list[int],
    pos_header: dict[int, str],
    headers: list[str],
) -> str:
    prev = [p for p in h_positions if p <= pos]
    if not prev:
        return "other"
    header_pos = max(prev)
    h = pos_header.get(header_pos, "other")

    h_clean = str(h).strip()
    h_no_colon = h_clean[:-1] if h_clean.endswith(":") else h_clean
    legal_map = {
        str(lh).rstrip(":").strip().casefold(): str(lh).rstrip(":").strip()
        for lh in headers
    }
    key = h_no_colon.strip().casefold()
    if key in legal_map:
        return legal_map[key] + ":"
    if h_clean not in headers and h_no_colon not in headers:
        return "other"
    return h if h.endswith(":") else h + ":"


def is_in_header(text: str, pos: int) -> bool:
    nl = text.find("\n", pos)
    if nl < 0:
        nl = len(text)
    return text[:nl].strip().endswith(":")


# ---------------------------------------------------------------------------
# IndexedDict (same as KIRI)
# ---------------------------------------------------------------------------
class IndexedDict:
    def __init__(self, d: dict, prefilter: str = "bigram"):
        self._items = list(d.items())
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
                if len(toks) == 2:
                    self._bigram_index.setdefault((toks[0], toks[1] + "s"), []).append(idx)
            else:
                self._unigram_index.setdefault(toks[0], []).append(idx)
                self._unigram_index.setdefault(toks[0] + "s", []).append(idx)

    def iter_items_for_text(self, text: str):
        toks = _WORD_RE.findall(text.lower())
        if not toks:
            for idx in self._always:
                yield self._items[idx]
            return

        def _strip_s(t):
            out = t.rstrip("s")
            return out if out else t

        toks_base = [_strip_s(t) for t in toks]
        bigrams = set(zip(toks, toks[1:]))
        bigrams_base = set(zip(toks_base, toks_base[1:]))
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


# ---------------------------------------------------------------------------
# Annotation with dictionary (same as KIRI's annotate_with_dict)
# ---------------------------------------------------------------------------
def annotate_with_dict(
    text: str,
    d,
    headers: list[str],
    note_id: str | None,
    keep_overlaps: bool = False,
) -> pd.DataFrame:
    rows = []
    h_positions, pos_header = get_sections(text, headers)

    if isinstance(d, IndexedDict):
        items_iter = d.iter_items_for_text(text)
    else:
        items_iter = d.items()

    for (section, source_text), cid in items_iter:
        p = get_pattern(source_text)
        if p is None:
            continue
        for match in p.finditer(text):
            i, j = match.start(), match.end()
            if i < 100:
                continue
            if i > 0 and text[i - 1].isalnum():
                continue
            if j < len(text) and text[j].isalnum():
                continue
            if is_in_header(text, i) and not keep_overlaps:
                continue

            h = get_header_by_pos(i, h_positions, pos_header, headers)
            if h is None:
                continue
            hl = h.lower()
            if "medication" in hl or "service" in hl or "date of birth" in hl:
                continue

            if h == section or h in section or section == "any":
                if isinstance(cid, Counter):
                    for k in cid:
                        rows.append([note_id, i, j, k, section, source_text])
                else:
                    rows.append([note_id, i, j, cid, section, source_text])

    ann = pd.DataFrame(
        rows, columns=["note_id", "start", "end", "concept_id", "section", "dict_entry"]
    )
    if keep_overlaps:
        return ann
    return remove_overlaps(ann)


# ---------------------------------------------------------------------------
# Overlap removal (same as KIRI)
# ---------------------------------------------------------------------------
def remove_overlaps(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.sort_values("start").reset_index(drop=True)
    length = (df["end"] - df["start"]).astype(float)
    section_any = [isinstance(s, tuple) or s == "any" for s in df["section"]]
    length[section_any] -= 0.1

    to_remove = set()
    n = len(df)
    for i in range(n):
        if df.index[i] in to_remove:
            continue
        for j in range(i + 1, n):
            if df["start"].iloc[j] >= df["end"].iloc[i]:
                break
            if length.iloc[i] < length.iloc[j]:
                remove_idx = i
            else:
                remove_idx = j
            to_remove.add(df.index[remove_idx])
            if remove_idx == i:
                break

    df2 = df.drop(to_remove)
    for idx in to_remove:
        s, e = df.loc[idx, ["start", "end"]].values
        overlaps = ((df2["start"] <= s) & (df2["end"] > s)) | (
            (df2["start"] <= e) & (df2["end"] > e)
        )
        if overlaps.sum() == 0:
            df2.loc[idx] = df.loc[idx]

    return df2


# ---------------------------------------------------------------------------
# Training: build dictionary from annotations
# ---------------------------------------------------------------------------
def build_dict_from_annotations(
    text: str,
    annotations: pd.DataFrame,
    headers: list[str],
    blacklist: set[str],
) -> dict:
    d: dict[tuple, Counter] = {}
    h_positions, pos_header = get_sections(text, headers)

    for _, row in annotations.iterrows():
        mention = str(row.get("span") or text[int(row["start"]):int(row["end"])]).lower()
        mention = " ".join(mention.split())  # normalize internal whitespace (newlines, tabs, etc.)
        if not mention or len(mention) < 2:
            continue
        if mention in blacklist:
            continue
        h = get_header_by_pos(int(row["start"]), h_positions, pos_header, headers)
        d.setdefault((h, mention), Counter())[int(row["concept_id"])] += 1
        d.setdefault(("any", mention), Counter())[int(row["concept_id"])] += 1

    return d


# ---------------------------------------------------------------------------
# Scoring: compare predictions against gold
# ---------------------------------------------------------------------------
def overlap_score(
    ref_start, ref_end, ann_start, ann_end, ref_concept, ann_concept, mention
):
    if ref_start > ann_start:
        return -1
    elif ref_end < ann_start:
        return 0
    elif ref_concept == ann_concept:
        if (ref_start == ann_start and ref_end == ann_end) or " " in str(mention):
            return 1
        return -1
    else:
        return -1


def compare_ref_pred(ref: pd.DataFrame, ann: pd.DataFrame) -> pd.DataFrame:
    ref = ref.sort_values("start").reset_index(drop=True)
    ann = ann.sort_values("start").reset_index(drop=True)
    i_ref = 0
    n_ref = len(ref)
    ann["score"] = None

    for idx in ann.index:
        while (
            i_ref + 1 < n_ref - 1
            and ref.iloc[i_ref + 1]["start"] <= ann.loc[idx, "start"]
        ):
            i_ref += 1

        score = overlap_score(
            ref.iloc[i_ref]["start"], ref.iloc[i_ref]["end"],
            ann.loc[idx, "start"], ann.loc[idx, "end"],
            ref.iloc[i_ref]["concept_id"], ann.loc[idx, "concept_id"],
            ann.loc[idx, "dict_entry"],
        )
        if score == 0 and i_ref + 1 < n_ref - 1:
            score = overlap_score(
                ref.iloc[i_ref + 1]["start"], ref.iloc[i_ref + 1]["end"],
                ann.loc[idx, "start"], ann.loc[idx, "end"],
                ref.iloc[i_ref + 1]["concept_id"], ann.loc[idx, "concept_id"],
                ann.loc[idx, "dict_entry"],
            )
        if score == 0:
            score = -1
        ann.loc[idx, "score"] = score
    return ann


def score_dict(text, ref, d, d_indexed, headers):
    ann = annotate_with_dict(text, d_indexed, headers, None, keep_overlaps=True)
    if ann.empty:
        return {}, Counter()
    ann["start"] = ann["start"].astype(int)
    ann["end"] = ann["end"].astype(int)
    ann["concept_id"] = ann["concept_id"].astype(int)
    ann = compare_ref_pred(ref.copy(), ann)

    scores = {}
    scores_counter = Counter()
    for idx in ann.index:
        h = ann.loc[idx, "section"]
        source = ann.loc[idx, "dict_entry"]
        k = (h, source)
        if k in d:
            if isinstance(d[k], Counter):
                k = (k, ann.loc[idx, "concept_id"])
            scores.setdefault(k, []).append(ann.loc[idx, "score"])
            scores_counter[(k, ann.loc[idx, "score"])] += 1
    return scores, scores_counter


# ---------------------------------------------------------------------------
# Bad key removal
# ---------------------------------------------------------------------------
def count_correct(lst):
    s = pd.Series(lst)
    return (s == 1).sum(), (s == -1).sum()


def is_naive_key_remove(counts, k, double_thr=False):
    section = k[0] if not isinstance(k[0], tuple) or len(k[0]) != 2 else k[0][0]
    th = CORRECT_FRAC_FOR_ANY if section == "any" else CORRECT_FRAC_FOR_DICT
    if double_thr:
        th *= 2
    correct, incorrect = counts
    if correct == 1:
        th = 1
    return correct < th * incorrect


def remove_bad_keys(d, scores):
    bad_keys = []
    for k in scores:
        if is_naive_key_remove(count_correct(scores[k]), k):
            bad_keys.append(k)

    in_d = len(set(bad_keys) & set(d.keys()))
    print(f"  Bad keys: {len(bad_keys)} ({in_d} in dict)")
    for k in bad_keys:
        d.pop(k, None)
    return bad_keys


def yuvals_key_selection(d, scores_by_mention, annotations):
    cids = annotations["concept_id"].unique()
    bad_keys = []

    for cid in cids:
        keys = [k for k in d if d[k] == cid]
        t_scores = {
            k: count_correct(scores_by_mention[k])
            for k in keys if k in scores_by_mention
        }
        n_annotations = (annotations["concept_id"] == cid).sum()
        if n_annotations == 0:
            continue

        # Rank keys by correct/incorrect ratio, greedily remove low-value ones
        key_to_ratio = pd.Series(
            {k: t_scores[k][0] / (t_scores[k][1] + 0.01) for k in t_scores}
        )
        key_to_ratio.sort_values(ascending=False, inplace=True)

        correct = 0
        incorrect = 0
        for i, k in enumerate(key_to_ratio.index):
            curr_score = correct / (incorrect + n_annotations)
            if curr_score < key_to_ratio[k] or not is_naive_key_remove(
                t_scores[k], k, double_thr=(i > 2)
            ):
                correct += t_scores[k][0]
                incorrect += t_scores[k][1]
            else:
                bad_keys.append(k)

    in_d = len(set(bad_keys) & set(d.keys()))
    print(f"  Bad keys (Yuval's method): {len(bad_keys)} ({in_d} in dict)")
    for k in bad_keys:
        d.pop(k, None)
    return bad_keys


# ---------------------------------------------------------------------------
# Uppercase mention extraction
# ---------------------------------------------------------------------------
def extract_uppercase_mentions(d, annotations):
    if "span" not in annotations.columns:
        return {}

    # Find mentions that are almost always uppercase in original annotations
    # Normalize whitespace to match build_dict normalization
    src = annotations["span"].apply(lambda x: " ".join(str(x).lower().split()) if pd.notna(x) else x)
    orig = annotations["span"].apply(lambda x: " ".join(str(x).split()) if pd.notna(x) else x)
    is_upper = orig.notna() & orig.eq(orig.str.upper())
    frac_upper = is_upper.groupby(src, sort=False).mean()
    uc_mentions = set(frac_upper[frac_upper > 0.99].index.astype(str))

    sec_map = {h.lower() + ":": h + ":" for h in COMMON_HEADERS}
    sec_map["other"] = "other"
    sec_map["any"] = "any"

    uc_d = {}
    to_remove = []
    for k in d:
        section, mention = k
        if str(mention) in uc_mentions:
            cap_section = sec_map.get(section, section)
            uc_d[(cap_section, str(mention).upper())] = d[k]
            to_remove.append(k)
    for k in to_remove:
        d.pop(k, None)

    return uc_d


# ---------------------------------------------------------------------------
# SNOMED synonym addition from super dictionary
# ---------------------------------------------------------------------------
def process_term(t: str) -> str:
    t = t.lower()
    if "(" in t:
        t = t[: t.rindex("(") - 1]
    if "]" in t:
        t = t[t.index("]") + 1:]
    return t.strip()


def load_snomed_synonyms_from_super_dict(
    dict_path: Path,
    min_len: int = SNOMED_MIN_LEN,
    max_len: int = SNOMED_MAX_LEN,
    allowed_concept_ids: Optional[set[int]] = None,
) -> dict:
    d = {}
    skipped_cid = 0
    with dict_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp, delimiter="\t")
        for row in reader:
            cid_str = (row.get("snomed_concept_id") or "").strip()
            term = (row.get("term") or "").strip()
            if not cid_str or not term:
                continue
            try:
                cid = int(cid_str)
            except ValueError:
                continue

            # Only add synonyms for concepts in the flattened terminology
            # so limit_any_to_allowed_sections can restrict them
            if allowed_concept_ids is not None and cid not in allowed_concept_ids:
                skipped_cid += 1
                continue

            # KIRI filters: skip machine-translated and bracketed entries
            if "machine translation" in term:
                continue
            if "]" in term:
                try:
                    if term.index("[") > 5:
                        continue
                except ValueError:
                    pass

            pt = process_term(term)
            n = len(pt.split())
            if n > max_len or n < min_len or len(pt) < 3:
                continue
            if not pt[0].isalnum():
                continue
            if len(pt) > 1:
                d[("any", pt)] = cid
    if skipped_cid:
        print(f"    Skipped {skipped_cid:,} entries (concept_id not in terminology)")
    return d


# ---------------------------------------------------------------------------
# Concept type -> allowed sections mapping
# ---------------------------------------------------------------------------
def load_concept_types(flat_terminology_path: Path) -> pd.Series:
    df = pd.read_csv(flat_terminology_path)
    df = df.drop_duplicates("concept_id", keep="first").set_index("concept_id")["concept_name"]
    replacements = {
        "procedure": "procedure",
        "body structure": "body structure",
        "disorder": "finding",
        "finding": "finding",
        "morphologic abnormality": "body structure",
        "cell structure": "body structure",
        "regime/therapy": "finding",
    }
    cid_to_type = df.apply(lambda x: x.split("(")[-1][:-1] if "(" in x else "other")
    cid_to_type = cid_to_type.replace(replacements)
    return cid_to_type


def get_allowed_sections(texts, annotations, headers, cid_to_type):
    act = {}
    for nid in annotations["note_id"].unique():
        if nid not in texts.index:
            continue
        df = annotations[annotations["note_id"] == nid]
        h_positions, pos_header = get_sections(texts[nid], headers)
        for _, row in df.iterrows():
            cid = int(row["concept_id"])
            if cid not in cid_to_type.index:
                continue
            h = get_header_by_pos(int(row["start"]), h_positions, pos_header, headers)
            ct = cid_to_type[cid]
            act.setdefault(ct, set()).add(h)
    return act


def limit_any_to_allowed_sections(d, allowed_sec, cid_to_type):
    any_keys = [k for k in d if k[0] == "any"]
    for k in any_keys:
        cid = d[k]
        if cid not in cid_to_type.index:
            continue
        ct = cid_to_type[cid]
        if ct not in allowed_sec:
            continue
        d[(tuple(allowed_sec[ct]), k[1])] = cid
        d.pop(k, None)


# ---------------------------------------------------------------------------
# Word replacements and permutations
# ---------------------------------------------------------------------------
def get_word_replacements(d):
    wr = {}
    replacements = {
        ",": "",
        " and ": " with ",
        " with ": " and ",
        " valve ": " ",
        " of ": " of the ",
    }
    for k in d:
        mention = k[1]
        if not isinstance(mention, str):
            continue
        for s1, s2 in replacements.items():
            if s1 in mention:
                wr[(k[0], mention.replace(s1, s2))] = d[k]
    return wr


def get_permutations(d, blacklist):
    permuted = {}
    for k in d:
        words = k[1].split()
        n = len(words)
        new_mentions = []
        if n < 3 or n > 4:
            continue
        if n == 3 and words[1] == "of":
            new_mentions = [f"{words[2]} {words[0]}"]
        elif n == 4:
            if words[1] == "of":
                new_mentions = [f"{words[2]} {words[3]} {words[0]}"]
            elif words[2] == "of":
                new_mentions = [
                    f"{words[3]} {words[0]} {words[1]}",
                    f"{words[0]} {words[3]} {words[1]}",
                ]
        elif all(w not in blacklist for w in words):
            new_mentions = [" ".join(p) for p in iter_permutations(words)]
        for nm in new_mentions:
            nk = (k[0], nm)
            if nk not in d:
                permuted[nk] = d[k]
    return permuted


# ---------------------------------------------------------------------------
# Linguistic variants (abbreviation expansion + fracture permutations)
# ---------------------------------------------------------------------------
_ABBREV_EXPAND = {"pt": "patient", "c/o": "complains of", "l": "left", "r": "right", "fx": "fracture"}
_ABBREV_CONTRACT = {"patient": "pt", "left": "l", "right": "r", "fracture": "fx"}


def _fracture_variants(text: str) -> list[str]:
    variants = []
    lower = text.lower()
    if "fracture" not in lower and "fx" not in lower:
        return variants
    if lower.startswith("fracture of "):
        rest = text[len("fracture of "):].strip()
        if rest:
            variants.extend([f"fx of {rest}", f"{rest} fx"])
    if lower.startswith("fx of "):
        rest = text[len("fx of "):].strip()
        if rest:
            variants.extend([f"fracture of {rest}", f"{rest} fracture"])
    if lower.endswith(" fracture"):
        rest = text[:-len(" fracture")].strip()
        if rest:
            variants.extend([f"{rest} fx", f"fx of {rest}"])
    if lower.endswith(" fx"):
        rest = text[:-len(" fx")].strip()
        if rest:
            variants.extend([f"{rest} fracture", f"fracture of {rest}"])
    return variants


def _abbreviation_variants(text: str, max_variants: int = 16) -> list[str]:
    tokens = str(text).split()
    variants = {""}
    for token in tokens:
        options = [token]
        lower = token.lower()
        if lower in _ABBREV_EXPAND:
            options.append(_ABBREV_EXPAND[lower])
        if lower in _ABBREV_CONTRACT:
            options.append(_ABBREV_CONTRACT[lower])
        next_v = set()
        for v in variants:
            for opt in options:
                next_v.add(f"{v} {opt}".strip())
                if len(next_v) >= max_variants:
                    break
            if len(next_v) >= max_variants:
                break
        variants = next_v
        if len(variants) >= max_variants:
            break
    return list(variants)


def add_linguistic_variants(d, blacklist) -> int:
    added = 0
    trigger = set(_ABBREV_EXPAND) | set(_ABBREV_CONTRACT)
    for (section, mention) in list(d.keys()):
        if section != "any":
            continue
        if mention in blacklist:
            continue
        toks = set(str(mention).lower().split())
        if "fracture" not in toks and "fx" not in toks and not (toks & trigger):
            continue

        all_mentions = set(_abbreviation_variants(mention))
        for v in list(all_mentions):
            all_mentions.update(_fracture_variants(v))

        for nm in all_mentions:
            if not nm or len(nm) < 2 or not nm[0].isalnum():
                continue
            nk = (section, nm)
            if nk not in d:
                d[nk] = d[(section, mention)]
                added += 1
    return added


# ---------------------------------------------------------------------------
# Conditional update (add entries only if not already present or is FSN match)
# ---------------------------------------------------------------------------
def cond_update(d, d2, sno_fsn, blacklist):
    for k, v in d2.items():
        if not isinstance(k, tuple) or len(k) != 2 or not isinstance(k[1], str):
            continue
        if k[1] in blacklist:
            continue
        sno_name = sno_fsn.get(v)
        sno_lc = sno_name.lower() if isinstance(sno_name, str) else None
        if k not in d or (sno_lc is not None and k[1] == sno_lc):
            d[k] = v


# ---------------------------------------------------------------------------
# Case-sensitive hardcoded entries
# ---------------------------------------------------------------------------
CASE_SENSITIVE_DICT = {
    (("other", "Pertinent Results:"), "K"): 312468003,
    ("any", "T"): 105723007,
    (("other", "Pertinent Results:"), "Mg"): 271285000,
    ("Physical Exam:", "RA"): 722742002,
    (("other", "Pertinent Results:"), "Plt"): 61928009,
    (("other", "Pertinent Results:"), "MR"): 48724000,
}


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------
def train(
    train_notes: pd.DataFrame,
    train_annotations: pd.DataFrame,
    super_dict_path: Path,
    flat_terminology_path: Optional[Path] = None,
    skip_replacements: bool = False,
    replacements_only: bool = False,
    snomed_unigram_paths: Optional[list[Path]] = None,
) -> tuple[dict, dict]:
    t0 = time.perf_counter()

    texts = train_notes.set_index("note_id")["text"]
    texts_lc = texts.str.lower()
    headers = [h.lower() for h in COMMON_HEADERS]

    # Normalize annotations
    annotations = train_annotations.copy()
    if "span" in annotations.columns:
        annotations["source orig"] = annotations["span"]
        annotations["source"] = annotations["span"].str.lower()
    else:
        # Extract spans from text
        spans = []
        for _, row in annotations.iterrows():
            t = texts.get(row["note_id"], "")
            spans.append(t[int(row["start"]):int(row["end"])])
        annotations["source orig"] = spans
        annotations["source"] = [s.lower() for s in spans]
        annotations["span"] = annotations["source orig"]

    # Step 1: Build word counter + dynamic blacklist
    print("Building word counter...", flush=True)
    words_counter = Counter()
    for text in texts_lc:
        words_counter.update(text.split())
    blacklist_set = {w for w in words_counter if words_counter[w] > BLACKLIST_THRESH}
    blacklist_set.update(INTERNAL_BLACKLIST)
    blacklist_list = list(blacklist_set)
    print(f"  Dynamic blacklist: {len(blacklist_set)} terms")

    # Step 2: Build training dictionary from annotations
    print("Building training dictionary from annotations...", flush=True)
    d_combined: dict[tuple, Counter] = {}
    for nid in texts_lc.index:
        note_anns = annotations[annotations["note_id"] == nid]
        if note_anns.empty:
            continue
        t = build_dict_from_annotations(texts_lc[nid], note_anns, headers, blacklist_set)
        for k in t:
            d_combined.setdefault(k, Counter()).update(t[k])

    d: dict[tuple, int] = {}
    for k in d_combined:
        mc = d_combined[k].most_common(1)
        if mc:
            d[k] = mc[0][0]

    print(f"  Initial dictionary: {len(d)} entries")

    # Step 3: Score entries against training data
    print("Scoring dictionary entries...", flush=True)
    t1 = time.perf_counter()
    d_indexed = IndexedDict(d, prefilter="bigram")
    refs = {}
    for nid, df in annotations[annotations["note_id"].isin(texts_lc.index)].groupby("note_id", sort=False):
        refs[str(nid)] = df[["start", "end", "concept_id", "source"]].copy()

    scores_by_note = {}
    scores_by_mention = {}
    for nid in texts_lc.index:
        ref = refs.get(str(nid))
        if ref is None:
            continue
        t_scores, _ = score_dict(texts_lc[nid], ref, d, d_indexed, headers)
        for k in t_scores:
            scores_by_mention.setdefault(k, []).extend(t_scores[k])
            for s in [1, -1]:
                if s in t_scores[k]:
                    scores_by_note.setdefault(k, []).append(s)
    print(f"  Scoring done in {time.perf_counter() - t1:.1f}s")

    # Step 4: Remove bad keys
    print("Removing bad keys...", flush=True)
    remove_bad_keys(d, scores_by_note)
    print(f"  Dictionary after filtering: {len(d)} entries")

    # Step 5: Extract uppercase mentions
    print("Extracting uppercase mentions...", flush=True)
    uc_d = extract_uppercase_mentions(d, annotations)
    print(f"  Uppercase dict: {len(uc_d)} entries")

    # Step 6: Add SNOMED synonyms from super dictionary
    # Only add synonyms for concept_ids in the flattened terminology so
    # limit_any_to_allowed_sections can restrict them.
    allowed_cids = None
    if flat_terminology_path and flat_terminology_path.exists():
        ft = pd.read_csv(flat_terminology_path)
        allowed_cids = set(ft["concept_id"].astype(int).unique())
        print(f"  Restricting synonyms to {len(allowed_cids):,} known concept IDs")

    print(f"Adding SNOMED synonyms from {super_dict_path}...", flush=True)
    sno_syns = load_snomed_synonyms_from_super_dict(
        super_dict_path, allowed_concept_ids=allowed_cids,
    )
    # Build FSN lookup for cond_update from flattened_terminology (matches KIRI).
    # KIRI uses ALL concept names from flattened_terminology.csv, not just FSN
    # entries from the super dict. This affects which entries cond_update
    # overwrites (it overwrites when the mention matches the FSN name).
    if flat_terminology_path and flat_terminology_path.exists():
        _ft = pd.read_csv(flat_terminology_path)
        _ft = _ft.drop_duplicates("concept_id", keep="first").set_index("concept_id")["concept_name"]
        sno_fsn = _ft.apply(lambda t: process_term(t))
        print(f"  FSN lookup: {len(sno_fsn):,} concepts from flattened_terminology")
    else:
        # Fallback: build from super dict FSN entries only
        sno_fsn = {}
        with super_dict_path.open("r", encoding="utf-8", newline="") as fp:
            reader = csv.DictReader(fp, delimiter="\t")
            for row in reader:
                cid_str = (row.get("snomed_concept_id") or "").strip()
                term = (row.get("term") or "").strip()
                source = (row.get("source_detail") or "").strip()
                if cid_str and term and "FSN" in source.upper():
                    try:
                        sno_fsn[int(cid_str)] = process_term(term)
                    except ValueError:
                        pass
        sno_fsn = pd.Series(sno_fsn)
        print(f"  FSN lookup: {len(sno_fsn):,} concepts from super dict (fallback)")

    cond_update(d, sno_syns, sno_fsn, blacklist_set)
    print(f"  After SNOMED synonyms: {len(d)} entries")

    # Step 6b: Add SNOMED unigram dictionaries if provided
    if snomed_unigram_paths:
        for upath in snomed_unigram_paths:
            if upath.exists():
                with upath.open("rb") as fp:
                    d_uni = pickle.load(fp)
                cond_update(d, d_uni, sno_fsn, blacklist_set)
                print(f"  After {upath.name}: {len(d)} entries")
            else:
                print(f"  Warning: unigram dict not found: {upath}")

    # Step 7: Linguistic variants (optional, off by default to match KIRI eval)
    if os.environ.get("KIRI_LINGUISTIC_RULES", "0") == "1":
        print("Adding linguistic variants...", flush=True)
        added = add_linguistic_variants(d, blacklist_set)
        print(f"  Added {added} linguistic variants, total: {len(d)}")
    else:
        print("  Skipping linguistic variants (set KIRI_LINGUISTIC_RULES=1 to enable)")

    # Step 8: Word replacements + permutations
    if skip_replacements:
        print("  Skipping word replacements and permutations (--skip-replacements)")
    elif replacements_only:
        print("Adding word replacements (no permutations)...", flush=True)
        wr = get_word_replacements(d)
        cond_update(d, wr, sno_fsn, blacklist_set)
        print(f"  After replacements: {len(d)} entries")
    else:
        print("Adding word replacements and permutations...", flush=True)
        wr = get_word_replacements(d)
        cond_update(d, wr, sno_fsn, blacklist_set)
        permuted = get_permutations(d, blacklist_set)
        cond_update(d, permuted, sno_fsn, blacklist_set)
        print(f"  After replacements + permutations: {len(d)} entries")

    # Step 9: Limit "any" keys to allowed sections
    if flat_terminology_path and flat_terminology_path.exists():
        print("Limiting 'any' keys to allowed sections...", flush=True)
        cid_to_type = load_concept_types(flat_terminology_path)
        allowed_sec = get_allowed_sections(texts_lc, annotations, headers, cid_to_type)
        limit_any_to_allowed_sections(d, allowed_sec, cid_to_type)
        print(f"  After section limiting: {len(d)} entries")
    else:
        print("  Skipping section limiting (no flattened_terminology.csv)")

    print(f"Training complete in {time.perf_counter() - t0:.1f}s")
    print(f"  Final dictionary: {len(d)} entries")
    print(f"  Uppercase dictionary: {len(uc_d)} entries")

    return d, uc_d


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train a filtered section-aware dictionary from training annotations."
    )
    parser.add_argument(
        "--train-notes",
        required=True,
        help="Path to training notes CSV (note_id, text)",
    )
    parser.add_argument(
        "--train-annotations",
        required=True,
        help="Path to training annotations CSV (note_id, start, end, concept_id, span)",
    )
    parser.add_argument(
        "--super-dict",
        default="data/interim/super_dictionary_full.tsv",
        help="Path to super_dictionary_full.tsv",
    )
    parser.add_argument(
        "--flat-terminology",
        default=None,
        help="Path to flattened_terminology.csv (for section limiting)",
    )
    parser.add_argument(
        "--abbr-dict",
        default=None,
        help="Path to abbreviation dictionary pickle (abbr_dict.pkl)",
    )
    parser.add_argument(
        "--snomed-unigrams",
        nargs="*",
        default=None,
        help="Paths to SNOMED unigram dictionary pickles (snomed_unigrams_*.pkl)",
    )
    parser.add_argument(
        "--skip-replacements",
        action="store_true",
        help="Skip word replacements and permutations (reduces FPs)",
    )
    parser.add_argument(
        "--replacements-only",
        action="store_true",
        help="Add word replacements but skip permutations",
    )
    parser.add_argument(
        "--output",
        default="data/interim/trained_dict.pkl",
        help="Output path for trained dictionary pickle",
    )
    args = parser.parse_args(argv)

    notes = pd.read_csv(args.train_notes)
    annotations = pd.read_csv(args.train_annotations)

    flat_path = Path(args.flat_terminology) if args.flat_terminology else None
    super_path = Path(args.super_dict)

    if not super_path.exists():
        print(f"Missing: {super_path}", file=sys.stderr)
        return 2

    d, uc_d = train(
        notes, annotations, super_path, flat_path,
        skip_replacements=args.skip_replacements,
        replacements_only=args.replacements_only,
        snomed_unigram_paths=[Path(p) for p in args.snomed_unigrams] if args.snomed_unigrams else None,
    )

    # Merge abbreviation dictionary into uc_d if provided
    if args.abbr_dict:
        abbr_path = Path(args.abbr_dict)
        if abbr_path.exists():
            with abbr_path.open("rb") as fp:
                abbr = pickle.load(fp)
            n_before = len(uc_d)
            abbr.update(uc_d)  # uc_d entries take priority
            uc_d = abbr
            print(f"Merged abbreviation dict: {n_before} -> {len(uc_d)} UC entries")
        else:
            print(f"Warning: abbr_dict not found: {abbr_path}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as fp:
        pickle.dump({"d": d, "uc_d": uc_d}, fp, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved trained dictionary -> {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
