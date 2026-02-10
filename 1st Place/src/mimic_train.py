# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 09:19:31 2024

@author: Yonatan
"""

import os
import pickle
from collections import Counter
from itertools import permutations
from pathlib import Path
from time import perf_counter

import pandas as pd
from mimic_common import (
    IndexedDict,
    annotate_with_dict,
    common_headers,
    get_header_by_pos,
    get_sections,
    get_pattern,
    internal_blacklist,
)
from tqdm import tqdm

data_directory = Path(__file__).parent.parent / "data"
(debug_directory := Path(__file__).parent.parent / "debug").mkdir(exist_ok=True)
correct_frac_for_dict = 0.2
correct_frac_for_any = 0.3
yuvals_method_ratio = 1
snomed_min_len = int(os.environ.get("KIRI_SNOMED_MIN_LEN", "2"))
snomed_max_len = int(os.environ.get("KIRI_SNOMED_MAX_LEN", "5"))
blacklist_thresh = 2000
train_size = 150
test_size = None
words_counter = Counter()
_CACHED_SNO_UNIGRAMS_3K = None
_CACHED_SNO_UNIGRAMS_20K = None
_CACHED_UC_MENTIONS = None
_CACHED_UC_MENTIONS_SIG = None


def get_blacklist():
    if len(words_counter) == 0:
        print("word counter not initiated")
        return None
    bl = [v for v in words_counter if words_counter[v] > blacklist_thresh]
    bl.extend(internal_blacklist)
    return bl


def build_dict(text, text_annotations, headers, blacklist):
    d = {}
    length = text_annotations["end"] - text_annotations["start"]
    h_positions, pos_header = get_sections(text, headers)
    section_blacklist = {}
    for bl in blacklist:
        if type(bl) == tuple:
            section_blacklist.setdefault(bl[0], set()).add(bl[1])

    rows = (length > 1) & text_annotations["source"].notna() & ~text_annotations["source"].isin(blacklist)
    for i in text_annotations.index[rows]:
        mention = text_annotations["source"][i]
        h = get_header_by_pos(text_annotations["start"][i], h_positions, pos_header, headers)
        if h in section_blacklist and mention in section_blacklist[h]:
            continue
        d.setdefault((h, mention), Counter())[text_annotations["concept_id"][i]] += 1
        d.setdefault(("any", mention), Counter())[text_annotations["concept_id"][i]] += 1

    return d


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def score_dict(text, ref, d, d_for_annot, headers):
    scores = {}
    # `d` is used for lookup/membership. For faster matching, optionally wrap
    # with `IndexedDict` (semantics-preserving prefilter) during annotation.
    ann = annotate_with_dict(text, d_for_annot, headers, None, keep_overlaps=True)
    ann["start"] = ann["start"].astype(int)
    ann["end"] = ann["end"].astype(int)
    ann["concept_id"] = ann["concept_id"].astype(int)
    compare_ref_pred(ref.copy(), ann)
    scores_counter = Counter()
    for i in ann.index:
        h = ann["section"][i]
        source = ann["dict_entry"][i]
        k = (h, source)
        if k in d:
            if type(d[k]) == Counter:
                k = (k, ann["concept_id"][i][-1])
            scores.setdefault(k, []).append(ann["score"][i])
            scores_counter[(k, ann["score"][i])] += 1
        else:
            print("key not in dict:", k)

    return scores, scores_counter


_SCORE_TEXTS = None
_SCORE_REFS = None
_SCORE_D = None
_SCORE_D_FOR_ANNOT = None
_SCORE_HEADERS = None


def _score_one(note_id: str):
    return (
        note_id,
        score_dict(
            _SCORE_TEXTS[note_id],
            _SCORE_REFS[note_id],
            _SCORE_D,
            _SCORE_D_FOR_ANNOT,
            _SCORE_HEADERS,
        ),
    )


def compare_ref_pred(ref, ann):
    ref.sort_values("start", inplace=True)
    ann.sort_values("start", inplace=True)
    i_ref = 0
    n_ref = len(ref)
    ann["score"] = None
    for i in ann.index:
        while i_ref + 1 < n_ref - 1 and ref.iloc[i_ref + 1]["start"] <= ann.loc[i, "start"]:
            i_ref = i_ref + 1

        score = overlap_score(
            ref.iloc[i_ref]["start"],
            ref.iloc[i_ref]["end"],
            ann.loc[i, "start"],
            ann.loc[i, "end"],
            ref.iloc[i_ref]["concept_id"],
            ann.loc[i, "concept_id"],
            ann.loc[i, "dict_entry"],
        )

        if score == 0 and i_ref + 1 < n_ref - 1:
            score = overlap_score(
                ref.iloc[i_ref + 1]["start"],
                ref.iloc[i_ref + 1]["end"],
                ann.loc[i, "start"],
                ann.loc[i, "end"],
                ref.iloc[i_ref + 1]["concept_id"],
                ann.loc[i, "concept_id"],
                ann.loc[i, "dict_entry"],
            )
        if score == 0:
            score = -1
        ann.loc[i, "score"] = score


def overlap_score(ref_start, ref_end, ann_start, ann_end, ref_concept, ann_concept, mention):
    if ref_start > ann_start:
        return -1
    elif ref_end < ann_start:
        return 0  # it might overlap the next one
    elif ref_concept == ann_concept:
        if (ref_start == ann_start and ref_end == ann_end) or " " in mention:
            return 1
        return -1
    else:
        return -1


def add_snomed_syn(d, c_id, c_name, min_len, max_len):
    n = len(c_name.split())
    if len(c_name) < 3:
        return
    if "machine translation" in c_name:
        return
    if "]" in c_name and c_name.index("[") > 5:
        return
    pt = process_term(c_name)
    n = len(pt.split())
    if n > max_len or n < min_len:
        return
    if not pt[0].isalnum():
        return
    if len(pt) > 1:
        d[("any", pt)] = c_id


def get_snomed_synonyms(min_len=snomed_min_len, max_len=snomed_max_len, fsn_only=False):
    synonyms_path = os.environ.get(
        "KIRI_SYNONYMS_PATH",
        str(data_directory / "interim" / "flattened_terminology_syn_snomed+omop_v5.csv"),
    )
    snomed_syns = pd.read_csv(synonyms_path).drop_duplicates("concept_name", keep="first")

    flat_term_path = os.environ.get(
        "KIRI_FLAT_TERMINOLOGY_PATH",
        str(data_directory / "interim" / "flattened_terminology.csv"),
    )
    sno_fsn = (
        pd.read_csv(flat_term_path)
        .drop_duplicates("concept_id", keep="first")
        .set_index("concept_id")["concept_name"]
    )
    replacements = {
        "procedure": "procedure",
        "body structure": "body structure",
        "disorder": "finding",
        "finding": "finding",
        "morphologic abnormality": "body structure",
        "cell structure": "body structure",
        "regime/therapy": "finding",
    }
    cid_to_type = sno_fsn.apply(lambda x: x.split("(")[-1][:-1]).replace(replacements)

    d = {}
    if not fsn_only:
        for c_id, c_name in snomed_syns[["concept_id", "concept_name"]].values:
            add_snomed_syn(d, c_id, c_name, min_len, max_len)
    for c_id in sno_fsn.index:
        add_snomed_syn(d, c_id, sno_fsn[c_id], min_len, max_len)

    sno_fsn = sno_fsn.apply(lambda t: process_term(t))

    return d, sno_fsn, cid_to_type


def process_term(t):
    t = t.lower()
    if "(" in t:
        t = t[: t.rindex("(") - 1]
    if "]" in t:
        t = t[t.index("]") + 1 :]

    return t.strip()


def get_permutations(d, blacklist):
    permuted = {}
    for k in d:
        words = k[1].split()
        new_mentions = []
        n = len(words)
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
        elif (n == 3 or n == 4) and all([w not in blacklist for w in words]):
            new_mentions = [" ".join(p) for p in permutations(words)]
        for new_mention in new_mentions:
            new_key = (k[0], new_mention)
            if new_key not in d:
                permuted[new_key] = d[k]
    return permuted


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


_LINGUISTIC_ABBREV_EXPANSIONS = {
    "pt": "patient",
    "c/o": "complains of",
    "l": "left",
    "r": "right",
    "fx": "fracture",
}
_LINGUISTIC_ABBREV_CONTRACTIONS = {
    "patient": "pt",
    "left": "l",
    "right": "r",
    "fracture": "fx",
}


def _fracture_phrase_variants(text: str) -> list[str]:
    variants: list[str] = []
    lower = text.lower()
    if "fracture" not in lower and "fx" not in lower:
        return variants

    if lower.startswith("fracture of "):
        rest = text[len("fracture of ") :].strip()
        if rest:
            variants.append(f"fx of {rest}")
            variants.append(f"{rest} fx")
    if lower.startswith("fx of "):
        rest = text[len("fx of ") :].strip()
        if rest:
            variants.append(f"fracture of {rest}")
            variants.append(f"{rest} fracture")
    if lower.endswith(" fracture"):
        rest = text[: -len(" fracture")].strip()
        if rest:
            variants.append(f"{rest} fx")
            variants.append(f"fx of {rest}")
    if lower.endswith(" fx"):
        rest = text[: -len(" fx")].strip()
        if rest:
            variants.append(f"{rest} fracture")
            variants.append(f"fracture of {rest}")
    return variants


def _abbreviation_variants(text: str, max_variants: int = 16) -> list[str]:
    tokens = str(text).split()
    variants = {""}
    for token in tokens:
        options = [token]
        lower = token.lower()
        if lower in _LINGUISTIC_ABBREV_EXPANSIONS:
            options.append(_LINGUISTIC_ABBREV_EXPANSIONS[lower])
        if lower in _LINGUISTIC_ABBREV_CONTRACTIONS:
            options.append(_LINGUISTIC_ABBREV_CONTRACTIONS[lower])

        next_variants = set()
        for v in variants:
            for opt in options:
                combined = f"{v} {opt}".strip()
                next_variants.add(combined)
                if len(next_variants) >= max_variants:
                    break
            if len(next_variants) >= max_variants:
                break
        variants = next_variants
        if len(variants) >= max_variants:
            break
    return list(variants)


def add_linguistic_variants(d, blacklist, max_variants_per_key: int = 16) -> int:
    """
    Augment dictionary keys with lightweight clinical NLP style variants.

    Enabled via env var: `KIRI_LINGUISTIC_RULES=1`.
    """
    added = 0
    trigger_tokens = set(_LINGUISTIC_ABBREV_EXPANSIONS) | set(_LINGUISTIC_ABBREV_CONTRACTIONS)
    for (section, mention) in list(d.keys()):
        if section != "any":
            continue
        if mention in blacklist:
            continue

        toks = set(str(mention).lower().split())
        if "fracture" not in toks and "fx" not in toks and not (toks & trigger_tokens):
            continue

        base_variants = _abbreviation_variants(mention, max_variants=max_variants_per_key)
        all_mentions: set[str] = set(base_variants)
        for v in list(all_mentions):
            for fv in _fracture_phrase_variants(v):
                all_mentions.add(fv)

        for new_mention in all_mentions:
            if not new_mention or len(new_mention) < 2:
                continue
            if not str(new_mention)[0].isalnum():
                continue
            new_key = (section, new_mention)
            if new_key in d:
                continue
            d[new_key] = d[(section, mention)]
            added += 1
    return added


def remove_bad_keys(d, scores, scores_include_cid=False):
    bad_keys = []
    for k in scores:
        if scores_include_cid and (k[0] not in d or d[k[0]] != k[1]):
            continue

        if is_naive_key_remove(count_correct(scores[k]), k, scores_include_cid):
            if scores_include_cid:
                bad_keys.append(k[0])
            else:
                bad_keys.append(k)

    print(
        "number of bad keys:",
        len(bad_keys),
        "in d:",
        len(set(bad_keys).intersection(d.keys())),
    )
    for k in bad_keys:
        d.pop(k, None)
    return bad_keys


def yuvals_key_selection(d, scores_by_mention, scores_by_note, annotations):
    cids = annotations["concept_id"].unique()
    bad_keys = []
    print("removing bad keys")
    for cid in tqdm(cids):
        keys = [k for k in d if d[k] == cid]
        t_scores = {k: count_correct(scores_by_mention[k]) for k in keys if k in scores_by_mention}
        n_annotations = (annotations["concept_id"] == cid).sum()
        bad_keys.extend(get_bad_keys_for_concept(t_scores, n_annotations))

    print(
        "number of bad keys (yuval method):",
        len(bad_keys),
        "in d:",
        len(set(bad_keys).intersection(d.keys())),
    )
    for k in bad_keys:
        d.pop(k, None)
    return bad_keys


def count_correct(l):
    s = pd.Series(l)
    return (s == 1).sum(), (s == -1).sum()


def get_bad_keys_for_concept(scores, n):
    assert n > 0
    bad_keys = []
    key_to_ratio = pd.Series(
        [scores[k][0] / (scores[k][1] + 0.01) for k in scores], index=scores.keys()
    )
    correct = 0
    incorrect = 0
    key_to_ratio.sort_values(ascending=False, inplace=True)
    for i, k in enumerate(key_to_ratio.index):
        curr_score = correct / (incorrect + n)

        if curr_score < key_to_ratio[k] or not is_naive_key_remove(
            scores[k], k, double_thr=(i > 2)
        ):
            correct += scores[k][0]
            incorrect += scores[k][1]
        else:
            bad_keys.append(k)
    return bad_keys


def is_naive_key_remove(counts, k, scores_include_cid=False, double_thr=False):
    section = k[0] if not scores_include_cid else k[0][0]
    th = correct_frac_for_any if section == "any" else correct_frac_for_dict
    if double_thr:
        th = th * 2

    correct = counts[0]
    if correct == 1:
        th = 1
    incorrect = counts[1]
    return correct < th * incorrect


def mock_train(texts, annotations, headers, run_name):
    blacklist = get_blacklist()
    d, d_all = {}, {}
    print("extracting annotations")

    ids = texts.index

    d_combined = {}
    for i in tqdm(ids):
        t = build_dict(texts[i], annotations.query(f'note_id == "{i}"'), headers, blacklist)

        for k in t:
            d_combined.setdefault(k, Counter()).update(t[k])
        d_all[i] = t

    for k in d_combined:
        mc = d_combined[k].most_common(1)
        if len(mc) == 1:
            d[k] = mc[0][0]

    scores_by_note = {}
    scores_by_mention = {}
    scores_counter = {}
    print("scoring (training dict; used to remove bad keys)")
    t0 = perf_counter()

    refs = {
        str(note_id): df[["start", "end", "concept_id", "source"]].copy()
        for note_id, df in annotations[annotations["note_id"].isin(ids)].groupby("note_id", sort=False)
    }

    use_index = _env_bool("KIRI_TRAIN_INDEX", True)
    if use_index:
        prefilter = "unigram" if _env_bool("KIRI_STOPWORD_TRANSPARENT", False) else "bigram"
        d_for_annot = IndexedDict(d, prefilter=prefilter)
    else:
        d_for_annot = d

    want_parallel = _env_bool("KIRI_TRAIN_PARALLEL", _env_bool("KIRI_PARALLEL", False))
    workers = 1
    if want_parallel and os.name == "posix":
        raw = str(os.environ.get("KIRI_TRAIN_WORKERS", "")).strip()
        if not raw:
            # Back-compat / convenience: allow the more general env var to drive training too.
            raw = str(os.environ.get("KIRI_WORKERS", "")).strip()
        try:
            workers = int(raw) if raw else min(8, (os.cpu_count() or 1))
        except Exception:
            workers = min(8, (os.cpu_count() or 1))
        if workers < 1:
            workers = 1

    if want_parallel and workers > 1 and os.name == "posix":
        chunksize_raw = str(os.environ.get("KIRI_TRAIN_CHUNKSIZE", "")).strip()
        try:
            chunksize = int(chunksize_raw) if chunksize_raw else 1
        except Exception:
            chunksize = 1
        if chunksize < 1:
            chunksize = 1

        precompile = _env_bool("KIRI_TRAIN_PRECOMPILE", True)
        log_cfg = _env_bool("KIRI_TRAIN_LOG", True)
        if log_cfg:
            print(
                f"[train-score] notes={len(ids)} use_index={use_index} parallel={want_parallel} "
                f"workers={workers} chunksize={chunksize} precompile={precompile}",
                flush=True,
            )

        # Precompile regex patterns once in the parent so forked workers can
        # share them via copy-on-write (avoids N workers recompiling the same patterns).
        if precompile:
            for mention in set(k[1] for k in d.keys()):
                get_pattern(mention)

        global _SCORE_TEXTS, _SCORE_REFS, _SCORE_D, _SCORE_D_FOR_ANNOT, _SCORE_HEADERS
        _SCORE_TEXTS = texts
        _SCORE_REFS = refs
        _SCORE_D = d
        _SCORE_D_FOR_ANNOT = d_for_annot
        _SCORE_HEADERS = headers

        import multiprocessing as mp

        ctx = mp.get_context("fork")
        with ctx.Pool(processes=workers) as pool:
            for note_id, (t, ctr) in tqdm(
                pool.imap_unordered(_score_one, list(ids), chunksize=chunksize),
                total=len(ids),
            ):
                scores_counter[note_id] = ctr
                for k in t:
                    scores_by_mention.setdefault(k, []).extend(t[k])
                    for s in [1, -1]:
                        if s in t[k]:
                            scores_by_note.setdefault(k, []).append(s)
    else:
        log_cfg = _env_bool("KIRI_TRAIN_LOG", True)
        if log_cfg:
            print(
                f"[train-score] notes={len(ids)} use_index={use_index} parallel={want_parallel} workers={workers}",
                flush=True,
            )
        for i in tqdm(ids):
            t, scores_counter[i] = score_dict(texts[i], refs[str(i)], d, d_for_annot, headers)
            for k in t:
                scores_by_mention.setdefault(k, []).extend(t[k])
                for s in [1, -1]:
                    if s in t[k]:
                        scores_by_note.setdefault(k, []).append(s)
    print(f"scoring done in {perf_counter() - t0:0.1f}s")

    d_full = d.copy()
    t1 = perf_counter()
    skip_bad_key = _env_bool("KIRI_SKIP_BAD_KEY_REMOVAL", False)
    if skip_bad_key:
        bad_keys = []
        print("SKIPPED remove_bad_keys (KIRI_SKIP_BAD_KEY_REMOVAL=1)")
    else:
        bad_keys = remove_bad_keys(d, scores_by_note)
        print(f"remove_bad_keys done in {perf_counter() - t1:0.1f}s")

    # This debug pickle can get very large (especially `d_all`) and can dominate
    # wall time on slower disks/volumes while using little CPU.
    save_debug = _env_bool("KIRI_TRAIN_SAVE_DEBUG", True)
    save_d_all = _env_bool("KIRI_TRAIN_SAVE_D_ALL", True)
    if save_debug:
        t2 = perf_counter()
        payload = {
            "d_trained": d,
            "d_full": d_full,
            "bad_keys": bad_keys,
            "d_combined": d_combined,
            "scores_by_note": scores_by_note,
            "scores_by_mention": scores_by_mention,
            "scores_counter": scores_counter,
        }
        if save_d_all:
            payload["d_all"] = d_all
        with (debug_directory / f"{run_name}.pkl").open("wb") as fp:
            pickle.dump(payload, fp, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"debug pickle dump done in {perf_counter() - t2:0.1f}s")

    return d, scores_by_note, scores_by_mention


def get_cid_type_sections_pairs(texts, annotations, headers, cid_to_type):
    pairs = set()
    for nid in annotations["note_id"].unique():
        if nid not in texts.index:
            continue
        df = annotations.query(f'note_id == "{nid}"')
        h_positions, pos_header = get_sections(texts[nid], headers)
        for i, cid in df[["start", "concept_id"]].values:
            if cid not in cid_to_type.index:
                continue
            h = get_header_by_pos(i, h_positions, pos_header, headers)
            pairs.add((h, cid_to_type[cid]))
    return pairs


def get_allowed_sections(texts, annotations, headers, cid_to_type):
    act = {}
    for section, ct in get_cid_type_sections_pairs(texts, annotations, headers, cid_to_type):
        act.setdefault(ct, set()).add(section)
    return act


def limit_any_to_allowed_sections(d, allowed_sec, cid_to_type):
    any_keys = [k for k in d if k[0] == "any"]
    for k in any_keys:
        cid = d[k]
        if cid not in cid_to_type.index:
            print(f"CID {cid} not in flat snomed, skipping")
            continue
        ct = cid_to_type[cid]
        d[(tuple(allowed_sec[ct]), k[1])] = cid
        d.pop(k, None)


def cond_update(d, d2, sno_fsn, blacklist):
    for k, v in d2.items():
        if not isinstance(k, tuple) or len(k) != 2 or not isinstance(k[1], str):
            continue
        if k[1] in blacklist:
            continue
        sno_name = sno_fsn.loc[v] if v in sno_fsn.index else None
        sno_name_lc = sno_name.lower() if isinstance(sno_name, str) else None
        if k not in d or (sno_name_lc is not None and k[1] == sno_name_lc):
            d[k] = v


def _uppercase_mentions_set(annotations: pd.DataFrame, thr: float = 0.99) -> set[str]:
    """
    Find mentions whose original surface forms are "almost always" upper-case.

    This replaces an O(|dict| * |annotations|) filter loop with a single vectorized
    groupby over annotations.
    """
    global _CACHED_UC_MENTIONS, _CACHED_UC_MENTIONS_SIG
    # Cache by dataframe identity; this function is often called twice back-to-back
    # in compare runs (default/super) with the same annotations object.
    sig = (id(annotations), len(annotations))
    if _CACHED_UC_MENTIONS is not None and _CACHED_UC_MENTIONS_SIG == sig:
        return _CACHED_UC_MENTIONS

    if annotations.empty or "source" not in annotations or "source orig" not in annotations:
        _CACHED_UC_MENTIONS = set()
        _CACHED_UC_MENTIONS_SIG = sig
        return _CACHED_UC_MENTIONS

    src = annotations["source"]
    orig = annotations["source orig"]
    # Match the prior semantics: NaNs should not count as "upper".
    is_upper = orig.notna() & orig.eq(orig.str.upper())
    frac_upper = is_upper.groupby(src, sort=False).mean()
    _CACHED_UC_MENTIONS = set(frac_upper[frac_upper > thr].index.astype(str))
    _CACHED_UC_MENTIONS_SIG = sig
    return _CACHED_UC_MENTIONS


def extract_uppercase_mentions(d, annotations):
    uc_mentions = _uppercase_mentions_set(annotations, thr=0.99)
    uc_d = {}
    to_remove = []

    # Avoid re-scanning `common_headers` for every key.
    sec_map = {h.lower() + ":": h + ":" for h in common_headers}
    sec_map["other"] = "other"
    sec_map["any"] = "any"

    for k in d:
        section, mention = k
        if str(mention) in uc_mentions:
            uc_d[(capitalize_section(section, sec_map), str(mention).upper())] = d[k]
            to_remove.append(k)
    for k in to_remove:
        d.pop(k, None)
    return uc_d


def capitalize_section(section, sec_map=None):
    if sec_map is not None:
        out = sec_map.get(section)
        if out is not None:
            return out
    if section in ["other", "any"]:
        return section
    for h in common_headers:
        if section == h.lower() + ":":
            return h + ":"
    print("section not found", section)
    return "any"


def add_external_dicts(d, sno_syns, sno_fsn, blacklist):
    t0 = perf_counter()
    print("initial dict size", len(d))

    t1 = perf_counter()
    cond_update(d, sno_syns, sno_fsn, blacklist)
    print("after adding snomed", len(d))
    print(f"[train] add_snomed_syns in {perf_counter() - t1:0.1f}s", flush=True)

    if _env_bool("KIRI_LINGUISTIC_RULES", False):
        t_lr = perf_counter()
        added = add_linguistic_variants(d, blacklist, max_variants_per_key=16)
        print(f"after adding linguistic variants (+{added:,})", len(d))
        print(f"[train] linguistic_variants in {perf_counter() - t_lr:0.1f}s", flush=True)

    t2 = perf_counter()
    global _CACHED_SNO_UNIGRAMS_3K
    if _CACHED_SNO_UNIGRAMS_3K is None:
        with open(
            data_directory / "interim" / "snomed_unigrams_annotation_dict_3k_v4_new.pkl", "rb"
        ) as fp:
            _CACHED_SNO_UNIGRAMS_3K = pickle.load(fp)
    d_unigrams = _CACHED_SNO_UNIGRAMS_3K
    cond_update(d, d_unigrams, sno_fsn, blacklist)
    print("after adding snomed unigrams", len(d))
    print(f"[train] add_unigrams_3k in {perf_counter() - t2:0.1f}s", flush=True)

    t3 = perf_counter()
    global _CACHED_SNO_UNIGRAMS_20K
    if _CACHED_SNO_UNIGRAMS_20K is None:
        with open(
            data_directory / "interim" / "snomed_unigrams_annotation_dict_20k_v4_fsn.pkl", "rb"
        ) as fp:
            _CACHED_SNO_UNIGRAMS_20K = pickle.load(fp)
    d_unigrams = _CACHED_SNO_UNIGRAMS_20K
    cond_update(d, d_unigrams, sno_fsn, blacklist)
    print("after adding FSN snomed unigrams", len(d))
    print(f"[train] add_unigrams_20k in {perf_counter() - t3:0.1f}s", flush=True)

    t4 = perf_counter()
    wr = get_word_replacements(d)
    cond_update(d, wr, sno_fsn, blacklist)
    print("after doing word replacements", len(d))
    print(f"[train] word_replacements in {perf_counter() - t4:0.1f}s", flush=True)

    t5 = perf_counter()
    permuted = get_permutations(d, blacklist)
    cond_update(d, permuted, sno_fsn, blacklist)
    print("after adding permutations", len(d))
    print(f"[train] permutations in {perf_counter() - t5:0.1f}s", flush=True)
    print(f"[train] add_external_dicts total {perf_counter() - t0:0.1f}s", flush=True)


def train(texts, annotations, headers=common_headers, run_name="debug"):
    t0 = perf_counter()
    texts_lc = texts.str.lower()
    words = "\n".join(texts_lc).split()
    if len(words_counter) == 0:
        words_counter.update(Counter(words))

    headers = [h.lower() for h in headers]
    if "source orig" not in annotations:
        annotations["source orig"] = annotations["source"]
        annotations["source"] = annotations["source"].str.lower()

    sno_syns, sno_fsn, cid_to_type = get_snomed_synonyms()
    allowed_sec = get_allowed_sections(texts_lc, annotations, headers, cid_to_type)

    t1 = perf_counter()
    d, scores_by_note, scores_by_mention = mock_train(texts_lc, annotations, headers, run_name)
    print(f"[train] mock_train done in {perf_counter() - t1:0.1f}s", flush=True)

    t2 = perf_counter()
    uc_d = extract_uppercase_mentions(d, annotations)
    print("number of entries moved to uc dict", len(uc_d))
    print(f"[train] extract_uppercase_mentions done in {perf_counter() - t2:0.1f}s", flush=True)

    t3 = perf_counter()
    add_external_dicts(d, sno_syns, sno_fsn, get_blacklist())
    print(f"[train] add_external_dicts done in {perf_counter() - t3:0.1f}s", flush=True)

    t4 = perf_counter()
    limit_any_to_allowed_sections(d, allowed_sec, cid_to_type)
    print(
        f"[train] limit_any_to_allowed_sections done in {perf_counter() - t4:0.1f}s",
        flush=True,
    )

    save_full = _env_bool("KIRI_TRAIN_SAVE_FULL", True)
    if save_full:
        t5 = perf_counter()
        with (debug_directory / f"{run_name}_full.pkl").open("wb") as fp:
            pickle.dump(d, fp, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[train] full dict pickle dump done in {perf_counter() - t5:0.1f}s", flush=True)

    print(f"[train] total train() time {perf_counter() - t0:0.1f}s", flush=True)

    return d, uc_d
