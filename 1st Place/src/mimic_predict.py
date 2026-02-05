# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 09:29:11 2024

@author: Yonatan
"""

import os
import pickle
import sys
from pathlib import Path

import pandas as pd
from mimic_common import IndexedDict, annotate_with_dict, common_headers, remove_overlaps
from mimic_postprocess_attributes import postprocess_annotations
from tqdm import tqdm

data_directory = Path(__file__).parent.parent / "data"


_GLOBAL_TEXTS = None
_GLOBAL_HEADERS = None
_GLOBAL_DICT = None


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def _annotate_one(note_id: str):
    return annotate_with_dict(_GLOBAL_TEXTS[note_id], _GLOBAL_DICT, _GLOBAL_HEADERS, note_id)


def predict(texts, headers, d, submission, run_name):
    # Default to using the prefilter index in dev-mode (non-submission), since it is
    # semantics-preserving and significantly reduces regex scans.
    use_index = _env_bool("KIRI_INDEX", not submission)
    if use_index and isinstance(d, dict):
        d = IndexedDict(d)

    want_parallel = _env_bool("KIRI_PARALLEL", submission)
    progress = _env_bool("KIRI_PROGRESS", not submission)
    unordered = _env_bool("KIRI_UNORDERED", True)
    maxtasks_raw = str(os.environ.get("KIRI_MAXTASKSPERCHILD", "")).strip()
    maxtasksperchild = None
    if maxtasks_raw:
        try:
            maxtasksperchild = int(maxtasks_raw)
        except Exception:
            maxtasksperchild = None
        if maxtasksperchild is not None and maxtasksperchild < 1:
            maxtasksperchild = None
    workers_raw = str(os.environ.get("KIRI_WORKERS", "")).strip()
    workers = 1
    if want_parallel:
        try:
            workers = int(workers_raw) if workers_raw else min(8, (os.cpu_count() or 1))
        except Exception:
            workers = min(8, (os.cpu_count() or 1))
        if workers < 1:
            workers = 1

    pred = []
    if not submission:
        print("generating predictions")
    note_ids = list(texts.index)
    pbar = None
    if progress:
        pbar = tqdm(
            total=len(note_ids),
            disable=False,
            dynamic_ncols=True,
            file=sys.stdout,
        )
    thread_mode = _env_bool("KIRI_THREAD", False)
    if want_parallel and workers > 1 and os.name == "posix" and not thread_mode:
        global _GLOBAL_TEXTS, _GLOBAL_HEADERS, _GLOBAL_DICT
        _GLOBAL_TEXTS = texts
        _GLOBAL_HEADERS = headers
        _GLOBAL_DICT = d

        chunksize_raw = str(os.environ.get("KIRI_CHUNKSIZE", "")).strip()
        try:
            chunksize = int(chunksize_raw) if chunksize_raw else 1
        except Exception:
            chunksize = 1
        if chunksize < 1:
            chunksize = 1

        import multiprocessing as mp

        ctx = mp.get_context("fork")
        with ctx.Pool(processes=workers, maxtasksperchild=maxtasksperchild) as pool:
            it = (
                pool.imap_unordered(_annotate_one, note_ids, chunksize=chunksize)
                if unordered
                else pool.imap(_annotate_one, note_ids, chunksize=chunksize)
            )
            for df in it:
                pred.append(df)
                if pbar is not None:
                    pbar.update(1)
    elif want_parallel and workers > 1 and thread_mode:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(annotate_with_dict, texts[nid], d, headers, nid)
                for nid in note_ids
            ]
            if unordered:
                for fut in as_completed(futures):
                    pred.append(fut.result())
                    if pbar is not None:
                        pbar.update(1)
            else:
                for fut in futures:
                    pred.append(fut.result())
                    if pbar is not None:
                        pbar.update(1)
    else:
        for i in note_ids:
            pred.append(annotate_with_dict(texts[i], d, headers, i))
            if pbar is not None:
                pbar.update(1)
    if pbar is not None:
        pbar.close()
    pred = pd.concat(pred)
    if not submission:
        debug_dir = data_directory.parent / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        pred.to_csv(debug_dir / f"{run_name}_pred.csv", index=False)
    return pred


def get_case_sensitive_dict():
    csd = {
        (("other", "Pertinent Results:"), "K"): 312468003,
        ("any", "T"): 105723007,
        (("other", "Pertinent Results:"), "Mg"): 271285000,
        ("Physical Exam:", "RA"): 722742002,
        (("other", "Pertinent Results:"), "Plt"): 61928009,
        (("other", "Pertinent Results:"), "MR"): 48724000,
    }
    return csd


def join_predictions(predictions):
    predictions = pd.concat(predictions)
    no_overlaps = []
    for nid in predictions["note_id"].unique():
        df = predictions.query(f'note_id == "{nid}"')
        no_overlaps.append(remove_overlaps(df))
    return pd.concat(no_overlaps)


def get_abbr_dict(submission: bool):
    if submission:
        path = Path("assets") / "abbr_dict.pkl"
    else:
        path = data_directory / "interim" / "abbr_dict.pkl"
    with path.open("rb") as f:
        abbr_dict = pickle.load(f)
    return abbr_dict


def get_attr_file(submission):
    if submission:
        fn = "assets/term_extension.csv"
    else:
        fn = data_directory / "interim" / "term_extension.csv"
    df = pd.read_csv(fn)
    return df


def make_predictions(texts, d, uc_dict, submission=False, run_name="default"):
    assert isinstance(uc_dict, dict)
    texts_lc = texts.str.lower()
    headers = [h.lower() for h in common_headers]

    pred_lc = predict(texts_lc, headers, d, submission, run_name + "_lc")
    uc_dict.update(get_case_sensitive_dict())
    abbr_dict = get_abbr_dict(submission)
    abbr_dict.update(uc_dict)
    uc_dict = abbr_dict
    pred_uc = predict(texts, common_headers, uc_dict, submission, run_name + "_uc")
    pred = join_predictions((pred_lc, pred_uc))

    att_df = get_attr_file(submission)
    pred = postprocess_annotations(texts, pred, att_df, submission)

    return pred
