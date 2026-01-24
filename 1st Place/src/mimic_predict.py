# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 09:29:11 2024

@author: Yonatan
"""

import pickle
import os
from pathlib import Path

import pandas as pd
from mimic_common import IndexedDict, annotate_with_dict, common_headers, remove_overlaps
from mimic_postprocess_attributes import postprocess_annotations
from tqdm import tqdm

data_directory = Path(__file__).parent.parent / "data"


_GLOBAL_TEXTS = None
_GLOBAL_HEADERS = None
_GLOBAL_DICT = None


def _annotate_one(note_id: str):
    return annotate_with_dict(_GLOBAL_TEXTS[note_id], _GLOBAL_DICT, _GLOBAL_HEADERS, note_id)


def predict(texts, headers, d, submission, run_name):
    if submission and isinstance(d, dict):
        if str(os.environ.get("KIRI_INDEX", "1")).lower() not in {"0", "false", "no"}:
            d = IndexedDict(d)

    progress = str(os.environ.get("KIRI_PROGRESS", "0")).lower() in {"1", "true", "yes"}
    workers_raw = str(os.environ.get("KIRI_WORKERS", "0")).strip()
    try:
        workers = int(workers_raw) if workers_raw else 0
    except Exception:
        workers = 0
    if workers <= 0:
        workers = min(8, (os.cpu_count() or 1))

    pred = []
    if not submission:
        print("generating predictions")
    note_ids = list(texts.index)
    if submission and workers > 1 and os.name == "posix":
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
        it = note_ids
        if progress:
            it = tqdm(it, total=len(note_ids), disable=False)
        with ctx.Pool(processes=workers) as pool:
            for df in pool.imap(_annotate_one, note_ids, chunksize=chunksize):
                pred.append(df)
    else:
        for i in tqdm(note_ids, disable=(submission and not progress)):
            pred.append(annotate_with_dict(texts[i], d, headers, i))
    pred = pd.concat(pred)
    if not submission:
        pred.to_csv(f"../debug/{run_name}_pred.csv")
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
        fn = "../term_extension.csv"
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
