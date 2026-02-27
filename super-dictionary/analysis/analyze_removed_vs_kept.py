#!/usr/bin/env python3
"""Analyze removed vs kept dictionary keys to find discriminative features.

Runs the training pipeline through scoring/removal, then computes features
on removed vs kept keys to understand what makes a bad dictionary entry.

Key insight: at test time we can see ALL notes before classifying, so
corpus-frequency features (how often does this mention appear in the notes?)
are directly usable at inference time, not just training time.
"""
from __future__ import annotations

import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# Add super-dictionary to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "super-dictionary"))
from train_dictionary import (
    COMMON_HEADERS, BLACKLIST_THRESH, INTERNAL_BLACKLIST,
    build_dict_from_annotations, score_dict, IndexedDict,
    count_correct, is_naive_key_remove, get_pattern,
    load_snomed_synonyms_from_super_dict, process_term,
    load_concept_types,
)


# ---------------------------------------------------------------------------
# Common English words
# ---------------------------------------------------------------------------
COMMON_ENGLISH = set("""
a about after again against all am an and any are as at be because been
before being below between both but by can could did do does doing down
during each few for from further get got had has have having he her here
hers herself him himself his how i if in into is it its itself just know
let like make me might more most my myself no nor not now of off on once
only or other our ours ourselves out over own quite rather really right
said same say she should so some still such take than that the their
theirs them themselves then there these they this those through to too
under until up us very want was we were what when where which while who
whom why will with would you your yours yourself yourselves
about above across after along also always another any back been before
began begin between both came can come could day did do down each end
even every far find first for found from get give go going good got great
had hand has have head help her here high him his home house how its just
keep kind know large last left life like line little long look made make
man many may men might more most much must my name need never new next
night no not now number off old only open order other our out over own
part people place point right run same saw say second see set she show
side since small so some something sometimes still story such take tell
than that the their them then there these they thing think this those
three through time to together too turn two under up us use very want
water way well went were what when where which while who why will with
without word work world would write year you young
change changes other negative follow mild normal used new old small large
high low long short open close left right back side
""".split())

MEDICAL_STOPWORDS = set("""
patient history present diagnosis treatment chronic acute status
assessment plan review positive negative reported denies mild moderate
severe significant normal abnormal noted stable improved worsening
bilateral right left upper lower anterior posterior lateral medial
proximal distal primary secondary initial subsequent prior current
recent onset duration episode recurrent intermittent continuous
""".split())


# ---------------------------------------------------------------------------
# Corpus frequency computation
# ---------------------------------------------------------------------------
def count_mention_occurrences(mention: str, texts_lc: pd.Series) -> int:
    """Count how many times a mention's regex pattern fires across all notes."""
    p = get_pattern(mention)
    if p is None:
        return 0
    total = 0
    for text in texts_lc:
        total += len(p.findall(text))
    return total


def build_corpus_frequency_index(
    texts_lc: pd.Series,
    word_counter: Counter,
) -> dict:
    """Build a lookup for word-level and bigram-level corpus frequencies."""
    # We already have word_counter from the training pipeline.
    # Also build bigram counts for multi-word lookup.
    bigram_counter = Counter()
    for text in texts_lc:
        words = text.split()
        for i in range(len(words) - 1):
            bigram_counter[(words[i], words[i + 1])] += 1
    return {"words": word_counter, "bigrams": bigram_counter}


def corpus_freq_features(
    mention: str,
    word_counter: Counter,
    corpus_total_words: int,
) -> dict:
    """Compute word-frequency-based features for a mention."""
    words = mention.lower().split()
    if not words:
        return {
            "max_word_freq": 0, "min_word_freq": 0, "mean_word_freq": 0,
            "log_max_word_freq": 0, "log_min_word_freq": 0,
            "freq_ratio_max_min": 0,
        }

    freqs = [word_counter.get(w, 0) for w in words]
    max_f = max(freqs)
    min_f = min(freqs)
    mean_f = np.mean(freqs)

    return {
        "max_word_freq": max_f,
        "min_word_freq": min_f,
        "mean_word_freq": mean_f,
        "log_max_word_freq": np.log1p(max_f),
        "log_min_word_freq": np.log1p(min_f),
        "freq_ratio_max_min": max_f / max(min_f, 1),
    }


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------
def compute_features(
    keys_with_concepts: list[tuple],
    word_counter: Counter,
    corpus_total_words: int,
    concept_types: pd.Series | None = None,
    mention_corpus_counts: dict[str, int] | None = None,
    mention_annotation_counts: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Compute features for a list of ((section, mention), concept_id) entries."""
    rows = []
    for key, concept_id in keys_with_concepts:
        section, mention = key
        mention_str = str(mention)
        words = mention_str.split()
        n_words = len(words)
        n_chars = len(mention_str)

        # Basic length features
        avg_word_len = np.mean([len(w) for w in words]) if words else 0

        # Common English overlap
        mention_words_lower = set(mention_str.lower().split())
        common_overlap = len(mention_words_lower & COMMON_ENGLISH) / max(len(mention_words_lower), 1)
        all_common = int(mention_words_lower.issubset(COMMON_ENGLISH))

        # Medical stopword overlap
        medical_stop_overlap = len(mention_words_lower & MEDICAL_STOPWORDS) / max(len(mention_words_lower), 1)

        # Character-level features
        has_digit = int(any(c.isdigit() for c in mention_str))
        has_hyphen = int("-" in mention_str)
        has_slash = int("/" in mention_str)

        # Specificity proxies
        is_single_word = int(n_words == 1)
        is_two_words = int(n_words == 2)
        is_multi_word = int(n_words >= 3)
        short_mention = int(n_chars <= 4)
        very_short = int(n_chars <= 2)

        # Section features
        is_any_section = int(section == "any" or isinstance(section, tuple))
        section_str = str(section)

        # Concept type
        concept_type = "unknown"
        if concept_types is not None and concept_id in concept_types.index:
            concept_type = concept_types[concept_id]

        # Medical suffix indicators
        has_medical_suffix = int(any(
            mention_str.endswith(s) for s in
            ("itis", "osis", "emia", "aemia", "ectomy", "ology", "pathy",
             "opathy", "plasty", "otomy", "oscopy", "ogenic", "algia")
        ))

        # Structural words
        has_of = int(" of " in mention_str)
        has_the = int(" the " in mention_str or mention_str.startswith("the "))
        has_with = int(" with " in mention_str)

        # --- CORPUS FREQUENCY FEATURES ---
        cf = corpus_freq_features(mention_str, word_counter, corpus_total_words)

        # Mention-level corpus count (exact pattern match count across notes)
        mention_corpus_count = 0
        if mention_corpus_counts is not None:
            mention_corpus_count = mention_corpus_counts.get(mention_str, 0)

        # Annotation count (how many times this mention was annotated in training)
        mention_ann_count = 0
        if mention_annotation_counts is not None:
            mention_ann_count = mention_annotation_counts.get(mention_str, 0)

        # Fire-without-annotation ratio
        # High ratio = mention appears in corpus way more than it's annotated
        # This is the killer feature: "discharge" fires 10000x but annotated 5x
        fires_without_ann_ratio = mention_corpus_count / max(mention_ann_count, 1)
        excess_fires = max(0, mention_corpus_count - mention_ann_count)

        row = {
            "section": section_str,
            "mention": mention_str,
            "concept_id": concept_id,
            "concept_type": concept_type,
            # Length
            "n_words": n_words,
            "n_chars": n_chars,
            "avg_word_len": avg_word_len,
            # Lexical
            "common_english_frac": common_overlap,
            "all_common_english": all_common,
            "medical_stop_frac": medical_stop_overlap,
            # Character
            "has_digit": has_digit,
            "has_hyphen": has_hyphen,
            "has_slash": has_slash,
            # Specificity
            "is_single_word": is_single_word,
            "is_two_words": is_two_words,
            "is_multi_word": is_multi_word,
            "short_mention": short_mention,
            "very_short": very_short,
            # Section
            "is_any_section": is_any_section,
            # Medical
            "has_medical_suffix": has_medical_suffix,
            "has_of": has_of,
            "has_the": has_the,
            "has_with": has_with,
            # Corpus frequency - word level
            "max_word_freq": cf["max_word_freq"],
            "min_word_freq": cf["min_word_freq"],
            "mean_word_freq": cf["mean_word_freq"],
            "log_max_word_freq": cf["log_max_word_freq"],
            "log_min_word_freq": cf["log_min_word_freq"],
            "freq_ratio_max_min": cf["freq_ratio_max_min"],
            # Corpus frequency - mention level
            "mention_corpus_count": mention_corpus_count,
            "log_mention_corpus_count": np.log1p(mention_corpus_count),
            "mention_ann_count": mention_ann_count,
            # Fire-without-annotation
            "fires_without_ann_ratio": fires_without_ann_ratio,
            "log_fires_without_ann_ratio": np.log1p(fires_without_ann_ratio),
            "excess_fires": excess_fires,
            "log_excess_fires": np.log1p(excess_fires),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    t0 = time.perf_counter()
    data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    super_dict_path = data_dir / "interim" / "super_dictionary_full.tsv"
    flat_term_path = Path(__file__).resolve().parent.parent.parent / "3rd Place" / "assets" / "dataflattened_terminology.csv"

    # Load data
    print("Loading data...", flush=True)
    train_notes = pd.read_csv(data_dir / "train_notes.csv")
    train_annotations = pd.read_csv(data_dir / "train_annotations.csv")

    texts = train_notes.set_index("note_id")["text"]
    texts_lc = texts.str.lower()
    headers = [h.lower() for h in COMMON_HEADERS]

    # Step 1: Build word counter + dynamic blacklist
    print("Building word counter...", flush=True)
    words_counter = Counter()
    for text in texts_lc:
        words_counter.update(text.split())
    corpus_total_words = sum(words_counter.values())
    blacklist_set = {w for w in words_counter if words_counter[w] > BLACKLIST_THRESH}
    blacklist_set.update(INTERNAL_BLACKLIST)
    print(f"  Corpus: {corpus_total_words:,} total words, {len(words_counter):,} unique")

    # Normalize annotations
    annotations = train_annotations.copy()
    if "span" in annotations.columns:
        annotations["source orig"] = annotations["span"]
        annotations["source"] = annotations["span"].str.lower()
    else:
        spans = []
        for _, row in annotations.iterrows():
            t = texts.get(row["note_id"], "")
            spans.append(t[int(row["start"]):int(row["end"])])
        annotations["source orig"] = spans
        annotations["source"] = [s.lower() for s in spans]
        annotations["span"] = annotations["source orig"]

    # Build annotation count per mention (how many times each mention is
    # annotated in training data - a mention that fires 1000x but is
    # annotated 2x is probably a bad key)
    print("Building annotation counts per mention...", flush=True)
    mention_annotation_counts = annotations["source"].value_counts().to_dict()

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
    for nid in texts_lc.index:
        ref = refs.get(str(nid))
        if ref is None:
            continue
        t_scores, _ = score_dict(texts_lc[nid], ref, d, d_indexed, headers)
        for k in t_scores:
            scores_by_note.setdefault(k, []).extend(t_scores[k])
            for s in [1, -1]:
                if s in t_scores[k]:
                    scores_by_note.setdefault(k, []).append(s)
    print(f"  Scoring done in {time.perf_counter() - t1:.1f}s")

    # Step 4: Identify removed vs kept
    print("Identifying removed vs kept keys...", flush=True)
    removed_keys = []
    kept_scored_keys = []
    for k in scores_by_note:
        if is_naive_key_remove(count_correct(scores_by_note[k]), k):
            removed_keys.append((k, d.get(k, -1)))
        else:
            kept_scored_keys.append((k, d.get(k, -1)))

    scored_set = set(scores_by_note.keys())
    unscored_keys = [(k, d[k]) for k in d if k not in scored_set]

    print(f"  Scored keys:   {len(scores_by_note)}")
    print(f"    Removed:     {len(removed_keys)}")
    print(f"    Kept:        {len(kept_scored_keys)}")
    print(f"  Unscored keys: {len(unscored_keys)}")

    # Load concept types
    concept_types = None
    if flat_term_path.exists():
        concept_types = load_concept_types(flat_term_path)

    # Load SNOMED synonyms
    print("\nLoading SNOMED synonyms (the unscored pool)...", flush=True)
    ft = pd.read_csv(flat_term_path)
    allowed_cids = set(ft["concept_id"].astype(int).unique())
    sno_syns = load_snomed_synonyms_from_super_dict(
        super_dict_path, allowed_concept_ids=allowed_cids,
    )
    snomed_new = [(k, sno_syns[k]) for k in sno_syns if k not in d]
    print(f"  SNOMED synonyms (new, not in training dict): {len(snomed_new)}")

    # -----------------------------------------------------------------------
    # Compute mention-level corpus counts (regex match counts across notes)
    # This is the expensive part - count how many times each mention fires.
    # We do this for all scored mentions + a sample of SNOMED.
    # -----------------------------------------------------------------------
    print("\nCounting mention occurrences in corpus...", flush=True)
    t2 = time.perf_counter()

    # Collect unique mentions from all groups
    all_mentions = set()
    for k, _ in removed_keys:
        all_mentions.add(k[1])
    for k, _ in kept_scored_keys:
        all_mentions.add(k[1])
    for k, _ in unscored_keys:
        all_mentions.add(k[1])

    # For SNOMED, take all of them (we need this for the classifier)
    # but batch it efficiently
    snomed_mentions = set()
    for k, _ in snomed_new:
        snomed_mentions.add(k[1])
    all_mentions.update(snomed_mentions)

    print(f"  Unique mentions to count: {len(all_mentions):,}")

    # Batch count: for each note, find all mentions that match
    # This is O(notes * mentions) in the worst case, but we use the
    # pattern cache so each mention's regex is compiled once.
    # For efficiency, we concatenate all notes and count per mention.
    corpus_text = "\n".join(texts_lc.values)
    mention_corpus_counts: dict[str, int] = {}
    done = 0
    batch_report = max(1, len(all_mentions) // 20)
    for mention in all_mentions:
        p = get_pattern(mention)
        if p is None:
            mention_corpus_counts[mention] = 0
            continue
        mention_corpus_counts[mention] = len(p.findall(corpus_text))
        done += 1
        if done % batch_report == 0:
            print(f"    {done:,}/{len(all_mentions):,} mentions counted...", flush=True)

    print(f"  Corpus counting done in {time.perf_counter() - t2:.1f}s")

    # -----------------------------------------------------------------------
    # Compute features for all groups
    # -----------------------------------------------------------------------
    print("\nComputing features...", flush=True)
    common_args = dict(
        word_counter=words_counter,
        corpus_total_words=corpus_total_words,
        concept_types=concept_types,
        mention_corpus_counts=mention_corpus_counts,
        mention_annotation_counts=mention_annotation_counts,
    )

    df_removed = compute_features(removed_keys, **common_args)
    df_removed["group"] = "removed"

    df_kept = compute_features(kept_scored_keys, **common_args)
    df_kept["group"] = "kept"

    df_unscored = compute_features(unscored_keys, **common_args)
    df_unscored["group"] = "unscored_training"

    df_snomed = compute_features(snomed_new, **common_args)
    df_snomed["group"] = "snomed_new"

    # Add scoring stats for scored entries
    for i, (k, _) in enumerate(removed_keys):
        scores = scores_by_note.get(k, [])
        s = pd.Series(scores)
        df_removed.loc[i, "n_correct"] = (s == 1).sum()
        df_removed.loc[i, "n_incorrect"] = (s == -1).sum()
        df_removed.loc[i, "precision"] = (s == 1).sum() / max(len(s), 1)

    for i, (k, _) in enumerate(kept_scored_keys):
        scores = scores_by_note.get(k, [])
        s = pd.Series(scores)
        df_kept.loc[i, "n_correct"] = (s == 1).sum()
        df_kept.loc[i, "n_incorrect"] = (s == -1).sum()
        df_kept.loc[i, "precision"] = (s == 1).sum() / max(len(s), 1)

    # -----------------------------------------------------------------------
    # Analysis
    # -----------------------------------------------------------------------
    all_df = pd.concat([df_removed, df_kept, df_unscored, df_snomed], ignore_index=True)

    feature_cols_v1 = [
        "n_words", "n_chars", "avg_word_len",
        "common_english_frac", "all_common_english", "medical_stop_frac",
        "has_digit", "has_hyphen", "has_slash",
        "is_single_word", "is_two_words", "is_multi_word",
        "short_mention", "very_short",
        "is_any_section", "has_medical_suffix",
        "has_of", "has_the", "has_with",
    ]

    corpus_freq_cols = [
        "log_max_word_freq", "log_min_word_freq",
        "freq_ratio_max_min",
        "log_mention_corpus_count",
        "mention_ann_count",
        "log_fires_without_ann_ratio",
        "log_excess_fires",
    ]

    feature_cols_v2 = feature_cols_v1 + corpus_freq_cols

    pd.set_option("display.max_columns", 30)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.3f}".format)

    print("\n" + "=" * 80)
    print("CORPUS FREQUENCY ANALYSIS")
    print("=" * 80)

    print("\n--- Corpus frequency stats by group ---")
    freq_stats = [
        "mention_corpus_count", "mention_ann_count",
        "fires_without_ann_ratio", "excess_fires",
        "max_word_freq", "min_word_freq", "mean_word_freq",
    ]
    for stat in freq_stats:
        print(f"\n  {stat}:")
        for grp in ["removed", "kept", "snomed_new"]:
            sub = all_df[all_df["group"] == grp][stat]
            print(f"    {grp:<20} mean={sub.mean():>12.1f}  median={sub.median():>10.1f}  "
                  f"p90={sub.quantile(0.9):>10.1f}  max={sub.max():>12.1f}")

    # The money question: corpus frequency vs annotation frequency
    print("\n\n--- Fire-without-annotation ratio: removed vs kept ---")
    print("(How many times does the mention fire in corpus per annotation?)")
    for grp, df_grp in [("removed", df_removed), ("kept", df_kept)]:
        sub = df_grp[df_grp["mention_corpus_count"] > 0]
        print(f"\n  {grp} (n={len(sub)} with corpus_count > 0):")
        for pct in [25, 50, 75, 90, 95, 99]:
            val = sub["fires_without_ann_ratio"].quantile(pct / 100)
            print(f"    p{pct}: {val:.1f}x")
        # Show the top offenders
        top = sub.nlargest(10, "fires_without_ann_ratio")
        print(f"\n    Top 10 by fire-without-annotation ratio:")
        for _, row in top.iterrows():
            print(f"      {row['mention']:<35} corpus={row['mention_corpus_count']:>6.0f} "
                  f"ann={row['mention_ann_count']:>4.0f} ratio={row['fires_without_ann_ratio']:>8.1f}x "
                  f"prec={row.get('precision', float('nan')):.2f}")

    # SNOMED entries ranked by corpus frequency
    print("\n\n--- SNOMED entries with HIGHEST corpus frequency ---")
    print("(These fire the most on training notes - highest FP risk)")
    top_snomed = df_snomed.nlargest(50, "mention_corpus_count")
    for _, row in top_snomed.iterrows():
        print(f"  corpus={row['mention_corpus_count']:>6.0f}  {row['mention']:<45} type={row['concept_type']}")

    # SNOMED entries: zero corpus frequency (would never fire anyway)
    n_zero = (df_snomed["mention_corpus_count"] == 0).sum()
    print(f"\n\n--- SNOMED entries with ZERO corpus hits: {n_zero:,} / {len(df_snomed):,} "
          f"({n_zero/max(len(df_snomed),1):.1%}) ---")
    print("(These never appear in training text, can't evaluate them here)")

    # -----------------------------------------------------------------------
    # Classifier comparison: v1 (no corpus freq) vs v2 (with corpus freq)
    # -----------------------------------------------------------------------
    print("\n\n" + "=" * 80)
    print("CLASSIFIER COMPARISON: v1 (surface features) vs v2 (+corpus frequency)")
    print("=" * 80)

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import precision_recall_curve, classification_report

        scored_df = pd.concat([df_removed, df_kept], ignore_index=True)
        scored_df["label"] = (scored_df["group"] == "removed").astype(int)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for name, feat_cols in [("v1 (surface only)", feature_cols_v1),
                                ("v2 (+corpus freq)", feature_cols_v2)]:
            X = scored_df[feat_cols].values
            y = scored_df["label"].values

            # Logistic Regression
            lr_pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ])
            lr_auc = cross_val_score(lr_pipe, X, y, cv=skf, scoring="roc_auc")

            # Gradient Boosting (handles non-linear interactions better)
            gb = GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.1,
                subsample=0.8, random_state=42,
            )
            gb_auc = cross_val_score(gb, X, y, cv=skf, scoring="roc_auc")

            print(f"\n  {name}:")
            print(f"    Logistic Regression  ROC-AUC: {lr_auc.mean():.3f} (+/- {lr_auc.std():.3f})")
            print(f"    Gradient Boosting    ROC-AUC: {gb_auc.mean():.3f} (+/- {gb_auc.std():.3f})")

        # Fit final v2 model and show importances
        print("\n\n--- Feature importances (v2 Logistic Regression) ---")
        lr_pipe_v2 = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])
        X_v2 = scored_df[feature_cols_v2].values
        y = scored_df["label"].values
        lr_pipe_v2.fit(X_v2, y)
        coefs = lr_pipe_v2.named_steps["lr"].coef_[0]
        feat_importance = pd.Series(coefs, index=feature_cols_v2).sort_values(key=abs, ascending=False)
        print(f"{'Feature':<35} {'Coef':>8}  Direction")
        print("-" * 60)
        for feat, coef in feat_importance.items():
            direction = "-> REMOVE" if coef > 0 else "-> KEEP"
            marker = " ***" if abs(coef) > 0.3 else ""
            print(f"  {feat:<33} {coef:>+8.3f}  {direction}{marker}")

        # Fit GBM for feature importance too
        print("\n\n--- Feature importances (v2 Gradient Boosting) ---")
        gb_v2 = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.8, random_state=42,
        )
        gb_v2.fit(X_v2, y)
        gb_importance = pd.Series(
            gb_v2.feature_importances_, index=feature_cols_v2
        ).sort_values(ascending=False)
        print(f"{'Feature':<35} {'Importance':>10}")
        print("-" * 50)
        for feat, imp in gb_importance.items():
            marker = " ***" if imp > 0.05 else ""
            print(f"  {feat:<33} {imp:>10.4f}{marker}")

        # ---------------------------------------------------------------
        # Apply v2 model to SNOMED entries
        # ---------------------------------------------------------------
        print("\n\n" + "=" * 80)
        print("APPLYING CLASSIFIER TO SNOMED ENTRIES")
        print("=" * 80)

        if not df_snomed.empty:
            X_snomed = df_snomed[feature_cols_v2].values

            # Use both models
            lr_probs = lr_pipe_v2.predict_proba(X_snomed)[:, 1]
            gb_probs = gb_v2.predict_proba(X_snomed)[:, 1]
            # Ensemble
            ensemble_probs = (lr_probs + gb_probs) / 2
            df_snomed["lr_removal_prob"] = lr_probs
            df_snomed["gb_removal_prob"] = gb_probs
            df_snomed["ensemble_removal_prob"] = ensemble_probs

            print(f"\nRemoval probability distribution (ensemble):")
            for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                n = (ensemble_probs > threshold).sum()
                print(f"  > {threshold:.1f}: {n:>6} ({n/len(ensemble_probs):.1%})")

            print(f"\n\nTop 50 SNOMED entries most likely to be bad (ensemble):")
            risky = df_snomed.nlargest(50, "ensemble_removal_prob")
            for _, row in risky.iterrows():
                print(f"  p={row['ensemble_removal_prob']:.3f}  "
                      f"corpus={row['mention_corpus_count']:>5.0f}  "
                      f"{row['mention']:<45} type={row['concept_type']}")

            print(f"\n\nSNOMED entries with HIGH corpus frequency AND high removal prob:")
            high_risk = df_snomed[
                (df_snomed["mention_corpus_count"] > 10) &
                (df_snomed["ensemble_removal_prob"] > 0.4)
            ].sort_values("mention_corpus_count", ascending=False)
            print(f"  Count: {len(high_risk)}")
            for _, row in high_risk.head(50).iterrows():
                print(f"  p={row['ensemble_removal_prob']:.3f}  "
                      f"corpus={row['mention_corpus_count']:>5.0f}  "
                      f"{row['mention']:<45} type={row['concept_type']}")

            # Breakdown: what fraction of SNOMED entries that DO fire in
            # corpus are flagged?
            fires = df_snomed[df_snomed["mention_corpus_count"] > 0]
            print(f"\n\nAmong SNOMED entries that fire in training corpus "
                  f"({len(fires):,} / {len(df_snomed):,}):")
            for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
                n = (fires["ensemble_removal_prob"] > threshold).sum()
                print(f"  ensemble_prob > {threshold}: {n:>5} "
                      f"({n/max(len(fires),1):.1%})")

        # ---------------------------------------------------------------
        # Practical test: what if we use corpus frequency as a simple
        # test-time gate? (no classifier needed)
        # ---------------------------------------------------------------
        print("\n\n" + "=" * 80)
        print("SIMPLE CORPUS-FREQUENCY GATE (no classifier)")
        print("=" * 80)
        print("Idea: at test time, count how often each SNOMED mention fires")
        print("in the test notes. If it fires above a threshold, flag it.\n")

        # On training data, what threshold separates removed from kept?
        for grp_name, df_grp in [("removed", df_removed), ("kept", df_kept)]:
            mc = df_grp["mention_corpus_count"]
            print(f"  {grp_name:<10} corpus count: "
                  f"mean={mc.mean():>8.1f}  median={mc.median():>6.1f}  "
                  f"p90={mc.quantile(0.9):>8.1f}  p99={mc.quantile(0.99):>10.1f}")

        # Simulate: for training dict entries, if we gate on
        # "mention fires > N times AND has no annotation nearby", how many
        # removed/kept do we catch?
        print("\n  Simulated gate: mention_corpus_count > threshold")
        print(f"  {'Threshold':>10} {'Removed caught':>15} {'Kept wrongly flagged':>22} {'Precision':>10}")
        print("  " + "-" * 60)
        for thr in [5, 10, 20, 50, 100, 200, 500, 1000]:
            r_flagged = (df_removed["mention_corpus_count"] > thr).sum()
            k_flagged = (df_kept["mention_corpus_count"] > thr).sum()
            total_flagged = r_flagged + k_flagged
            prec = r_flagged / max(total_flagged, 1)
            print(f"  {thr:>10} {r_flagged:>10} / {len(df_removed):<5} "
                  f"{k_flagged:>15} / {len(df_kept):<5}   {prec:>9.1%}")

        # Better: fire-without-annotation ratio threshold
        print(f"\n  Simulated gate: fires_without_ann_ratio > threshold")
        print(f"  {'Threshold':>10} {'Removed caught':>15} {'Kept wrongly flagged':>22} {'Precision':>10}")
        print("  " + "-" * 60)
        for thr in [2, 5, 10, 20, 50, 100, 500]:
            r_flagged = (df_removed["fires_without_ann_ratio"] > thr).sum()
            k_flagged = (df_kept["fires_without_ann_ratio"] > thr).sum()
            total_flagged = r_flagged + k_flagged
            prec = r_flagged / max(total_flagged, 1)
            print(f"  {thr:>10} {r_flagged:>10} / {len(df_removed):<5} "
                  f"{k_flagged:>15} / {len(df_kept):<5}   {prec:>9.1%}")

        # Compound gate: corpus_count > T1 AND fires_without_ann_ratio > T2
        print(f"\n  Compound gate: corpus_count > T1 AND fires/ann_ratio > T2")
        print(f"  {'T1':>5} {'T2':>5} {'Removed':>10} {'Kept flagged':>14} {'Prec':>8} {'SNOMED flagged':>16}")
        print("  " + "-" * 65)
        for t1, t2 in [(10, 5), (10, 10), (20, 5), (20, 10), (50, 5), (50, 10), (100, 5)]:
            r_f = ((df_removed["mention_corpus_count"] > t1) &
                   (df_removed["fires_without_ann_ratio"] > t2)).sum()
            k_f = ((df_kept["mention_corpus_count"] > t1) &
                   (df_kept["fires_without_ann_ratio"] > t2)).sum()
            # For SNOMED entries, fires_without_ann_ratio = corpus_count
            # (since ann_count = 0, ratio = corpus_count / 1 = corpus_count)
            s_f = (df_snomed["mention_corpus_count"] > t1).sum()
            prec = r_f / max(r_f + k_f, 1)
            print(f"  {t1:>5} {t2:>5} {r_f:>6}/{len(df_removed):<5} "
                  f"{k_f:>10}/{len(df_kept):<5} {prec:>7.1%}  {s_f:>10}/{len(df_snomed)}")

    except ImportError as e:
        print(f"  sklearn not available ({e}), skipping classifier")

    print(f"\nTotal analysis time: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
