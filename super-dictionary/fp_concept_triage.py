#!/usr/bin/env python3
"""Triage FP concepts: which are genuinely broken dictionary matches vs
plausible medical concepts that could be TPs on different data?"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Triage FP concepts with risk classification."
    )
    parser.add_argument("--blocklist", required=True, help="Path to FP blocklist CSV")
    parser.add_argument("--dict-path", required=True, help="Path to KIRI synonym dictionary CSV")
    parser.add_argument("--test-pred", required=True, help="Path to test predictions CSV")
    parser.add_argument("--oof-pred", required=True, help="Path to OOF predictions CSV")
    parser.add_argument("--test-notes", required=True, help="Path to test notes CSV")
    parser.add_argument("--train-notes", required=True, help="Path to train notes CSV")
    parser.add_argument("--train-gold", required=True, help="Path to train gold annotations CSV")
    parser.add_argument("--test-gold", required=True, help="Path to test gold annotations CSV")
    parser.add_argument("--output", default="outputs/fp_blocklist_triaged.csv",
                        help="Output path for triaged blocklist")
    args = parser.parse_args(argv)

    blocklist = pd.read_csv(args.blocklist)
    fp_concepts = set(blocklist["concept_id"].tolist())

    dict_df = pd.read_csv(args.dict_path)
    test_pred = pd.read_csv(args.test_pred)
    oof_pred = pd.read_csv(args.oof_pred)
    test_notes = pd.read_csv(args.test_notes)
    train_notes = pd.read_csv(args.train_notes)
    train_gold = pd.read_csv(args.train_gold)
    test_gold = pd.read_csv(args.test_gold)

    for col in ["start", "end", "concept_id"]:
        test_pred[col] = test_pred[col].astype(int)
        oof_pred[col] = oof_pred[col].astype(int)
        train_gold[col] = train_gold[col].astype(int)
        test_gold[col] = test_gold[col].astype(int)

    notes_map = {}
    for _, row in test_notes.iterrows():
        notes_map[row["note_id"]] = row["text"]
    for _, row in train_notes.iterrows():
        notes_map[row["note_id"]] = row["text"]

    train_gold_concepts = set(train_gold["concept_id"].unique())
    test_gold_concepts = set(test_gold["concept_id"].unique())

    def get_matched_texts(pred_df, concept_id):
        rows = pred_df[pred_df["concept_id"] == concept_id]
        texts = []
        for _, r in rows.iterrows():
            note_text = notes_map.get(r["note_id"], "")
            span = note_text[int(r["start"]):int(r["end"])]
            texts.append(span.strip())
        return texts

    print("=" * 120)
    print("TRIAGE OF FP CONCEPTS")
    print("For each: dictionary synonyms, matched text, and risk assessment")
    print("=" * 120)

    for _, brow in blocklist.iterrows():
        cid = int(brow["concept_id"])
        cname = brow["concept_name"]

        syns = dict_df[dict_df["concept_id"] == cid]["concept_name"].tolist()

        test_texts = get_matched_texts(test_pred, cid)
        oof_texts = get_matched_texts(oof_pred, cid)

        test_unique = sorted(set(t.lower() for t in test_texts))
        oof_unique = sorted(set(t.lower() for t in oof_texts))
        all_matched = sorted(set(test_unique + oof_unique))

        syn_lengths = [len(s) for s in syns if isinstance(s, str)]
        min_syn_len = min(syn_lengths) if syn_lengths else 0

        in_train_gold = cid in train_gold_concepts
        in_test_gold = cid in test_gold_concepts
        has_short_syn = min_syn_len <= 4
        all_spans_short = all(len(t) <= 5 for t in all_matched) if all_matched else False

        if in_train_gold or in_test_gold:
            risk = "HIGH"
        elif has_short_syn and all_spans_short:
            risk = "LOW"
        elif any(t in ["less", "l", "d", "mrs", "wild", "shift", "catch",
                        "mac", "bas", "pep", "dish", "aps"] for t in all_matched):
            risk = "LOW"
        else:
            risk = "MEDIUM"

        print(f"\n{'─' * 120}")
        print(f"Concept {cid}: {cname}")
        print(f"  Risk: {risk} | In train gold: {in_train_gold} | In test gold: {in_test_gold}")
        print(f"  Dictionary synonyms ({len(syns)}): {syns[:8]}")
        print(f"  Shortest synonym length: {min_syn_len}")
        print(f"  Test matched texts ({len(test_texts)}): {test_unique[:6]}")
        print(f"  OOF matched texts ({len(oof_texts)}): {oof_unique[:6]}")

    # Summary
    print("\n" + "=" * 120)
    print("SUMMARY BY RISK LEVEL")
    print("=" * 120)

    results = []
    for _, brow in blocklist.iterrows():
        cid = int(brow["concept_id"])
        syns = dict_df[dict_df["concept_id"] == cid]["concept_name"].tolist()
        syn_lengths = [len(s) for s in syns if isinstance(s, str)]
        min_syn_len = min(syn_lengths) if syn_lengths else 0

        test_texts = get_matched_texts(test_pred, cid)
        oof_texts = get_matched_texts(oof_pred, cid)
        all_matched = sorted(set(t.lower().strip() for t in test_texts + oof_texts))
        has_short_syn = min_syn_len <= 4
        all_spans_short = all(len(t) <= 5 for t in all_matched) if all_matched else False
        in_train_gold = cid in train_gold_concepts
        in_test_gold = cid in test_gold_concepts

        if in_train_gold or in_test_gold:
            risk = "HIGH"
        elif has_short_syn and all_spans_short:
            risk = "LOW"
        elif any(t in ["less", "l", "d", "mrs", "wild", "shift", "catch",
                        "mac", "bas", "pep", "dish", "aps", "persistence",
                        "adjustment", "scheduling", "toilet", "collapse",
                        "constitutional"] for t in all_matched):
            risk = "LOW"
        else:
            risk = "MEDIUM"

        results.append({
            "concept_id": cid,
            "concept_name": brow["concept_name"],
            "risk": risk,
            "n_test_preds": len(test_texts),
            "n_oof_preds": len(oof_texts),
            "matched_texts": all_matched[:5],
            "min_syn_len": min_syn_len,
        })

    results_df = pd.DataFrame(results)
    for risk_level in ["LOW", "MEDIUM", "HIGH"]:
        subset = results_df[results_df["risk"] == risk_level]
        print(f"\n{risk_level} risk: {len(subset)} concepts")
        for _, r in subset.iterrows():
            texts = ", ".join(f'"{t}"' for t in r["matched_texts"][:3])
            print(f"  {r['concept_id']:>20}: {str(r['concept_name'])[:45]:<45} matched: {texts}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved triaged blocklist to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
