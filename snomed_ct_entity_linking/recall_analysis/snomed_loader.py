"""Load and filter SNOMED CT concepts and descriptions for indexing."""

import re
import time

import pandas as pd
from .config import Config

# SNOMED RF2 type IDs
FSN_TYPE_ID = 900000000000003001    # Fully Specified Name
SYNONYM_TYPE_ID = 900000000000013009  # Synonym

# Language Refset acceptability
PREFERRED_ACCEPTABILITY = 900000000000548007
ACCEPTABLE_ACCEPTABILITY = 900000000000549004

# Relationship type
IS_A_TYPE = 116680003

SEMANTIC_TAG_RE = re.compile(r"\(([^)]+)\)\s*$")


def parse_semantic_tag(term: str) -> str | None:
    """Extract semantic tag from a SNOMED FSN, e.g. 'Headache (finding)' -> 'finding'."""
    m = SEMANTIC_TAG_RE.search(term)
    return m.group(1) if m else None


def strip_semantic_tag(term: str) -> str:
    """Remove semantic tag from FSN, e.g. 'Headache (finding)' -> 'Headache'."""
    return SEMANTIC_TAG_RE.sub("", term).strip()


def load_snomed(
    cfg: Config,
) -> tuple[pd.DataFrame, dict[int, str], dict[int, str], dict[int, set[int]]]:
    """
    Load active SNOMED CT concepts filtered to clinical finding / disorder hierarchy.

    Returns:
        descriptions_df: DataFrame with columns
            [description_id, sctid, index_term, term_type, semantic_tag, is_preferred]
        sctid_to_fsn: Dict mapping SCTID -> FSN (without semantic tag) for display.
        sctid_to_tag: Dict mapping ALL active SCTIDs -> their semantic tag.
        parent_map: Dict mapping SCTID -> set of parent SCTIDs (IS-A hierarchy).
    """
    print(f"Loading concepts from {cfg.concept_file} ...")
    concepts = pd.read_csv(
        cfg.concept_file, sep="\t", dtype={"id": int, "active": int},
        usecols=["id", "active"],
    )
    active_concept_ids = set(concepts.loc[concepts["active"] == 1, "id"])
    print(f"  Active concepts: {len(active_concept_ids):,}")

    print(f"Loading descriptions from {cfg.description_file} ...")
    desc = pd.read_csv(
        cfg.description_file, sep="\t",
        dtype={"id": int, "active": int, "conceptId": int, "typeId": int},
        usecols=["id", "active", "conceptId", "typeId", "term"],
    )
    # Keep only active descriptions for active concepts
    desc = desc[(desc["active"] == 1) & (desc["conceptId"].isin(active_concept_ids))]
    print(f"  Active descriptions: {len(desc):,}")

    # Identify FSNs and parse their semantic tags
    fsn_mask = desc["typeId"] == FSN_TYPE_ID
    fsn_df = desc[fsn_mask].copy()
    fsn_df["semantic_tag"] = fsn_df["term"].apply(parse_semantic_tag)

    # Build full SCTID -> semantic tag lookup (all active concepts)
    sctid_to_tag: dict[int, str] = {}
    for _, row in fsn_df.iterrows():
        if row["semantic_tag"]:
            sctid_to_tag[row["conceptId"]] = row["semantic_tag"]

    # Filter for target semantic tags
    target_tags = set(cfg.semantic_tags)
    matching_fsns = fsn_df[fsn_df["semantic_tag"].isin(target_tags)]
    target_sctids = set(matching_fsns["conceptId"])
    print(f"  Concepts matching tags {target_tags}: {len(target_sctids):,}")

    # Build FSN lookup (stripped of semantic tag)
    sctid_to_fsn: dict[int, str] = {}
    for _, row in matching_fsns.iterrows():
        sctid_to_fsn[row["conceptId"]] = strip_semantic_tag(row["term"])

    # Get ALL descriptions (FSN + synonyms) for the target concepts
    filtered = desc[desc["conceptId"].isin(target_sctids)].copy()
    filtered["semantic_tag"] = filtered["term"].apply(parse_semantic_tag)
    filtered["term_type"] = filtered["typeId"].map({
        FSN_TYPE_ID: "fsn", SYNONYM_TYPE_ID: "synonym",
    }).fillna("other")

    # For synonyms (no semantic tag), store as-is. For FSNs, strip the tag for indexing.
    filtered["index_term"] = filtered.apply(
        lambda r: strip_semantic_tag(r["term"]) if r["term_type"] == "fsn" else r["term"],
        axis=1,
    )

    # Deduplicate identical terms for the same concept
    filtered = filtered.drop_duplicates(subset=["conceptId", "index_term"])

    # --- Load Language Refset for preferred term identification ---
    preferred_desc_ids = _load_preferred_descriptions(cfg)
    filtered["is_preferred"] = filtered["id"].isin(preferred_desc_ids)

    result = filtered.rename(columns={
        "id": "description_id", "conceptId": "sctid",
    })[["description_id", "sctid", "index_term", "term_type", "semantic_tag", "is_preferred"]].reset_index(drop=True)

    n_preferred = result["is_preferred"].sum()
    print(f"  Indexable descriptions (multi-vector): {len(result):,}")
    n_concepts = result["sctid"].nunique()
    print(f"  Unique concepts: {n_concepts:,} | Avg descriptions/concept: {len(result) / n_concepts:.1f}")
    print(f"  Preferred terms in index: {n_preferred:,} ({n_preferred / len(result) * 100:.1f}%)")

    # --- Load IS-A hierarchy ---
    parent_map = _load_hierarchy(cfg)

    return result, sctid_to_fsn, sctid_to_tag, parent_map


def _load_preferred_descriptions(cfg: Config) -> set[int]:
    """Load language refset and return set of preferred description IDs."""
    print(f"Loading language refset from {cfg.language_refset_file} ...")
    t0 = time.time()
    lang = pd.read_csv(
        cfg.language_refset_file, sep="\t",
        dtype={"referencedComponentId": int, "acceptabilityId": int, "active": int},
        usecols=["active", "referencedComponentId", "acceptabilityId"],
    )
    preferred = lang[
        (lang["active"] == 1) &
        (lang["acceptabilityId"] == PREFERRED_ACCEPTABILITY)
    ]
    preferred_ids = set(preferred["referencedComponentId"])
    print(f"  Preferred descriptions: {len(preferred_ids):,} (loaded in {time.time() - t0:.1f}s)")
    return preferred_ids


def _load_hierarchy(cfg: Config) -> dict[int, set[int]]:
    """Load active IS-A relationships and build parent map."""
    print(f"Loading IS-A hierarchy from {cfg.relationship_file} ...")
    t0 = time.time()
    rels = pd.read_csv(
        cfg.relationship_file, sep="\t",
        dtype={"sourceId": int, "destinationId": int, "typeId": int, "active": int},
        usecols=["active", "sourceId", "destinationId", "typeId"],
    )

    # Filter to active IS-A relationships
    is_a = rels[(rels["active"] == 1) & (rels["typeId"] == IS_A_TYPE)]
    print(f"  Active IS-A relationships: {len(is_a):,}")

    # Build parent map: child -> set of parents
    parent_map: dict[int, set[int]] = {}
    sources = is_a["sourceId"].values
    destinations = is_a["destinationId"].values
    for src, dst in zip(sources, destinations):
        src, dst = int(src), int(dst)
        if src not in parent_map:
            parent_map[src] = set()
        parent_map[src].add(dst)

    print(f"  Parent map: {len(parent_map):,} concepts with parents (loaded in {time.time() - t0:.1f}s)")
    return parent_map


def load_validation_data(cfg: Config) -> pd.DataFrame:
    """
    Load validation examples from train_annotations.csv.

    Uses annotation_type == 'test' as the validation split.

    Returns:
        DataFrame with columns [annotation_id, note_id, start, end, span, concept_id, context]
        where 'context' is the ±N token window around the mention.
    """
    print("Loading annotations ...")
    ann = pd.read_csv(cfg.train_annotations, dtype={"concept_id": int})
    # Use 'test' split as validation
    val = ann[ann["annotation_type"] == "test"].copy()
    val["start"] = val["start"].astype(int)
    val["end"] = val["end"].astype(int)
    print(f"  Validation annotations (type='test'): {len(val):,}")

    print("Loading notes ...")
    notes = pd.read_csv(cfg.train_notes)
    note_texts = dict(zip(notes["note_id"], notes["text"]))

    # Build context windows
    print(f"Building context windows (±{cfg.context_window_tokens} tokens) ...")
    contexts = []
    for _, row in val.iterrows():
        text = note_texts.get(row["note_id"], "")
        ctx = _build_context_window(text, row["start"], row["end"], cfg.context_window_tokens)
        contexts.append(ctx)
    val["context"] = contexts

    return val[["annotation_id", "note_id", "start", "end", "span", "concept_id", "context"]].reset_index(drop=True)


def _build_context_window(text: str, start: int, end: int, n_tokens: int) -> str:
    """
    Extract a window of ±n_tokens around the mention span.

    Returns the contextualized sentence with the mention embedded in surrounding text.
    """
    # Tokenize crudely by whitespace to find token boundaries
    mention = text[start:end]

    # Find word boundaries before the mention
    prefix = text[:start]
    prefix_tokens = prefix.split()
    if len(prefix_tokens) > n_tokens:
        # Take last n_tokens words
        window_start = prefix.rfind(" ", 0, prefix.rfind(" "))  # rough
        # More precise: rejoin last N tokens
        kept_prefix = " ".join(prefix_tokens[-n_tokens:])
    else:
        kept_prefix = prefix

    # Find word boundaries after the mention
    suffix = text[end:]
    suffix_tokens = suffix.split()
    if len(suffix_tokens) > n_tokens:
        kept_suffix = " ".join(suffix_tokens[:n_tokens])
    else:
        kept_suffix = suffix

    context = (kept_prefix + text[start:end] + kept_suffix).strip()
    # Collapse whitespace
    context = re.sub(r"\s+", " ", context)
    return context
