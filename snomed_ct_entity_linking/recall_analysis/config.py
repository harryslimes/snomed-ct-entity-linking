"""Configuration for the Recall@K analysis pipeline."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # --- Paths ---
    project_root: Path = Path("/workspaces/snomed-ct-entity-linking")
    snomed_root: Path = Path(
        "data/SnomedCT_InternationalRF2_PRODUCTION_20260101T120000Z/Snapshot/Terminology"
    )
    train_annotations: Path = Path("data/train_annotations.csv")
    train_notes: Path = Path("data/train_notes.csv")
    index_dir: Path = Path("snomed_index")
    output_dir: Path = Path("outputs/recall_analysis")

    # --- Models ---
    embedding_model: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext-mean-token"
    cross_encoder_model: str = "ncbi/MedCPT-Cross-Encoder"

    # --- SNOMED Filtering ---
    semantic_tags: tuple = ("finding", "disorder")

    # --- Context Windowing ---
    context_window_tokens: int = 50  # ±N tokens around mention

    # --- Retrieval ---
    dense_top_k: int = 100
    sparse_top_k: int = 100
    rrf_k: int = 60  # RRF constant
    fusion_top_k: int = 50  # candidates after fusion

    # --- Batching (RTX 5090 / 32GB VRAM) ---
    index_batch_size: int = 1024   # for encoding SNOMED descriptions
    query_batch_size: int = 256    # for encoding validation queries
    rerank_batch_size: int = 64    # for cross-encoder inference
    bm25_threads: int = 16         # threads for BM25S retrieval

    # --- Evaluation ---
    recall_k_values: tuple = (1, 5, 10, 50)

    def __post_init__(self):
        # Resolve relative paths against project root
        for attr in [
            "snomed_root", "train_annotations", "train_notes",
            "index_dir", "output_dir",
        ]:
            val = getattr(self, attr)
            if not val.is_absolute():
                setattr(self, attr, self.project_root / val)

    @property
    def concept_file(self) -> Path:
        return self.snomed_root / "sct2_Concept_Snapshot_INT_20260101.txt"

    @property
    def description_file(self) -> Path:
        return self.snomed_root / "sct2_Description_Snapshot-en_INT_20260101.txt"

    @property
    def relationship_file(self) -> Path:
        return self.snomed_root / "sct2_Relationship_Snapshot_INT_20260101.txt"

    @property
    def language_refset_file(self) -> Path:
        return (
            self.snomed_root.parent
            / "Refset" / "Language"
            / "der2_cRefset_LanguageSnapshot-en_INT_20260101.txt"
        )
