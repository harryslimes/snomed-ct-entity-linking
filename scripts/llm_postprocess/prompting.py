from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptConfig:
    context_chars: int = 64
    max_spans_in_prompt: int = 80
    suspicious_only: bool = False


def _is_suspicious_mention(text: str, context: str) -> bool:
    t = " ".join((text or "").strip().split())
    if not t:
        return False

    tl = t.lower()
    tc = t.upper()
    ctx = (context or "").lower()

    # Structural / label tokens.
    if tl in {"labs", "vitals", "vs"}:
        return True

    # Common lab / vital abbreviations that are frequently spurious in this task.
    lab_tokens = {
        "wbc",
        "rbc",
        "hgb",
        "hct",
        "mcv",
        "mch",
        "mchc",
        "rdw",
        "plt",
        "bun",
        "urea n",
        "na",
        "k",
        "cl",
        "co2",
        "bp",
        "hr",
        "rr",
        "spo2",
        "o2",
        "po2",
    }
    if tl in lab_tokens:
        return True
    if t == tc and tl in lab_tokens:
        return True

    # Single-letter / roman numeral / unit fragments.
    if len(t) <= 2 and (t.isalpha() or t.isdigit()):
        if t in {"I", "II", "III", "IV", "V"}:
            return True
        if tl in {"t", "l", "k"}:
            return True

    # Section-driven suspiciousness.
    if "admission labs" in ctx or "pertinent results" in ctx or "lab" in ctx:
        if tl in lab_tokens or tl in {"labs"}:
            return True

    # Generic boilerplate adjectives.
    if tl in {"clear", "stable"} and ("physical exam" in ctx or "pe:" in ctx or "lungs" in ctx):
        return True

    return False


def render_template(template: str, variables: dict[str, str]) -> str:
    out = template
    for k, v in variables.items():
        out = out.replace("{{" + k + "}}", v)
    return out


def build_pred_rows_for_prompt(
    *,
    note_text: str,
    pred_note_sorted,
    cfg: PromptConfig,
) -> str:
    # pred_note_sorted has columns: start,end,concept_id and is already stably sorted.
    rows = []
    note_len = len(note_text)
    included = 0
    for stable_idx, (start, end, cid) in enumerate(
        pred_note_sorted[["start", "end", "concept_id"]].itertuples(index=False, name=None)
    ):
        if included >= cfg.max_spans_in_prompt:
            rows.append({"idx": stable_idx, "truncated": True})
            break
        start_i = max(0, min(int(start), note_len))
        end_i = max(start_i, min(int(end), note_len))
        mention = note_text[start_i:end_i]
        l = max(0, start_i - cfg.context_chars)
        r = min(note_len, end_i + cfg.context_chars)
        context = note_text[l:r]
        if cfg.suspicious_only and not _is_suspicious_mention(mention, context):
            continue
        rows.append(
            {
                "idx": stable_idx,
                "start": start_i,
                "end": end_i,
                "concept_id": int(cid),
                "text": mention,
                "context": context,
            }
        )
        included += 1
    import json

    return json.dumps(rows, ensure_ascii=False, indent=2)
