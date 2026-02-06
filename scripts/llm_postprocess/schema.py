from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Literal


Op = Literal["delete", "shift"]


@dataclass(frozen=True)
class AgentConstraints:
    max_edits_per_note: int = 10
    max_shift_chars: int = 30
    forbid_new_concept_ids: bool = True


@dataclass(frozen=True)
class Span:
    idx: int
    start: int
    end: int
    concept_id: int


@dataclass(frozen=True)
class Edit:
    op: Op
    idx: int
    start: int | None = None
    end: int | None = None


@dataclass(frozen=True)
class EditScript:
    note_id: str
    edits: tuple[Edit, ...]


class ScriptError(ValueError):
    pass


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_first_json_object(text: str) -> str:
    """Extract the first JSON object-ish substring from model output.

    SGLang JSON-constrained decoding should make this unnecessary most of the time,
    but it's a useful fallback for occasional formatting drift.
    """
    if not isinstance(text, str):
        raise ScriptError("model output is not a string")
    m = _JSON_OBJECT_RE.search(text.strip())
    if not m:
        raise ScriptError("no JSON object found in output")
    return m.group(0)


def _require_int(x: Any, path: str) -> int:
    if isinstance(x, bool) or not isinstance(x, (int, float, str)):
        raise ScriptError(f"{path} must be an int")
    try:
        return int(x)
    except Exception as e:
        raise ScriptError(f"{path} must be an int") from e


def parse_script(raw_text: str) -> dict[str, Any]:
    raw = raw_text.strip()
    try:
        return json.loads(raw)
    except Exception:
        obj = extract_first_json_object(raw_text)
        return json.loads(obj)


def coerce_edit_script(obj: dict[str, Any]) -> EditScript:
    if not isinstance(obj, dict):
        raise ScriptError("script must be a JSON object")
    note_id = obj.get("note_id")
    if not isinstance(note_id, str) or not note_id:
        raise ScriptError("note_id must be a non-empty string")
    edits_obj = obj.get("edits")
    if edits_obj is None:
        edits_obj = []
    if not isinstance(edits_obj, list):
        raise ScriptError("edits must be a list")

    edits: list[Edit] = []
    for i, e in enumerate(edits_obj):
        path = f"edits[{i}]"
        if not isinstance(e, dict):
            raise ScriptError(f"{path} must be an object")
        op = e.get("op")
        if op not in ("delete", "shift"):
            raise ScriptError(f"{path}.op must be 'delete' or 'shift'")
        idx = _require_int(e.get("idx"), f"{path}.idx")
        start = e.get("start")
        end = e.get("end")
        if op == "shift":
            if start is None or end is None:
                raise ScriptError(f"{path} shift requires start and end")
            start_i = _require_int(start, f"{path}.start")
            end_i = _require_int(end, f"{path}.end")
            edits.append(Edit(op=op, idx=idx, start=start_i, end=end_i))
        else:
            edits.append(Edit(op=op, idx=idx))

    return EditScript(note_id=note_id, edits=tuple(edits))


def validate_and_normalize_edits(
    *,
    script: EditScript,
    expected_note_id: str,
    spans: list[Span],
    note_len: int | None,
    constraints: AgentConstraints,
) -> tuple[EditScript, list[str]]:
    """Validate script against note spans + constraints.

    Returns:
      (normalized_script, warnings)
    """
    warnings: list[str] = []
    if script.note_id != expected_note_id:
        warnings.append(
            f"note_id mismatch: script={script.note_id!r} expected={expected_note_id!r}"
        )
        script = EditScript(note_id=expected_note_id, edits=script.edits)

    n_spans = len(spans)
    if n_spans == 0:
        if script.edits:
            warnings.append("no spans for note; dropping all edits")
        return EditScript(note_id=expected_note_id, edits=()), warnings

    # Merge ops per idx while preserving intent: delete wins over shift, and the
    # last shift for an idx wins.
    delete_idx: set[int] = set()
    shift_by_idx: dict[int, tuple[int, int]] = {}
    for e in script.edits[: max(0, constraints.max_edits_per_note)]:
        if e.idx < 0 or e.idx >= n_spans:
            warnings.append(f"dropping edit with out-of-range idx={e.idx}")
            continue
        if e.op == "delete":
            delete_idx.add(e.idx)
            shift_by_idx.pop(e.idx, None)
        else:
            assert e.start is not None and e.end is not None
            shift_by_idx[e.idx] = (int(e.start), int(e.end))

    normalized: list[Edit] = []
    for idx in sorted(delete_idx):
        normalized.append(Edit(op="delete", idx=idx))
    for idx in sorted(shift_by_idx):
        start, end = shift_by_idx[idx]
        normalized.append(Edit(op="shift", idx=idx, start=start, end=end))

    # Validate shift constraints.
    span_by_idx = {s.idx: s for s in spans}
    ok_edits: list[Edit] = []
    for e in normalized:
        if e.op == "delete":
            ok_edits.append(e)
            continue
        s0 = span_by_idx.get(e.idx)
        if s0 is None:
            warnings.append(f"dropping shift for missing idx={e.idx}")
            continue
        assert e.start is not None and e.end is not None
        start, end = int(e.start), int(e.end)
        if start < 0 or end < 0:
            warnings.append(f"dropping shift idx={e.idx}: negative bounds")
            continue
        if note_len is not None and (start > note_len or end > note_len):
            warnings.append(f"dropping shift idx={e.idx}: out of note bounds")
            continue
        if start >= end:
            warnings.append(f"dropping shift idx={e.idx}: start>=end")
            continue
        if abs(start - s0.start) > constraints.max_shift_chars or abs(end - s0.end) > constraints.max_shift_chars:
            warnings.append(f"dropping shift idx={e.idx}: exceeds max_shift_chars")
            continue
        ok_edits.append(Edit(op="shift", idx=e.idx, start=start, end=end))

    if len(ok_edits) > constraints.max_edits_per_note:
        warnings.append("too many edits; truncating")
        ok_edits = ok_edits[: constraints.max_edits_per_note]

    return EditScript(note_id=expected_note_id, edits=tuple(ok_edits)), warnings


def json_schema_any_object(*, max_edits: int = 10) -> str:
    """A permissive JSON schema for constrained decoding.

    We validate strictly in Python, but this encourages the model to emit a JSON
    object with the right high-level fields.
    """
    schema = {
        "type": "object",
        "properties": {
            "note_id": {"type": "string"},
            "edits": {
                "type": "array",
                "maxItems": max(0, int(max_edits)),
                "items": {
                    "type": "object",
                    "properties": {
                        "op": {"type": "string", "enum": ["delete", "shift"]},
                        "idx": {"type": "integer"},
                        "start": {"type": "integer"},
                        "end": {"type": "integer"},
                    },
                    "required": ["op", "idx"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["note_id", "edits"],
        "additionalProperties": False,
    }
    return json.dumps(schema)


def json_schema_delete_only(*, max_edits: int = 10) -> str:
    """JSON schema for constrained decoding (delete-only scripts)."""
    schema = {
        "type": "object",
        "properties": {
            "note_id": {"type": "string"},
            "edits": {
                "type": "array",
                "maxItems": max(0, int(max_edits)),
                "items": {
                    "type": "object",
                    "properties": {
                        "op": {"type": "string", "enum": ["delete"]},
                        "idx": {"type": "integer"},
                    },
                    "required": ["op", "idx"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["note_id", "edits"],
        "additionalProperties": False,
    }
    return json.dumps(schema)


def spans_from_pred_rows(rows: Iterable[tuple[int, int, int]]) -> list[Span]:
    spans: list[Span] = []
    for i, (start, end, cid) in enumerate(rows):
        spans.append(Span(idx=i, start=int(start), end=int(end), concept_id=int(cid)))
    return spans
