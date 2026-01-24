#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from iou_metric import score_macro_iou


ROOT = Path(__file__).resolve().parents[1]
SECOND_PLACE = ROOT / "2nd Place"


def link_or_copy(src: Path, dest: Path) -> None:
    if dest.is_symlink() and not dest.exists():
        dest.unlink()
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(src, dest, target_is_directory=src.is_dir())
    except FileExistsError:
        if dest.is_symlink():
            dest.unlink()
            os.symlink(src, dest, target_is_directory=src.is_dir())
    except OSError:
        if dest.exists() or dest.is_symlink():
            try:
                if dest.is_dir() and not dest.is_symlink():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            except OSError:
                pass
        if src.is_dir():
            shutil.copytree(src, dest)
        else:
            shutil.copyfile(src, dest)


def resolve_existing_path(*candidates: Path) -> Path | None:
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    return None


def find_snomed_release_dirs(data_dir: Path) -> list[Path]:
    matches: list[Path] = []
    for candidate in data_dir.glob("SnomedCT_*"):
        if (candidate / "Snapshot" / "Terminology").exists():
            matches.append(candidate)
    return matches


def try_symlink_dir(src: Path, dest: Path) -> None:
    if dest.exists() or dest.is_symlink():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(src, dest, target_is_directory=True)
    except OSError:
        return


def sync_second_place_data(data_dir: Path) -> None:
    data_root = SECOND_PLACE / "data"
    competition_dir = data_root / "competition_data"

    notes = resolve_existing_path(
        data_dir / "mimic-iv_notes_training_set.csv",
        data_dir / "train_notes.csv",
    )
    annotations = resolve_existing_path(data_dir / "train_annotations.csv")
    if notes is None or annotations is None:
        raise FileNotFoundError(
            "Missing training data. Expected mimic-iv_notes_training_set.csv (or train_notes.csv) "
            "and train_annotations.csv under --data-dir."
        )

    link_or_copy(notes, competition_dir / "mimic-iv_notes_training_set.csv")
    link_or_copy(annotations, competition_dir / "train_annotations.csv")
    # Prefer lightweight symlinks for SNOMED RF2 releases (they're large). If symlinks
    # are not permitted (common on Windows/UNC paths), they are not required for
    # inference/scoring.
    for release_dir in find_snomed_release_dirs(data_dir):
        try_symlink_dir(release_dir, competition_dir / release_dir.name)


def run_cmd(cmd: list[str], cwd: Path) -> None:
    printable = " ".join(cmd)
    print(f"[run] {printable} (cwd={cwd})")
    proc = subprocess.run(cmd, cwd=str(cwd), check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {printable}")


def cuda_sanity_check() -> None:
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "PyTorch is required. Install dependencies first (see `install_requirements.sh`)."
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in this Python environment. "
            "Install a CUDA-enabled PyTorch build and ensure NVIDIA drivers are set up."
        )

    try:
        x = torch.randn(8, device="cuda")
        (x * 2).sum().item()
    except Exception as exc:
        name = torch.cuda.get_device_name(0) if torch.cuda.device_count() else "unknown"
        cap = (
            ".".join(str(v) for v in torch.cuda.get_device_capability(0))
            if torch.cuda.device_count()
            else "unknown"
        )
        archs = getattr(torch.cuda, "get_arch_list", lambda: [])()
        raise RuntimeError(
            "CUDA is visible but cannot run kernels on this GPU.\n"
            f"- torch: {torch.__version__}\n"
            f"- gpu: {name}\n"
            f"- capability: {cap}\n"
            f"- torch arch list: {archs}\n\n"
            "If you have an RTX 5090 (sm_120), you likely need a newer CUDA wheel. "
            "On Windows, for example:\n"
            "  python -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu126 torch torchvision torchaudio\n"
        ) from exc


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run 2nd place (SNOBERT) pipeline and score outputs with the competition macro-IoU metric."
    )
    parser.add_argument(
        "--data-dir",
        default=str(ROOT / "data"),
        help="Base data directory containing train_notes/train_annotations (default: ./data)",
    )
    parser.add_argument(
        "--notes",
        default=None,
        help="Notes CSV to run as test set (default: cutmed_notes.csv if available, else train_notes.csv).",
    )
    parser.add_argument(
        "--annotations",
        default=None,
        help="Gold annotations CSV for scoring (default: cutmed_fixed_train_annotations.csv if available, else train_annotations.csv).",
    )
    parser.add_argument(
        "--ensure-preprocess",
        action="store_true",
        help="Run 2nd Place preprocessing if cutmed files are missing.",
    )
    parser.add_argument(
        "--skip-cuda-check",
        action="store_true",
        help="Skip the early CUDA kernel sanity-check.",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "outputs" / "2nd_place_train_eval.json"),
        help="Where to write a JSON report (default: outputs/2nd_place_train_eval.json)",
    )
    parser.add_argument(
        "--copy-submission",
        action="store_true",
        help="Copy the produced submission.csv into ./outputs as submission_2nd_place.csv",
    )
    args = parser.parse_args()

    if not args.skip_cuda_check:
        cuda_sanity_check()

    sync_second_place_data(Path(args.data_dir))

    competition_dir = SECOND_PLACE / "data" / "competition_data"
    cutmed_notes = competition_dir / "cutmed_notes.csv"
    cutmed_ann = competition_dir / "cutmed_fixed_train_annotations.csv"

    if args.ensure_preprocess and not (cutmed_notes.exists() and cutmed_ann.exists()):
        env = os.environ.copy()
        env.setdefault("TOKENIZERS_PARALLELISM", "true")
        printable = f"{sys.executable} src/preprocess.py"
        print(f"[run] {printable} (cwd={SECOND_PLACE})")
        proc = subprocess.run(
            [sys.executable, "src/preprocess.py"],
            cwd=str(SECOND_PLACE),
            env=env,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Preprocess failed ({proc.returncode})")

    notes_path = Path(args.notes) if args.notes else (cutmed_notes if cutmed_notes.exists() else Path(args.data_dir) / "train_notes.csv")
    annotations_path: Optional[Path]
    if args.annotations:
        annotations_path = Path(args.annotations)
    else:
        annotations_path = cutmed_ann if cutmed_ann.exists() else Path(args.data_dir) / "train_annotations.csv"

    if not notes_path.exists():
        raise FileNotFoundError(f"Notes not found: {notes_path}")
    if annotations_path and not annotations_path.exists():
        raise FileNotFoundError(f"Annotations not found: {annotations_path}")

    required = [
        SECOND_PLACE / "data" / "preprocess_data" / "most_common_concept.pkl",
        SECOND_PLACE / "data" / "second_stage" / "sapbert",
        SECOND_PLACE / "data" / "first_stage",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required 2nd-place assets:\n  - "
            + "\n  - ".join(str(p) for p in missing)
            + "\n\nRun `python scripts/train_2nd_place.py` first."
        )

    test_notes_dst = competition_dir / "test_notes.csv"
    test_notes_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(notes_path, test_notes_dst)

    run_cmd([sys.executable, "submission/main.py"], cwd=SECOND_PLACE)

    pred_path = SECOND_PLACE / "submission.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Expected output not found: {pred_path}")

    ROOT.joinpath("outputs").mkdir(parents=True, exist_ok=True)
    copied_pred_path = ROOT / "outputs" / "submission_2nd_place.csv"
    if args.copy_submission:
        shutil.copyfile(pred_path, copied_pred_path)

    pred_df = pd.read_csv(pred_path)
    gold_df = pd.read_csv(annotations_path) if annotations_path else None

    result = {
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        "notes_path": str(notes_path),
        "annotations_path": str(annotations_path) if annotations_path else None,
        "submission_csv": str(copied_pred_path) if args.copy_submission else str(pred_path),
        "n_notes": int(len(pd.read_csv(notes_path))),
        "n_predictions": int(len(pred_df)),
    }

    if gold_df is not None:
        score = score_macro_iou(pred_df, gold_df)
        result.update({k: v for k, v in score.items() if k != "per_class_iou"})
        result["per_class_iou"] = {str(k): v for k, v in score["per_class_iou"].items()}
    else:
        result["macro_iou"] = None
        result["n_classes"] = None

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"Wrote: {out_path}")
    if result.get("macro_iou") is not None:
        print(f"macro IoU: {result['macro_iou']:.4f} over {result['n_classes']} classes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
