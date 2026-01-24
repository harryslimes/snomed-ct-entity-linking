#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


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
    # are not permitted (common on Windows/UNC paths), `src/preprocess.py` can still
    # discover releases from the parent repo `./data` directory.
    for release_dir in find_snomed_release_dirs(data_dir):
        try_symlink_dir(release_dir, competition_dir / release_dir.name)


def parse_score_from_name(name: str) -> float | None:
    match = re.search(r"_score_([0-9]+(?:\\.[0-9]+)?)$", name)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def select_best_checkpoint(checkpoints: list[Path]) -> Path:
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoints found under {SECOND_PLACE / 'output'}. "
            "Expected folders containing fc.pth."
        )
    best = None
    best_key = None
    for ckpt in checkpoints:
        score = parse_score_from_name(ckpt.name)
        mtime = ckpt.stat().st_mtime
        key = (score if score is not None else -1.0, mtime)
        if best is None or key > best_key:
            best = ckpt
            best_key = key
    assert best is not None
    return best


def export_checkpoint_to_first_stage(src_ckpt: Path, force: bool) -> Path:
    dest_root = SECOND_PLACE / "data" / "first_stage"
    dest_root.mkdir(parents=True, exist_ok=True)
    dest = dest_root / src_ckpt.name
    if dest.exists():
        if not force:
            return dest
        if dest.is_dir() and not dest.is_symlink():
            shutil.rmtree(dest)
        else:
            dest.unlink()
    link_or_copy(src_ckpt, dest)
    return dest


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
        description="Train SNOBERT (2nd place) first-stage NER model and export a checkpoint for inference."
    )
    parser.add_argument(
        "--data-dir",
        default=str(ROOT / "data"),
        help="Base data directory containing training notes/annotations (default: ./data)",
    )
    parser.add_argument(
        "--preprocess-only",
        action="store_true",
        help="Only run preprocessing (cuts headers, builds SNOMED graph, embeddings, static dict).",
    )
    parser.add_argument(
        "--preprocess-cpu",
        action="store_true",
        help="Run preprocessing embedding generation on CPU (very slow).",
    )
    parser.add_argument(
        "--skip-cuda-check",
        action="store_true",
        help="Skip the early CUDA kernel sanity-check.",
    )
    parser.add_argument("--split", default="0", help="Fold index (0-3) or 'all' (default: 0)")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs (default: 5)")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch_size")
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=None,
        help="Override gradient_accumulation_steps",
    )
    parser.add_argument("--model", default=None, help="Override base HF model name/path")
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=1,
        help="Number of processes / GPUs to use (default: 1)",
    )
    parser.add_argument(
        "--ddp",
        action="store_true",
        help="Enable DDP (requires --nproc-per-node > 1 or torchrun usage).",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Do not copy/symlink the best checkpoint into 2nd Place/data/first_stage.",
    )
    parser.add_argument(
        "--force-export",
        action="store_true",
        help="Overwrite an existing exported checkpoint directory if present.",
    )
    args = parser.parse_args()

    if args.preprocess_cpu and not args.preprocess_only:
        raise ValueError("--preprocess-cpu is only supported with --preprocess-only.")

    if not args.skip_cuda_check and not args.preprocess_cpu:
        cuda_sanity_check()

    sync_second_place_data(Path(args.data_dir))

    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "true")

    print("[step] preprocess")
    preprocess_cmd = [sys.executable, "src/preprocess.py"]
    if args.preprocess_cpu:
        preprocess_cmd.append("--cpu")
    proc = subprocess.run(
        preprocess_cmd,
        cwd=str(SECOND_PLACE),
        env=env,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Preprocess failed ({proc.returncode})")
    if args.preprocess_only:
        return 0

    overrides = [f"split={args.split}", f"epochs={args.epochs}"]
    if args.batch_size is not None:
        overrides.append(f"batch_size={args.batch_size}")
    if args.grad_accum is not None:
        overrides.append(f"gradient_accumulation_steps={args.grad_accum}")
    if args.model is not None:
        overrides.append(f"model={args.model}")
    overrides.append(f"PARALLEL.DDP={'true' if args.ddp else 'false'}")

    start_time = time.time()
    print("[step] train")
    if args.nproc_per_node > 1 or args.ddp:
        torchrun = shutil.which("torchrun")
        if not torchrun:
            raise FileNotFoundError("torchrun not found on PATH (required for multi-GPU).")
        cmd = ["torchrun", f"--nproc-per-node={args.nproc_per_node}", "src/main.py", *overrides]
        proc = subprocess.run(cmd, cwd=str(SECOND_PLACE), env=env, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"Training failed ({proc.returncode})")
    else:
        cmd = [sys.executable, "src/main.py", *overrides]
        proc = subprocess.run(cmd, cwd=str(SECOND_PLACE), env=env, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"Training failed ({proc.returncode})")

    if args.no_export:
        return 0

    candidates = []
    for fc_path in (SECOND_PLACE / "output").rglob("fc.pth"):
        if fc_path.stat().st_mtime >= start_time - 5:
            candidates.append(fc_path.parent)
    if not candidates:
        for fc_path in (SECOND_PLACE / "output").rglob("fc.pth"):
            candidates.append(fc_path.parent)
    best = select_best_checkpoint(sorted(set(candidates)))
    exported = export_checkpoint_to_first_stage(best, force=args.force_export)
    print(f"[done] exported checkpoint: {exported}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
