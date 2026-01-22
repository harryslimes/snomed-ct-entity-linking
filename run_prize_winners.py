#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run_cmd(cmd, cwd):
    printable = " ".join(cmd)
    print(f"[run] {printable} (cwd={cwd})")
    return subprocess.run(cmd, cwd=cwd, check=False).returncode == 0


def link_or_copy(src, dest):
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
        else:
            return
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dest)
        else:
            shutil.copyfile(src, dest)


def resolve_existing_path(*candidates):
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    return None


def resolve_training_notes(data_dir):
    return resolve_existing_path(
        data_dir / "mimic-iv_notes_training_set.csv",
        data_dir / "train_notes.csv",
    )


def resolve_training_annotations(data_dir):
    return resolve_existing_path(data_dir / "train_annotations.csv")


def ensure_training_aliases(data_dir):
    notes = resolve_training_notes(data_dir)
    if notes and notes.name != "mimic-iv_notes_training_set.csv":
        link_or_copy(notes, data_dir / "mimic-iv_notes_training_set.csv")


def sync_named_paths(data_dir, target_dir, names):
    for name in names:
        src = data_dir / name
        if src.exists():
            link_or_copy(src, target_dir / name)


def find_snomed_release_dirs(data_dir):
    matches = []
    for candidate in data_dir.glob("SnomedCT_*"):
        if (candidate / "Snapshot" / "Terminology").exists():
            matches.append(candidate)
    return matches


def sync_first_place_data(data_dir, target_dir):
    if not data_dir.exists():
        return
    for subdir in ("raw", "interim"):
        src = data_dir / subdir
        dest = target_dir / subdir
        if src.exists() and not dest.exists():
            link_or_copy(src, dest)
    notes = resolve_training_notes(data_dir)
    if notes:
        link_or_copy(notes, target_dir / "raw" / "mimic-iv_notes_training_set.csv")
    annotations = resolve_training_annotations(data_dir)
    if annotations:
        link_or_copy(annotations, target_dir / "raw" / "train_annotations.csv")
    sync_named_paths(
        data_dir,
        target_dir / "raw",
        [
            "athena",
            "discharge.csv.gz",
            "medical_abbreviations.csv",
        ],
    )
    for release_dir in find_snomed_release_dirs(data_dir):
        link_or_copy(release_dir, target_dir / "raw" / release_dir.name)


def sync_second_place_data(data_dir, target_dir):
    if not data_dir.exists():
        return
    for subdir in ("competition_data", "first_stage", "second_stage", "preprocess_data"):
        src = data_dir / subdir
        dest = target_dir / subdir
        if src.exists() and not dest.exists():
            link_or_copy(src, dest)
    notes = resolve_training_notes(data_dir)
    if notes:
        link_or_copy(notes, target_dir / "competition_data" / "mimic-iv_notes_training_set.csv")
    annotations = resolve_training_annotations(data_dir)
    if annotations:
        link_or_copy(annotations, target_dir / "competition_data" / "train_annotations.csv")
    sync_named_paths(
        data_dir,
        target_dir / "competition_data",
        [],
    )
    for release_dir in find_snomed_release_dirs(data_dir):
        link_or_copy(release_dir, target_dir / "competition_data" / release_dir.name)


def sync_third_place_data(data_dir, target_dir):
    if not data_dir.exists():
        return
    if target_dir.is_symlink() and not target_dir.exists():
        target_dir.unlink()
    if data_dir.exists() and not target_dir.exists():
        link_or_copy(data_dir, target_dir)


def find_test_notes(data_dir, explicit_path):
    if explicit_path:
        path = Path(explicit_path)
        if path.exists():
            return path
        raise FileNotFoundError(f"Test notes file not found: {path}")

    candidates = [
        data_dir / "test_notes.csv",
        data_dir / "competition_data" / "test_notes.csv",
        data_dir / "raw" / "test_notes.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Unable to find test notes. Tried: "
        + ", ".join(str(c) for c in candidates)
    )


def ensure_parent(path):
    path.parent.mkdir(parents=True, exist_ok=True)


def copy_test_notes(src, dest):
    ensure_parent(dest)
    shutil.copyfile(src, dest)


def has_files(path):
    return path.exists() and any(path.iterdir())


def run_first_place(data_dir, test_notes, output_dir):
    base = ROOT / "1st Place"
    submission_dir = base / "submission"
    data_root = base / "data"

    sync_first_place_data(data_dir, data_root)

    interim_needed = [
        data_root / "interim" / "train_annotations_cln.csv",
        data_root / "interim" / "abbr_dict.pkl",
        data_root / "interim" / "term_extension.csv",
    ]
    raw_needed = [
        data_root / "raw" / "mimic-iv_notes_training_set.csv",
    ]

    if not submission_dir.exists() or not (submission_dir / "main.py").exists():
        if shutil.which("make"):
            ok = run_cmd(["make", "submission/main.py"], cwd=base)
            if not ok:
                print("[skip] 1st place: failed to build inference environment")
                return False
        else:
            if not all(p.exists() for p in interim_needed):
                print(
                    "[skip] 1st place: missing interim assets and make is unavailable; "
                    "cannot build submission environment"
                )
                return False
            ok = run_cmd(["python", "src/make_inference_env.py", "submission"], cwd=base)
            if not ok:
                print("[skip] 1st place: failed to build inference environment")
                return False

    missing = [p for p in raw_needed + interim_needed if not p.exists()]
    if missing:
        print("[skip] 1st place: missing required data files:")
        for path in missing:
            print(f"  - {path}")
        return False

    copy_test_notes(test_notes, submission_dir / "data" / "test_notes.csv")
    ok = run_cmd(["python", "main.py"], cwd=submission_dir)
    if not ok:
        print("[skip] 1st place: inference failed")
        return False

    submission_path = submission_dir / "submission.csv"
    if submission_path.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(submission_path, output_dir / "submission_1st_place.csv")
    return True


def run_second_place(data_dir, test_notes, output_dir):
    base = ROOT / "2nd Place"
    data_root = base / "data"
    competition_dir = data_root / "competition_data"
    first_stage_dir = data_root / "first_stage"
    second_stage_dir = data_root / "second_stage" / "sapbert"
    static_dict = data_root / "preprocess_data" / "most_common_concept.pkl"

    sync_second_place_data(data_dir, data_root)

    required = [
        competition_dir / "cutmed_notes.csv",
        competition_dir / "cutmed_fixed_train_annotations.csv",
        static_dict,
    ]
    missing = [p for p in required if not p.exists()]
    if missing or not has_files(first_stage_dir) or not second_stage_dir.exists():
        print("[skip] 2nd place: missing required assets:")
        for path in missing:
            print(f"  - {path}")
        if not has_files(first_stage_dir):
            print(f"  - {first_stage_dir} (missing checkpoints)")
        if not second_stage_dir.exists():
            print(f"  - {second_stage_dir}")
        return False

    copy_test_notes(test_notes, competition_dir / "test_notes.csv")
    ok = run_cmd(["python", "submission/main.py"], cwd=base)
    if not ok:
        print("[skip] 2nd place: inference failed")
        return False

    submission_path = base / "submission.csv"
    if submission_path.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(submission_path, output_dir / "submission_2nd_place.csv")
    return True


def run_third_place(data_dir, test_notes, output_dir):
    base = ROOT / "3rd Place"
    assets_dir = base / "assets"
    models_dir = base / "models"

    sync_third_place_data(data_dir, base / "data")

    required = [
        assets_dir / "faiss_index_constitution_all-MiniLM-L12-v2_finetuned",
        assets_dir / "newdict_snomed_extended.txt",
        models_dir / "Mistral-7B-Instruct-v0.2-AddAnnotations-lora-v0.4",
        models_dir / "Mistral-7B-Instruct-v0.2-AddAnnotations-lora-v0.6",
        models_dir / "Mistral-7B-Instruct-v0.2-Pescu-faiss-clasify-lora_2",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        print("[skip] 3rd place: missing required assets:")
        for path in missing:
            print(f"  - {path}")
        return False

    copy_test_notes(test_notes, base / "data" / "test_notes.csv")
    ok = run_cmd(["python", "main.py"], cwd=base)
    if not ok:
        print("[skip] 3rd place: inference failed")
        return False

    submission_path = base / "submission.csv"
    if submission_path.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(submission_path, output_dir / "submission_3rd_place.csv")
    return True


def parse_winners(value):
    value = value.lower()
    if value == "all":
        return ["1st", "2nd", "3rd"]
    allowed = {"1st", "2nd", "3rd"}
    selected = [v.strip() for v in value.split(",") if v.strip()]
    if not selected or any(v not in allowed for v in selected):
        raise ValueError("winners must be 'all' or a comma list of 1st,2nd,3rd")
    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Run inference for all prize-winning solutions."
    )
    parser.add_argument(
        "--winners",
        default="all",
        help="Which winners to run: all or comma-separated list of 1st,2nd,3rd",
    )
    parser.add_argument(
        "--data-dir",
        default=str(ROOT / "data"),
        help="Base data directory (default: ./data)",
    )
    parser.add_argument(
        "--test-notes",
        default=None,
        help="Path to test_notes.csv (overrides auto-detection)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs"),
        help="Directory to store submissions (default: ./outputs)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    winners = parse_winners(args.winners)
    ensure_training_aliases(data_dir)

    try:
        test_notes = find_test_notes(data_dir, args.test_notes)
    except FileNotFoundError as exc:
        print(f"[error] {exc}")
        return 1

    results = {}
    if "1st" in winners:
        results["1st"] = run_first_place(data_dir, test_notes, output_dir)
    if "2nd" in winners:
        results["2nd"] = run_second_place(data_dir, test_notes, output_dir)
    if "3rd" in winners:
        results["3rd"] = run_third_place(data_dir, test_notes, output_dir)

    failures = [k for k, v in results.items() if not v]
    if failures:
        print("[done] completed with skips/failures:", ", ".join(failures))
        return 1

    print("[done] all requested winners finished successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
