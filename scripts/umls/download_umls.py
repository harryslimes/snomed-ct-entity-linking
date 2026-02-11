#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import sys
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Artifact:
    name: str
    filename_suffix: str

    @property
    def label(self) -> str:
        return self.name


ARTIFACTS: dict[str, Artifact] = {
    "mrconso": Artifact(name="mrconso", filename_suffix="mrconso"),
    "metathesaurus-full": Artifact(name="metathesaurus-full", filename_suffix="metathesaurus-full"),
    "full": Artifact(name="full", filename_suffix="full"),
}


def _resolve_api_key(cli_value: str | None) -> str | None:
    if cli_value:
        return cli_value
    return (
        os.environ.get("UMLS_API_KEY")
        or os.environ.get("UTS_API_KEY")
        or os.environ.get("UMLS_UTS_API_KEY")
    )


def _download_url(release: str, artifact: Artifact) -> str:
    filename = f"umls-{release}-{artifact.filename_suffix}.zip"
    return f"https://download.nlm.nih.gov/umls/kss/{release}/{filename}"


def _uts_download_api_url(download_url: str, api_key: str) -> str:
    encoded_download_url = urllib.parse.quote(download_url, safe="")
    encoded_api_key = urllib.parse.quote(api_key, safe="")
    return f"https://uts-ws.nlm.nih.gov/download?url={encoded_download_url}&apiKey={encoded_api_key}"


def _download_file(url: str, dest_path: Path, *, overwrite: bool) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {dest_path}")

    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink()

    req = urllib.request.Request(url, headers={"User-Agent": "snomed-ct-entity-linking/umls-downloader"})
    with urllib.request.urlopen(req) as resp, tmp_path.open("wb") as out_fp:
        shutil.copyfileobj(resp, out_fp)

    tmp_path.replace(dest_path)


def _extract_zip(zip_path: Path, extract_dir: Path, *, overwrite: bool) -> list[Path]:
    extract_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            target_path = extract_dir / member.filename
            if target_path.exists() and not overwrite:
                continue
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member, "r") as src, target_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted.append(target_path)
    return extracted


def _find_first(path_root: Path, filename: str) -> Path | None:
    for path in path_root.rglob(filename):
        if path.is_file():
            return path
    return None


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download UMLS Knowledge Sources artifacts via NLM UTS download API. "
            "Requires that you have accepted the UMLS license and have a valid UTS API key."
        )
    )
    parser.add_argument("--release", required=True, help="UMLS release, e.g. 2025AB")
    parser.add_argument(
        "--artifact",
        required=True,
        choices=sorted(ARTIFACTS.keys()),
        help="Which UMLS artifact to download.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="UTS API key. If omitted, uses env var UMLS_API_KEY (or UTS_API_KEY).",
    )
    parser.add_argument(
        "--out-dir",
        default="data/umls",
        help="Base output directory (default: data/umls).",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract downloaded zip into out-dir/<release>/ (keeps paths inside zip).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing zip (and extracted files when --extract is set).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print URLs and paths without downloading.",
    )
    args = parser.parse_args(argv)

    api_key = _resolve_api_key(args.api_key)
    if not api_key:
        print(
            "Missing UTS API key. Set UMLS_API_KEY (recommended) or pass --api-key.",
            file=sys.stderr,
        )
        return 2

    artifact = ARTIFACTS[args.artifact]
    download_url = _download_url(args.release, artifact)
    api_url = _uts_download_api_url(download_url, api_key)

    out_dir = Path(args.out_dir)
    zip_path = out_dir / args.release / Path(download_url).name

    print(f"Release:   {args.release}")
    print(f"Artifact:  {artifact.label}")
    print(f"Download:  {download_url}")
    print(f"UTS API:   {api_url}")
    print(f"Zip path:  {zip_path}")

    if args.dry_run:
        return 0

    _download_file(api_url, zip_path, overwrite=args.overwrite)
    print(f"Downloaded: {zip_path}")

    if args.extract:
        extract_dir = out_dir / args.release
        extracted = _extract_zip(zip_path, extract_dir, overwrite=args.overwrite)
        print(f"Extracted:  {len(extracted)} file(s) into {extract_dir}")
        mrconso_path = _find_first(extract_dir, "MRCONSO.RRF")
        if mrconso_path:
            print(f"Found:      {mrconso_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

