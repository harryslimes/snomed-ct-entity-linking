# UMLS Metathesaurus download (MRCONSO.RRF)

UMLS files are **licensed** by the U.S. National Library of Medicine (NLM). This repo **does not** include UMLS data and you should not commit it.

## Prereqs

1. Create/verify an NLM UTS account and accept the UMLS license for your institution.
2. Create a UTS API key (in your UTS profile).

## Download MRCONSO.RRF (recommended)

This project primarily needs the Metathesaurus `MRCONSO.RRF` table to extract synonyms (including CHV terms).

```bash
export UMLS_API_KEY="YOUR_UTS_API_KEY"
python scripts/umls/download_umls.py --release 2025AB --artifact mrconso --extract
```

Outputs:

- Zip: `data/umls/2025AB/umls-2025AB-mrconso.zip`
- Extracted: `data/umls/2025AB/**/MRCONSO.RRF`

## Other artifacts (optional)

If you later need more Metathesaurus tables:

```bash
python scripts/umls/download_umls.py --release 2025AB --artifact metathesaurus-full --extract
```

## Notes

- If you’re using a different UMLS release (e.g. `2026AA`), change `--release` accordingly.
- If your download is denied, double-check that your UTS account has an accepted UMLS license and that the API key is valid.

