# Dictionary-Augmented Retrieval Findings

**Date**: 2026-02-22
**Sample**: 2,000 non-abbreviation annotations (train split, seed=42)

## Summary

Tested whether looking up LLM-generated search terms in the SNOMED dictionary (MRCONSO + terminology CSV, 823k terms) and adding matched concepts to the hybrid index candidate list improves gold retrieval recall.

## Results

| Method | Recall | Count |
|--------|--------|-------|
| Hybrid index only | 79.8% | 1,595 |
| Dict of LLM terms only | 58.8% | 1,176 |
| **Hybrid + Dict combined** | **81.8%** | **1,637** |
| Dict rescued (not in hybrid) | +2.1% | +42 |
| Neither | 18.1% | 363 |

## Dictionary-only analysis (raw span vs LLM terms)

| Method | Gold hits | Rate |
|--------|-----------|------|
| Span-only dict | 862 | 43.1% |
| LLM-terms dict (no rules) | 1,155 | 57.8% |
| LLM-terms dict (23 rules) | 1,175 | 58.8% |
| Combined oracle (span + LLM) | 1,211 | 60.5% |

- LLM terms rescue ~17% of annotations the raw span misses in the dictionary
- Rules add ~1pp on top of that
- Span text is included in LLM terms ~86% of the time (not guaranteed)

## Venn diagram: Dictionary vs Hybrid (full 41,597 annotations, span-as-query)

| Category | Count | Rate |
|----------|-------|------|
| Both dict + hybrid | 17,255 | 41.5% |
| Dict only | 453 | 1.1% |
| Hybrid only | 14,338 | 34.5% |
| Neither | 9,551 | 23.0% |
| Oracle (either) | 32,046 | 77.0% |

## Example rescues (dict finds gold but hybrid doesn't)

- `"fall"` -> gold "Falls (finding)" -- exact dict match, not in hybrid top-10
- `"murmurs"` -> gold "Murmur (finding)" -- plural form
- `"lesions"` -> gold "Lesion (morphologic abnormality)" -- generic concept
- `"GLUCOSE"` -> via LLM term `"blood glucose"` -> gold "Glucose measurement, blood"
- `"drainage"` -> via LLM term `"wound drainage"` -> gold "Wound discharge (finding)"

## Conclusion

Adding dictionary lookup of LLM-generated terms to the hybrid candidate list gives a +2.1pp retrieval recall boost (79.8% -> 81.8%) at essentially zero cost (no extra LLM or index calls). Extrapolates to ~870 additional correct retrievals across the full dataset.

Implementation would be straightforward: after the LLM generates search terms and before the select step, look up each term in the dictionary and merge any matched concept IDs into the candidate list.

**Decision**: Deferred. Focus on rule generation loop first, revisit if retrieval ceiling becomes the bottleneck.

## Scripts

- `scripts/test_dict_nonabbrev.py` -- raw span dictionary lookup + hybrid fallback
- `scripts/test_dict_llm_terms.py` -- LLM-term dictionary lookup analysis
- `scripts/test_dict_augmented_retrieval.py` -- combined hybrid + dict retrieval test
