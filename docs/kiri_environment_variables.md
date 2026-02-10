# KIRI Environment Variables

This document describes the environment variables that control KIRI pipeline behavior in `1st Place/src/mimic_train.py`.

## Dictionary Configuration

### `KIRI_SNOMED_MIN_LEN`
- **Default:** `2`
- **Type:** Integer
- **Description:** Minimum n-gram length for dictionary matching. Controls the shortest multi-word phrases that will be matched.
- **Impact:**
  - **Lower values (e.g., 1):** Matches single-word terms like "pain", "fever", "left". Higher recall but more false positives and overlap conflicts.
  - **Higher values (e.g., 3):** Only matches longer phrases. Lower recall, fewer false positives.
- **Example:**
  ```bash
  KIRI_SNOMED_MIN_LEN=1 python "1st Place/src/mimic_train.py"
  ```

### `KIRI_SNOMED_MAX_LEN`
- **Default:** `5`
- **Type:** Integer
- **Description:** Maximum n-gram length for dictionary matching. Controls the longest multi-word phrases that will be matched.
- **Impact:**
  - **Higher values (e.g., 7):** Matches longer medical phrases like "acute myocardial infarction type 1". Better specificity for complex terms but more computational cost and overlap conflicts.
  - **Lower values (e.g., 3):** Only matches shorter phrases. Misses detailed clinical terms.
- **Example:**
  ```bash
  KIRI_SNOMED_MAX_LEN=7 python "1st Place/src/mimic_train.py"
  ```
- **Note:** The original KIRI baseline used `min_len=1, max_len=7` (0.6025 macro char IoU).

### `KIRI_FLAT_TERMINOLOGY_PATH`
- **Default:** `"data/interim/flattened_terminology.csv"`
- **Type:** String (file path)
- **Description:** Path to the flattened SNOMED CT terminology CSV used for concept name lookups. Allows testing with expanded or alternative terminologies.
- **Use Cases:**
  - Test with UMLS-expanded synonyms
  - Use extended hierarchy with ancestor/descendant concepts
  - Compare against 3rd place solution's terminology
- **Example:**
  ```bash
  KIRI_FLAT_TERMINOLOGY_PATH="3rd Place/assets/dataflattened_terminology.csv" python "1st Place/src/mimic_train.py"
  ```

## Filtering Configuration

### `KIRI_SKIP_BAD_KEY_REMOVAL`
- **Default:** `False`
- **Type:** Boolean (0/1, true/false, yes/no)
- **Description:** When enabled, skips the `remove_bad_keys()` filtering step that removes dictionary terms with poor performance metrics.
- **Impact:**
  - **Enabled (skip filter):** Higher recall, more false positives. Recovers entities that would otherwise be filtered (e.g., common words like "iv", "left", "started" that appear 197+ times in gold annotations but are filtered as "bad keys").
  - **Disabled (apply filter):** Lower recall, fewer false positives. Standard behavior.
- **Performance Note:** From pipeline analysis, 12.8% of gold entities (4,208 spans) are never matched due to filtering. This includes 197 spans filtered by `remove_bad_keys()`.
- **Example:**
  ```bash
  KIRI_SKIP_BAD_KEY_REMOVAL=1 python "1st Place/src/mimic_train.py"
  ```

## Other Existing Environment Variables

The following environment variables were already present in the codebase:

- `KIRI_TRAIN_INDEX` - Enable/disable index-based matching (default: True)
- `KIRI_STOPWORD_TRANSPARENT` - Use unigram vs bigram prefilter (default: False)
- `KIRI_TRAIN_PARALLEL` / `KIRI_PARALLEL` - Enable parallel processing (default: False)
- `KIRI_TRAIN_PRECOMPILE` - Precompile regex patterns (default: True)
- `KIRI_TRAIN_LOG` - Enable logging during training (default: True)
- `KIRI_TRAIN_SAVE_DEBUG` - Save debug pickle files (default: True)
- `KIRI_TRAIN_SAVE_D_ALL` - Save full dictionary in debug output (default: True)
- `KIRI_LINGUISTIC_RULES` - Apply linguistic filtering rules (default: False)
- `KIRI_TRAIN_SAVE_FULL` - Save full model artifacts (default: True)

## Running Ablations

Combine multiple variables to test different configurations:

```bash
# Test with expanded dictionary and no bad key filtering
KIRI_SNOMED_MIN_LEN=1 \
KIRI_SNOMED_MAX_LEN=7 \
KIRI_SKIP_BAD_KEY_REMOVAL=1 \
python "1st Place/src/mimic_train.py"

# Test with 3rd place terminology
KIRI_FLAT_TERMINOLOGY_PATH="3rd Place/assets/dataflattened_terminology.csv" \
python "1st Place/src/mimic_train.py"

# Conservative configuration (fewer matches, higher precision)
KIRI_SNOMED_MIN_LEN=3 \
KIRI_SNOMED_MAX_LEN=5 \
python "1st Place/src/mimic_train.py"
```

## Performance Impact

From pipeline coverage analysis (`scripts/llm_postprocess/trace_pipeline_loss.py`):

| Configuration Aspect | Coverage Impact | Notes |
|---------------------|----------------|-------|
| Dictionary n-gram range | 74.4% entities (LC_RAW) | min_len=1, max_len=7 baseline |
| After overlap removal | 70.2% entities | Overlap resolution loses 4.2% |
| Bad key filtering | -12.8% entities | 4,208 spans never matched due to filters |
| Combined (LC+UC) | 87.2% entities | Before final merge |
| Final merged output | 82.6% entities | After all filtering |

**Key insight:** Adjusting these parameters trades off between recall (dictionary coverage) and precision (false positive rate). The deterministic FP filters (corpus lift, training FP-rate) provide better precision improvements than aggressive dictionary expansion.
