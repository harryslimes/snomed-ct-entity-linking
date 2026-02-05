import os
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KIRI_SRC = ROOT / "1st Place" / "src"
if str(KIRI_SRC) not in sys.path:
    sys.path.append(str(KIRI_SRC))


class TestKiriStopwordTransparency(unittest.TestCase):
    def test_stopword_transparent_pattern_matches(self):
        from mimic_common import get_pattern, pattern_cache  # noqa: WPS433

        old_mode = os.environ.get("KIRI_STOPWORD_TRANSPARENT")
        old_min = os.environ.get("KIRI_STOPWORD_MIN_TOKENS")
        try:
            os.environ["KIRI_STOPWORD_TRANSPARENT"] = "1"
            os.environ["KIRI_STOPWORD_MIN_TOKENS"] = "2"
            pattern_cache.clear()

            p = get_pattern("fracture femur")
            self.assertIsNotNone(p)
            self.assertIsNotNone(p.search("fracture of the femur"))
        finally:
            pattern_cache.clear()
            if old_mode is None:
                os.environ.pop("KIRI_STOPWORD_TRANSPARENT", None)
            else:
                os.environ["KIRI_STOPWORD_TRANSPARENT"] = old_mode
            if old_min is None:
                os.environ.pop("KIRI_STOPWORD_MIN_TOKENS", None)
            else:
                os.environ["KIRI_STOPWORD_MIN_TOKENS"] = old_min

    def test_stopword_transparent_does_not_collapse_to_single_token(self):
        from mimic_common import get_pattern, pattern_cache  # noqa: WPS433

        old_mode = os.environ.get("KIRI_STOPWORD_TRANSPARENT")
        old_min = os.environ.get("KIRI_STOPWORD_MIN_TOKENS")
        try:
            os.environ["KIRI_STOPWORD_TRANSPARENT"] = "1"
            os.environ["KIRI_STOPWORD_MIN_TOKENS"] = "2"
            pattern_cache.clear()

            p = get_pattern("patient discharge")
            self.assertIsNotNone(p)
            self.assertIsNone(p.search("discharge"))
            self.assertIsNotNone(p.search("patient discharge"))
        finally:
            pattern_cache.clear()
            if old_mode is None:
                os.environ.pop("KIRI_STOPWORD_TRANSPARENT", None)
            else:
                os.environ["KIRI_STOPWORD_TRANSPARENT"] = old_mode
            if old_min is None:
                os.environ.pop("KIRI_STOPWORD_MIN_TOKENS", None)
            else:
                os.environ["KIRI_STOPWORD_MIN_TOKENS"] = old_min

    def test_stopword_transparent_does_not_drop_stopwords_in_mention(self):
        from mimic_common import get_pattern, pattern_cache  # noqa: WPS433

        old_mode = os.environ.get("KIRI_STOPWORD_TRANSPARENT")
        old_min = os.environ.get("KIRI_STOPWORD_MIN_TOKENS")
        try:
            os.environ["KIRI_STOPWORD_TRANSPARENT"] = "1"
            os.environ["KIRI_STOPWORD_MIN_TOKENS"] = "2"
            pattern_cache.clear()

            p = get_pattern("to the")
            self.assertIsNotNone(p)
            self.assertIsNone(p.search("to"))
            self.assertIsNotNone(p.search("to the"))
        finally:
            pattern_cache.clear()
            if old_mode is None:
                os.environ.pop("KIRI_STOPWORD_TRANSPARENT", None)
            else:
                os.environ["KIRI_STOPWORD_TRANSPARENT"] = old_mode
            if old_min is None:
                os.environ.pop("KIRI_STOPWORD_MIN_TOKENS", None)
            else:
                os.environ["KIRI_STOPWORD_MIN_TOKENS"] = old_min

    def test_stopword_transparent_respects_min_tokens_default(self):
        from mimic_common import get_pattern, pattern_cache  # noqa: WPS433

        old_mode = os.environ.get("KIRI_STOPWORD_TRANSPARENT")
        old_min = os.environ.get("KIRI_STOPWORD_MIN_TOKENS")
        old_allow2 = os.environ.get("KIRI_STOPWORD_ALLOW_2TOKENS")
        try:
            os.environ["KIRI_STOPWORD_TRANSPARENT"] = "1"
            os.environ.pop("KIRI_STOPWORD_MIN_TOKENS", None)  # default=3
            os.environ["KIRI_STOPWORD_ALLOW_2TOKENS"] = ""  # disable 2-token allowlist
            pattern_cache.clear()

            # With default min_tokens=3 and no allowlist, this should fall back to strict.
            p = get_pattern("fracture femur")
            self.assertIsNotNone(p)
            self.assertIsNone(p.search("fracture of the femur"))
        finally:
            pattern_cache.clear()
            if old_mode is None:
                os.environ.pop("KIRI_STOPWORD_TRANSPARENT", None)
            else:
                os.environ["KIRI_STOPWORD_TRANSPARENT"] = old_mode
            if old_min is None:
                os.environ.pop("KIRI_STOPWORD_MIN_TOKENS", None)
            else:
                os.environ["KIRI_STOPWORD_MIN_TOKENS"] = old_min
            if old_allow2 is None:
                os.environ.pop("KIRI_STOPWORD_ALLOW_2TOKENS", None)
            else:
                os.environ["KIRI_STOPWORD_ALLOW_2TOKENS"] = old_allow2


class TestKiriLinguisticVariants(unittest.TestCase):
    def test_linguistic_variants_add_fx(self):
        from mimic_train import add_linguistic_variants  # noqa: WPS433

        d = {("any", "fracture of femur"): 123}
        added = add_linguistic_variants(d, blacklist=[], max_variants_per_key=16)
        self.assertGreaterEqual(added, 1)
        self.assertIn(("any", "fx of femur"), d)
