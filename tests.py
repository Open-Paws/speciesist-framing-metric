import unittest, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from speciesist_framing import _score_text, _detect_species_mentions, SPECIESIST_TERMS, NON_SPECIESIST_TERMS

class TestScoring(unittest.TestCase):
    def test_speciesist_high(self):
        s, t = _score_text("The livestock were harvested and processed.", SPECIESIST_TERMS)
        self.assertGreater(s, 0)
        self.assertTrue(any(x["term"] == "livestock" for x in t))
    def test_non_speciesist_high(self):
        s, t = _score_text("The sentient beings were killed, their suffering documented.", NON_SPECIESIST_TERMS)
        self.assertGreater(s, 0)
    def test_neutral_zero(self):
        s1, _ = _score_text("The weather is sunny.", SPECIESIST_TERMS)
        s2, _ = _score_text("The weather is sunny.", NON_SPECIESIST_TERMS)
        self.assertEqual(s1, 0); self.assertEqual(s2, 0)
    def test_industry_ratio_high(self):
        spec, _ = _score_text("Livestock inventory processed, throughput increased.", SPECIESIST_TERMS)
        anti, _ = _score_text("Livestock inventory processed, throughput increased.", NON_SPECIESIST_TERMS)
        self.assertGreater(spec / (spec + anti) if (spec+anti) > 0 else 0, 0.7)
    def test_advocacy_ratio_low(self):
        spec, _ = _score_text("Animal liberation requires recognizing sentient beings suffering from speciesism.", SPECIESIST_TERMS)
        anti, _ = _score_text("Animal liberation requires recognizing sentient beings suffering from speciesism.", NON_SPECIESIST_TERMS)
        self.assertLess(spec / (spec + anti) if (spec+anti) > 0 else 1, 0.3)
    def test_companion_detection(self):
        c, f = _detect_species_mentions("The dog and cat played.")
        self.assertIn("dog", c); self.assertEqual(len(f), 0)
    def test_farmed_detection(self):
        c, f = _detect_species_mentions("The pigs and chickens were confined.")
        self.assertEqual(len(c), 0); self.assertIn("pigs", f)
    def test_word_boundary(self):
        _, t = _score_text("The player scored.", SPECIESIST_TERMS)
        self.assertFalse(any(x["term"] == "layer" for x in t))

if __name__ == "__main__": unittest.main()
