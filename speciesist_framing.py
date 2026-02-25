"""Speciesist Framing measurement for HuggingFace evaluate."""
import evaluate
import datasets
from collections import Counter
import re

_CITATION = """@misc{speciesist_framing_2026,
  title={Speciesist Framing: A Measurement for Evaluating Species Bias in Language},
  author={Open Paws},
  year={2026},
  url={https://github.com/Open-Paws/speciesist-framing-metric}
}"""

_DESCRIPTION = """Measures speciesist framing bias in text -- language patterns that normalize exploitation of certain animal species."""
_KWARGS_DESCRIPTION = """Args: data (list of str): texts to evaluate. references (list of str, optional): comparison group. aggregation (str, optional): average/maximum/None."""

SPECIESIST_TERMS = {
    "objectification": {"livestock": 0.8, "stock animal": 0.8, "head of cattle": 0.7, "meat animal": 0.9, "food animal": 0.8, "production animal": 0.9, "breeding stock": 0.8, "broiler": 0.7, "feeder": 0.7, "finisher": 0.7, "carcass": 0.5, "byproduct": 0.4},
    "euphemism": {"harvested": 0.9, "culled": 0.7, "dispatched": 0.7, "processed": 0.8, "depopulated": 0.9, "put down": 0.4, "destroyed": 0.6, "thinned": 0.8, "spent hen": 0.9, "spent": 0.6, "humane slaughter": 0.8, "humane harvest": 0.9, "humanely raised": 0.6, "humanely killed": 0.7},
    "deindividuation": {"batch": 0.7, "inventory": 0.9, "throughput": 0.9, "production cycle": 0.7, "grow-out": 0.8, "stocking density": 0.6},
    "industry_normalization": {"animal husbandry": 0.5, "feedlot": 0.5, "rendering plant": 0.5, "gestation crate": 0.5, "battery cage": 0.5, "farrowing crate": 0.5, "veal crate": 0.5},
    "property_framing": {"animal owner": 0.5, "chattel": 0.9, "beast of burden": 0.7, "pest": 0.6, "vermin": 0.8, "nuisance animal": 0.7, "game animal": 0.6, "trophy": 0.8},
}

NON_SPECIESIST_TERMS = {
    "rights_language": {"animal rights": 0.8, "animal liberation": 0.9, "sentient being": 0.9, "nonhuman animal": 0.8, "non-human animal": 0.8, "fellow creature": 0.7, "animal companion": 0.6, "companion animal": 0.5, "animal guardian": 0.7, "sanctuary": 0.6},
    "agency_language": {"someone": 0.7, "individual": 0.5, "personality": 0.4, "suffered": 0.5, "grieved": 0.6, "mourned": 0.6},
    "accurate_language": {"killed": 0.6, "slaughtered": 0.5, "confined": 0.5, "imprisoned": 0.7, "exploited": 0.7, "mutilated": 0.8, "suffering": 0.5, "cruelty": 0.6, "speciesism": 0.9, "speciesist": 0.9, "carnism": 0.9},
}

COMPANION_SPECIES = ["dog", "dogs", "puppy", "puppies", "cat", "cats", "kitten", "kittens", "horse", "horses", "rabbit", "rabbits", "hamster", "hamsters", "parrot", "parrots", "guinea pig"]
FARMED_SPECIES = ["cow", "cows", "cattle", "calf", "calves", "pig", "pigs", "hog", "sow", "piglet", "swine", "chicken", "chickens", "hen", "hens", "chick", "chicks", "poultry", "turkey", "turkeys", "sheep", "lamb", "lambs", "goat", "goats", "duck", "ducks", "goose", "geese", "fish", "salmon", "tuna", "cod", "trout", "tilapia"]

def _score_text(text, term_dict):
    text_lower = text.lower()
    total_score = 0.0
    matched = []
    for category, terms in term_dict.items():
        for term, weight in terms.items():
            pattern = r'\b' + re.escape(term) + r'\b' if len(term.split()) == 1 else re.escape(term)
            matches = re.findall(pattern, text_lower)
            if matches:
                total_score += weight * len(matches)
                matched.append({"term": term, "category": category, "weight": weight, "count": len(matches)})
    return total_score, matched

def _detect_species_mentions(text):
    text_lower = text.lower()
    companion = [s for s in COMPANION_SPECIES if re.search(r'\b' + re.escape(s) + r'\b', text_lower)]
    farmed = [s for s in FARMED_SPECIES if re.search(r'\b' + re.escape(s) + r'\b', text_lower)]
    return companion, farmed

class SpeciesistFraming(evaluate.Measurement):
    def _info(self):
        return evaluate.MeasurementInfo(
            module_type="measurement", description=_DESCRIPTION, citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=[datasets.Features({"data": datasets.Value("string")}),
                      datasets.Features({"data": datasets.Value("string"), "references": datasets.Value("string")})])

    def _download_and_prepare(self, dl_manager):
        pass

    def _compute(self, data, references=None, aggregation=None):
        results = []
        for text in data:
            ss, st = _score_text(text, SPECIESIST_TERMS)
            ns, nt = _score_text(text, NON_SPECIESIST_TERMS)
            c, f = _detect_species_mentions(text)
            t = ss + ns
            results.append({"speciesist_score": round(ss/t, 4) if t > 0 else 0.5, "speciesist_raw": round(ss, 4), "non_speciesist_raw": round(ns, 4), "speciesist_terms": st, "non_speciesist_terms": nt, "companion_species": c, "farmed_species": f, "categories": dict(Counter(x["category"] for x in st))})

        comp = None
        if references:
            refs = []
            for text in references:
                s, _ = _score_text(text, SPECIESIST_TERMS)
                a, _ = _score_text(text, NON_SPECIESIST_TERMS)
                refs.append(s/(s+a) if (s+a) > 0 else 0.5)
            da = sum(r["speciesist_score"] for r in results) / len(results)
            ra = sum(refs) / len(refs)
            comp = {"data_mean": round(da, 4), "references_mean": round(ra, 4), "difference": round(da - ra, 4)}

        if aggregation == "average":
            out = {"speciesist_score": round(sum(r["speciesist_score"] for r in results) / len(results), 4), "num_texts": len(results)}
        elif aggregation == "maximum":
            mx = max(results, key=lambda r: r["speciesist_score"])
            out = {"speciesist_score": mx["speciesist_score"], "max_index": results.index(mx)}
        else:
            out = {"scores": results}
        if comp:
            out["comparison"] = comp
        return out
