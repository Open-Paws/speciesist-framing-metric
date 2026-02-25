---
title: Speciesist Framing
tags:
- evaluate
- measurement
- bias
- speciesism
- ethics
description: "Measures speciesist framing bias in text"
sdk: gradio
sdk_version: 3.19.1
app_file: app.py
pinned: false
---

# Speciesist Framing Measurement

Quantifies speciesist language patterns in text — framing that normalizes exploitation of certain animal species while extending moral consideration to others.

First bias measurement designed to detect species-based discrimination in language, complementing existing measurements for demographic biases in the HuggingFace evaluate ecosystem.

## Detected Patterns

| Category | Examples | Detects |
|----------|----------|---------|
| **Objectification** | "livestock", "production animal", "unit" | Treating sentient beings as commodities |
| **Euphemism** | "harvested", "processed", "depopulated" | Sanitized language about killing |
| **Deindividuation** | "batch", "inventory", "throughput" | Removing individual identity |
| **Industry normalization** | "feedlot", "gestation crate" | Normalizing exploitation systems |
| **Property framing** | "chattel", "vermin", "game animal" | Framing animals as property |

## Non-Speciesist Patterns Detected

| Category | Examples |
|----------|----------|
| **Rights language** | "sentient being", "animal liberation", "nonhuman animal" |
| **Agency language** | Attributing emotions, individuality, preferences |
| **Accurate language** | "killed" not "harvested", "confined" not "housed" |

## Usage

```python
import evaluate

speciesist = evaluate.load("open-paws/speciesist_framing")

# Score individual texts
results = speciesist.compute(
    data=["The livestock were processed at the facility."]
)

# Compare two groups
results = speciesist.compute(
    data=["The pigs were processed at the plant."],
    references=["The dogs were killed at the facility."],
    aggregation="average"
)
```

## Output

- **speciesist_score** (`float`): 0.0 (non-speciesist) to 1.0 (highly speciesist)
- **speciesist_categories** (`dict`): Breakdown by framing category
- **speciesist_terms** / **non_speciesist_terms** (`list`): Matched terms with weights
- **comparison** (`dict`, optional): Differential scores between text groups

## Limitations

- Lexicon-based approach may miss novel speciesist framings
- Context-dependent terms may produce false positives
- English-language only in current version
- Reflects animal rights ethical framework — what constitutes "speciesist" is a normative position

## Citation

```bibtex
@misc{speciesist_framing_2026,
  title={Speciesist Framing: A Measurement for Evaluating Species Bias in Language},
  author={Open Paws},
  year={2026},
  publisher={GitHub},
  url={https://github.com/Open-Paws/speciesist-framing-metric}
}
```

## References

- Singer, P. (1975). *Animal Liberation*. New York Review/Random House.
- Dunayer, J. (2001). *Animal Equality: Language and Liberation*. Ryce Publishing.
- Stibbe, A. (2012). *Animals Erased*. Wesleyan University Press.
- Cambridge Declaration on Consciousness (2012).
- Poore, J. & Nemecek, T. (2018). Science, 360(6392), 987-992.
