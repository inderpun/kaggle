# Kaggle

Competition work by [Inderpuneet Singh](https://www.kaggle.com/inderpuneetsingh).
Each competition lives in its own directory with the same anatomy:

```
<competition>/
├── BRIEFING.md     # what the comp really tests, industry context, approach ladder
├── docs/           # design & learning notes per approach rung
├── src/            # pipeline code (reproducible: `make data`, then run)
├── notebooks/      # Kaggle kernels + EDA findings
└── data/           # gitignored — fetched via the Kaggle CLI
```

## Competitions

| Competition | Focus | Status |
|---|---|---|
| [LLM Classification Finetuning](llm-classification-finetuning/) | Human-preference prediction (Chatbot Arena data) — preference modeling, judge biases, calibration | Rung 2: CV 1.0345 (diagnose-first tuning); rung 4 (QLoRA) designed |
| [ARC Prize 2026](arc-prize-2026/) | Abstract reasoning / program synthesis — DSL search, object priors, test-time training | Rungs 1-2B: harness + 140-op DSL (the measured v1→v2 cliff); rung 3 (TTT) designed |
| Digit Recognizer | MNIST starter | `Digit Recognizer Competition.ipynb` |

## Approach

Every entry climbs an explicit **approach ladder** (baseline → classical → neural →
competition-grade), with each rung measured and written up — including the failures
and how they were diagnosed. The notebooks are written for an external reader.
