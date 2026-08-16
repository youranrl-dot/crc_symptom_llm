# crc_symptom_llm

Code accompanying:

> **Leveraging large language models for symptom cluster in patients with
> colorectal cancer from MIMIC-IV clinical discharge notes.**
> Lee Y, Dinov I, Hu X, Jiang Y.

Extracts 46 cancer-related symptoms from the Chief Complaint / History of Present
Illness sections of MIMIC-IV colorectal-cancer discharge notes, benchmarks four
extraction methods against a manually adjudicated gold standard, and derives
symptom clusters from the best-performing method.

**Cohort:** 1,507 patients · 2,728 discharge notes · 46 symptoms
**Gold standard:** 200 double-annotated notes (9,200 symptom–note pairs)

---

## What is in this repository

There are two independent tracks. Both are here; only the first is in the paper.

### 1. Published pipeline — annotation + analysis

| Stage | Files |
|---|---|
| Symptom schema & prompt | `symptoms.json`, `system_prompt.txt` |
| Gold-standard input prep | `build_200_note_input.py` |
| Baselines | `run_rule_based.py`, `run_ner_scispacy.py` |
| Benchmarking vs gold | `compare_with_gold.py`, `validate_kappa.py` |
| **Analysis pipeline** | **`analysis/`, `run_analysis.py`, `validate.py`** — see [ANALYSIS.md](ANALYSIS.md) |

The two LLMs benchmarked in the manuscript are **Claude Haiku**
(`claude-haiku-4-5-20251001`) and **Gemini 3.5 Flash** (`gemini-3.5-flash`),
both applied zero-shot at `temperature=0.0` with the same system prompt and
SYNONYM_GUIDE.

### 2. Local open-weight track — Gemma via Ollama

`annotate_ollama.py`, `run_200_note_ollama.py`

An exploratory pipeline that runs the same prompt against a locally hosted Gemma
model. **It is not one of the methods benchmarked in the manuscript.** It is kept
because locally deployable open-weight models are the privacy-preserving
alternative discussed in the Limitations, and because it is useful for anyone who
cannot send clinical text to a hosted API. Results from it are not reported.

---

## Data

MIMIC-IV is **not** redistributed here. Access requires credentialing and a data
use agreement with PhysioNet: <https://physionet.org/content/mimiciv/>

`data/` and `output/` are gitignored. Do not commit note text, note IDs joined to
text, or any derived file that contains clinical narrative — the DUA prohibits it.

Expected layout once you have access:

```
data/
├── admissions.csv                              # MIMIC-IV hosp module
├── patients.csv                                # MIMIC-IV hosp module
├── diagnoses_icd.csv                           # MIMIC-IV hosp module
├── crc_discharge_cchpi.csv                     # derived: 2,728 notes + CC/HPI text
├── preds_gemini_full.csv                       # note_id + 46 pred_* columns
├── predictions_claude_haiku_full.csv           # note_id + 46 pred_* columns
└── 200_Note_Adjudication_Kappa_Results.xlsx    # 200-note double annotation
```

---

## Quick start

```bash
# analysis pipeline (tables, figures, models)
pip install -r requirements-analysis.txt
python run_analysis.py          # writes everything to results/
python validate.py              # 52 regression checks against the published numbers

# annotation pipeline
pip install -r requirements.txt
```

`validate.py` pins every value reported in the manuscript — cohort counts,
Table 1, kappa summaries, network structure, all sensitivity analyses, and all
odds ratios. If a refactor or a library upgrade moves a number, it fails and
names the number. Run it before any commit that touches `analysis/`.

---

## The 46-symptom schema

Derived by combining two validated instruments and de-duplicating:

* **MSAS** — Memorial Symptom Assessment Scale (Portenoy et al., 1994)
* **EORTC QLQ-CR29** — colorectal module (Whistance et al., 2009)

`symptoms.json` holds the canonical key list. Those keys are the contract across
the whole repository: they are the JSON keys the models must emit, the `label_*`
columns of the adjudicated gold standard, and the `pred_*` columns of every
prediction file. Do not rename one without renaming all three.

`system_prompt.txt` holds the system prompt, including the 20 coding rules
developed during pilot human-in-the-loop annotation. Those rules are the
substantive part — they resolve the ambiguous cases that recur in this note set
(negation scope, altered mental status mapping, diagnosis-to-symptom inference,
temporal scope).

---

## Methods benchmarked

| Method | Approach |
|---|---|
| Rule-based | curated synonym dictionary + 80-character negation window |
| NER | `samrawal/bert-base-uncased_clinical-ner` via spaCy EntityRuler, mapped to the 46-symptom inventory |
| Claude Haiku | zero-shot, structured JSON, `temperature=0.0` |
| Gemini 3.5 Flash | identical protocol; CC/HPI truncated to 2,500 characters |
| + Hybrid variants | post-hoc rule-based negation filtering applied to each LLM's output |

Headline result: Gemini 3.5 Flash reached the highest Macro F1 (0.70), followed
by Claude Haiku (0.63); rule-based (0.44) and NER (0.38) trailed. The hybrid
variants **degraded** performance — a fixed 100-character negation window
overrides correct LLM predictions on constructions like *"no relief from
nausea"*. Reported as a negative finding; do not apply post-hoc negation
filtering to LLM output without syntactic scope validation.

---

## Reproducing the results

See [**ANALYSIS.md**](ANALYSIS.md) for the module map, every operational
definition, and the output manifest. The definitions that most affect the numbers:

* **Cohort** — ICD-9 `153.x`/`154.x` or ICD-10 `C18`/`C19`/`C20`, restricted to
  patients with an extractable CC/HPI section.
* **Outcomes** — computed over *all* admissions of a cohort patient, anchored on
  that patient's first admission or discharge. Changing the admission set or the
  anchor moves 1-year mortality between 207 and 330;
  `results/sensitivity_outcome_definitions.csv` prints the full grid.
* **Network** — patient-level any-note aggregation → prevalence ≥ 5% → phi
  correlation → edges at ϕ ≥ 0.10 → Louvain (`random_state=42`, python-louvain).

---

## Known issues in this repository

Tracked openly so nobody builds on a stale artefact:

1. **`compare_with_gold.py` uses a stale label mapping.** Its alias table lists
   symptoms that are not in the 46-symptom inventory (`fever`, `chills`,
   `chest_pain`, `rash`, `early_satiety`, `facial_flushing`) and covers only 27
   labels. Use `validate_kappa.py`, or the `analysis/irr.py` module, until it is
   rewritten against the canonical `symptoms.json`.
2. **Prompt-token accounting is not recorded.** API cost and rate-limit figures
   are not yet in the repository or the Supplementary Materials.

---

## Citation

```bibtex
@article{lee_crc_symptom_llm,
  title   = {Leveraging large language models for symptom cluster in patients
             with colorectal cancer from MIMIC-IV clinical discharge notes},
  author  = {Lee, Youran and Dinov, Ivo and Hu, Xiaosu and Jiang, Yun},
  note    = {Manuscript},
  url     = {https://github.com/youranrl-dot/crc_symptom_llm}
}
```

MIMIC-IV: Johnson AEW, Bulgarelli L, Shen L, et al. *Sci Data*. 2023;10(1):1.

---

## License and use

Code is released for academic reuse. Any use of MIMIC-IV remains governed by the
PhysioNet data use agreement. This repository is research code for retrospective
analysis; it is not validated for clinical decision-making.
