# crc_symptom_llm

Code accompanying:

> **Leveraging large language models for symptom cluster in patients with
> colorectal cancer from MIMIC-IV clinical discharge notes.**
> Lee Y, Dinov I, Hu X, Jiang Y.

Extracts 46 cancer-related symptoms from the Chief Complaint / History of Present
Illness sections of MIMIC-IV colorectal-cancer discharge notes, benchmarks the
extraction methods against a manually adjudicated gold standard, and derives
symptom clusters from the LLM-extracted symptoms.

**Cohort:** 1,507 patients · 2,728 discharge notes · 46 symptoms
**Gold standard:** 200 double-annotated, adjudicated notes (9,200 symptom–note pairs)

---

## Repository layout

```
├── symptoms.json               canonical 46-symptom key list
├── symptom_dict.py             keyword/synonym dictionary for the rule-based method
├── system_prompt.txt           zero-shot LLM system prompt (46 labels + 20 coding rules)
│
├── 01_extract_crc_notes.py     MIMIC-IV -> CRC cohort + CC/HPI extraction
├── 02_rule_based.py            rule-based baseline
├── run_ner_scispacy.py         NER baseline (scispaCy en_ner_bc5cdr_md + negspaCy)
├── 05_llm_extraction.py        LLM extraction driver
├── run_hybrid.py               hybrid variants (LLM + rule-based negation filter)
├── negation_filter_config.py   the filter's cue list, window, and trigger terms
│
├── compare_with_gold.py        benchmark vs gold -> Tables 2-3, Figure 1
│
├── analysis/                   the published analysis package
├── run_analysis.py             full pipeline -> results/
├── validate.py                 58 regression checks against the published numbers
├── ANALYSIS.md                 module map + every operational definition
│
├── requirements.txt            extraction stage
└── requirements-analysis.txt   analysis stage
```

An earlier numbered pipeline (`03_ner_based.py`, `04_evaluate.py`,
`06_network_analysis.py`, `07_predictive_validity.py`) produced earlier versions of
the reported numbers and has been removed; `analysis/`, `compare_with_gold.py`, and
`run_ner_scispacy.py` replace it. Those files remain in the git history if the
provenance of an earlier value is ever needed.

---

## Data

MIMIC-IV is **not** redistributed here. Access requires credentialing and a data
use agreement with PhysioNet: <https://physionet.org/content/mimiciv/>

`data/`, `output/`, and `results/` are gitignored. Do not commit note text, or any
derived file containing clinical narrative — the DUA prohibits it.

Expected layout once you have access:

```
data/
├── admissions.csv                              # MIMIC-IV hosp module
├── patients.csv                                # MIMIC-IV hosp module
├── diagnoses_icd.csv                           # MIMIC-IV hosp module
├── crc_discharge_cchpi.csv                     # from 01_extract_crc_notes.py
├── preds_gemini_full.csv                       # note_id + 46 pred_* columns
├── predictions_claude_haiku_full.csv           # note_id + 46 pred_* columns
└── 200_Note_Adjudication_Kappa_Results.xlsx    # 200-note double annotation
```

---

## Quick start

```bash
# analysis stage — tables, figures, models
pip install -r requirements-analysis.txt
python run_analysis.py     # writes every table and figure to results/
python validate.py         # 58 regression checks

# hybrid variants, then the full six-method benchmark
python run_hybrid.py --pred data/predictions_claude_haiku_full.csv \
                     --out  data/preds_claude_hybrid.csv
python run_hybrid.py --pred data/preds_gemini_full.csv \
                     --out  data/preds_gemini_hybrid.csv

python compare_with_gold.py \
    --pred rule_based=data/preds_rule_based.csv \
    --pred NER=data/preds_ner_scispacy.csv \
    --pred claude=data/predictions_claude_haiku_full.csv \
    --pred claude_hybrid=data/preds_claude_hybrid.csv \
    --pred gemini=data/preds_gemini_full.csv \
    --pred gemini_hybrid=data/preds_gemini_hybrid.csv \
    --na-policy zero
```

`validate.py` pins every value reported in the manuscript — cohort counts,
Table 1, kappa summaries, network structure, all sensitivity analyses, and all
odds ratios. If a refactor or a library upgrade moves a number, it fails and
names the number. Run it before any commit that touches `analysis/`.

---

## The 46-symptom schema

Built by combining two validated instruments and de-duplicating:

* **MSAS** — Memorial Symptom Assessment Scale (Portenoy et al., 1994)
* **EORTC QLQ-CR29** — colorectal module (Whistance et al., 2009)

`symptoms.json` holds the canonical key list. Those keys are the contract across
the whole repository: the JSON keys the models must emit, the `label_*` columns
of the adjudicated gold standard, and the `pred_*` columns of every prediction
file. Renaming one without the other two silently breaks the benchmark, so
`compare_with_gold.py` treats any mismatch as a hard error.

`system_prompt.txt` holds the zero-shot prompt, including the 20 coding rules
developed during pilot human-in-the-loop annotation. Those rules are the
substantive part — they resolve the ambiguous cases that recur in this note set
(negation scope, altered mental status mapping, diagnosis-to-symptom inference,
temporal scope).

---

## Methods benchmarked

| Method | Approach |
|---|---|
| Rule-based | curated synonym dictionary + 80-character negation window |
| NER | scispaCy `en_ner_bc5cdr_md` DISEASE entities, negated spans dropped with negspaCy ConText, mapped to the 46-symptom inventory |
| Claude Haiku | zero-shot, structured JSON, `temperature=0.0` |
| Gemini 3.5 Flash | identical protocol; CC/HPI truncated to 2,500 characters |
| + Hybrid variants | post-hoc rule-based negation filtering applied to each LLM's output |

The hybrid variants **degraded** performance — a fixed 100-character negation
window overrides correct LLM predictions on constructions like *"no relief from
nausea"*. Reported as a negative finding: do not apply post-hoc negation
filtering to LLM output without syntactic scope validation.

---

## Reproducing the analysis

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

## Known issues

Tracked openly so nobody builds on a stale artefact.

**1. Superseded scripts were removed.** The earlier numbered pipeline scored
against the 1,000-note annotation file rather than the 200-note adjudicated gold
standard, evaluated 41 of the 46 symptoms, used a keyword matcher labelled as NER,
and predates the outcome-definition fixes documented in ANALYSIS.md. It was removed
rather than kept alongside the current code so that no one runs it by mistake; it
is recoverable from the git history.

**2. Hard-coded absolute paths.** `01`, `02`, and `05` contain absolute paths from
the machine they were developed on. They will not run elsewhere until those are
parameterised.

**3. `05_llm_extraction.py` names different models than the manuscript.** Its
docstring lists Claude Haiku / GPT-4o mini / Gemini 1.5 Flash; the manuscript
benchmarks Claude Haiku (`claude-haiku-4-5-20251001`) and Gemini 3.5 Flash
(`gemini-3.5-flash`), and does not report a GPT model.

**4. Two Gemini inference runs exist and they disagree.** A 200-note prediction
file and the 2,728-note file, produced by separate runs, disagree on **93 of 9,200
labels (1.0%)** for the notes they share. The manuscript reports the 2,728-note
run — the one the clusters were derived from. This is the `temperature=0.0`
non-determinism the Limitations section anticipates.

**5. 18 notes have no predictions in the Claude output** (`error=missing_cc_hpi`,
all 46 labels null). One is in the 200-note gold set; the published confusion
matrix counts it as all true negatives, which `--na-policy zero` reproduces. The
default policy is `error`, so this cannot pass unnoticed again.

**6. Earlier NER and hybrid results could not be reproduced and were regenerated.**
No saved prediction file reproduced the previously reported NER or hybrid rows, and
the scripts that produced them were not preserved. Both were re-run from the code
in this repository and the manuscript now reports those values. The conclusions are
unchanged; the hybrid penalty is slightly larger than previously reported.

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
PhysioNet data use agreement. This is research code for retrospective analysis;
it is not validated for clinical decision-making.
