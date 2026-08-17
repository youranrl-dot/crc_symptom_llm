# Analysis pipeline

Reproduces every number, table, and figure in *"Leveraging large language models
for symptom cluster in patients with colorectal cancer from MIMIC-IV clinical
discharge notes."*

The annotation pipeline (`annotate_ollama.py`, `validate_kappa.py`, …) turns
discharge notes into symptom predictions. This package takes those predictions
plus the MIMIC-IV structured tables and produces the published results.

---

## Install

```bash
pip install -r requirements-analysis.txt
```

`python-louvain` (imported as `community`) is required rather than networkx's
built-in Louvain. They give the same three-cluster solution here but different
ARI values in the seed-stability analysis, and the manuscript reports
python-louvain.

## Data layout

MIMIC-IV is **not** redistributed in this repository. Download it from PhysioNet
under a valid data use agreement and place the files as follows (or point
`--data-dir` elsewhere):

```
data/
├── admissions.csv                              # MIMIC-IV hosp
├── patients.csv                                # MIMIC-IV hosp
├── diagnoses_icd.csv                           # MIMIC-IV hosp
├── crc_discharge_cchpi.csv                     # derived: 2,728 notes + CC/HPI
├── preds_gemini_full.csv                       # note_id + 46 pred_* columns
├── predictions_claude_haiku_full.csv           # note_id + 46 pred_* columns
└── 200_Note_Adjudication_Kappa_Results.xlsx    # 200-note double annotation
```

`data/` and `results/` are gitignored.

## Run

```bash
python run_analysis.py                    # everything -> results/
python run_analysis.py --skip bootstrap   # skip the slow step (~3 min)
python validate.py                        # 52 regression checks
```

`validate.py` pins every published value. Run it before any commit that touches
`analysis/` — if a refactor or library upgrade moves a number, it fails and names
the number.

---

## Module map

| Module | Produces |
|---|---|
| `analysis/config.py` | every threshold, seed, ICD code list, and cluster definition |
| `analysis/data.py` | cohort construction, outcome derivation, patient-level matrices |
| `analysis/table1.py` | Table 1 |
| `analysis/irr.py` | Section 3.2 summary, Table S2 |
| `analysis/network.py` | phi matrix, Louvain clustering, Table 4 |
| `analysis/sensitivity.py` | the four pre-specified sensitivity analyses + outcome-definition grid |
| `analysis/predictive.py` | Section 3.7 logistic models, metastatic sensitivity, stratified models |
| `analysis/supplementary.py` | Table S1 |
| `analysis/figures.py` | Figure 3 |

```python
from analysis import Paths, load_cohort, load_predictions, patient_level_matrix, build_network

paths  = Paths()
cohort = load_cohort(paths)
preds  = load_predictions(paths.preds_gemini, cohort.notes)
matrix = patient_level_matrix(preds, cohort.subject_ids)
network = build_network(matrix)
```

---

## Operational definitions

These are the decisions that change the numbers. All of them live in
`config.py`; none are hard-coded elsewhere.

**Cohort.** Patients carrying ICD-9 `153.x`/`154.x` or ICD-10 `C18`/`C19`/`C20`
who also have a discharge note with an extractable CC/HPI section →
**1,507 patients, 2,728 notes**.

**Outcomes.** All three are computed over **every** admission of a cohort
patient, anchored on that patient's first admission or discharge:

| Outcome | Definition | n (%) |
|---|---|---|
| In-hospital mortality | `hospital_expire_flag` in any admission | 155 (10.3) |
| 30-day readmission | any admission 0 < t ≤ 30 days after the **first discharge** | 409 (27.1) |
| 1-year mortality | death ≤ 366 days after the **first admission** | 233 (15.5) |

Restricting the admission set to CRC-coded admissions, or re-anchoring on the
first *analysed note*, moves these materially — 1-year mortality ranges from 207
to 330 across definitions. `sensitivity.outcome_definition_grid()` prints the
whole grid, and `run_analysis.py` writes it to
`results/sensitivity_outcome_definitions.csv`. Report one definition and use it
everywhere.

The 366-day window (rather than 365) is what the published n = 233 reflects; 365
days gives n = 231. Both appear in the grid.

**Metastatic disease.** ICD-10 `C77`–`C80` or ICD-9 `196`–`199` in **any**
admission → n = 775 (51.4%). Restricting to CRC-coded admissions gives 690
(45.8%), which does not match Table 1.

**Emergency admission.** `EW EMER.` + `DIRECT EMER.` + `URGENT` → 1,128 (74.9%).
Emergency types alone give 1,046 (69.4%), so the Table 1 row is labelled
"emergency or urgent".

**Network.** Patient-level binary aggregation (present if documented in *any*
note) → prevalence ≥ 5% → phi correlation → edges at ϕ ≥ 0.10 → Louvain
(`random_state=42`). Gemini: 20 symptoms, 136 edges, 3 communities of 7/7/6.
Claude: 21 symptoms, 145 edges, 3 communities of 10/7/4.

No negative phi correlation survives the prevalence filter (range 0.006–0.760),
which is why Figure 3 uses a sequential single-hue ramp rather than a diverging
scale.

---

## Two things worth knowing before you extend this

**The three-cluster count is stable; the membership is less so.** Louvain returns
three communities at both ϕ ≥ 0.10 and ϕ ≥ 0.15, but ARI between those two
partitions is 0.438 and the sizes go 7/7/6 → 11/5/4. "Same number of clusters"
is not "same clusters". `threshold_sensitivity()` reports ARI alongside the count
for this reason.

**Stage adjustment is a sensitivity analysis, not the primary model.** The
primary models adjust for age and sex only, because the question is whether
cluster burden carries prognostic information available from the note itself, and
metastatic disease sits upstream of the symptoms. Adjusting for it is reported
separately, for all three outcomes, so its effect is visible where it weakens a
result as well as where it strengthens one — the Gastrointestinal cluster's
association with both mortality outcomes does not survive it, while the
CRC Disease-Specific cluster's inverse association with 1-year mortality
strengthens (OR 0.85 → 0.83).

---

## Outputs

```
results/
├── table1_patient_characteristics.csv
├── table4_symptom_clusters.csv
├── table_s1_symptom_prevalence.csv
├── table_s2_inter_rater.csv
├── predictive_primary_models.csv
├── predictive_metastatic_sensitivity.csv
├── sensitivity_phi_threshold.csv
├── sensitivity_first_note_dropouts.csv
├── sensitivity_outcome_definitions.csv
├── figure3_phi_matrix.png
└── summary.json
```
