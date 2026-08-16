# LLM-Based Symptom Extraction from CRC Clinical Notes

Code repository for: **"Extracting Symptoms from Colorectal Cancer Patients Using Large Language Models in MIMIC-IV Clinical Discharge Summaries: A Multi-Method Comparative Study"**

Authors: Youran Lee, Ivo Dinov, Xiaosu Hu, Yun Jiang

---

## Overview

This repository contains the full pipeline for:
1. Extracting CRC patient discharge notes from MIMIC-IV
2. Symptom extraction via 4 methods (Rule-based, NER, Claude Haiku, Gemini 2.5 Flash) + 2 hybrid variants
3. Evaluation against 1,000 manually annotated notes (46 MSAS/EORTC QLQ-CR29 symptoms)
4. Symptom co-occurrence network analysis (Louvain community detection)
5. Predictive validity via logistic regression

---

## Data Access

This study uses **MIMIC-IV** (Medical Information Mart for Intensive Care IV), a de-identified EHR database from Beth Israel Deaconess Medical Center.

- Access requires a PhysioNet credentialed account and signed Data Use Agreement (DUA)
- Apply at: https://physionet.org/content/mimic-iv-note/2.2/
- Data files are **not included** in this repository

---

## Pipeline

```
01_extract_crc_notes.py       Extract CRC discharge notes (CC/HPI) from MIMIC-IV
02_rule_based.py              Rule-based symptom extraction (keyword matching + negation)
03_ner_based.py               NER-based extraction (samrawal/bert-base-uncased_clinical-ner)
04_evaluate.py                Evaluation vs ground truth (F1, precision, recall)
05_llm_extraction.py          LLM extraction (Claude Haiku, Gemini 2.5 Flash)
06_network_analysis.py        Symptom co-occurrence network + Louvain stability
07_predictive_validity.py     Logistic regression: cluster burden → clinical outcomes
symptom_dict.py               46-symptom dictionary with synonyms
```

---

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Set API keys as environment variables:
```bash
export ANTHROPIC_API_KEY="your_key_here"
export GOOGLE_API_KEY="your_key_here"
```

---

## Usage

```bash
# 1. Extract CRC notes
python 01_extract_crc_notes.py

# 2. Rule-based extraction
python 02_rule_based.py

# 3. NER extraction
python 03_ner_based.py

# 4. LLM extraction
python 05_llm_extraction.py --model claude
python 05_llm_extraction.py --model gemini

# 5. Evaluate all methods
python 04_evaluate.py

# 6. Network analysis (Louvain + stability)
python 06_network_analysis.py --method gemini
python 06_network_analysis.py --method claude

# 7. Predictive validity (OR analysis)
python 07_predictive_validity.py
```

---

## LLM Prompt Template

Identical structured prompt used for both Claude Haiku and Gemini 2.5 Flash:

```
System: You are a clinical NLP system. For each symptom, output 1 if PRESENT
        (positively mentioned, currently experienced) or 0 if ABSENT
        (not mentioned, denied, negated, or resolved). Return ONLY valid JSON.

User:   Clinical note (CC/HPI, truncated to 2,000 characters):
        """[NOTE TEXT]"""

        Extract the following symptoms (1=present, 0=absent):
        { "lack_of_energy": ?, "pain": ?, ... }
```

**Model versions:**
- Claude Haiku: `claude-haiku-4-5-20251001`
- Gemini: `gemini-2.5-flash` *(confirm exact version string)*

**Parameters:** `temperature=0.0`, `max_tokens=512`

---

## Symptom List (46 symptoms)

Derived from Memorial Symptom Assessment Scale (MSAS) and EORTC QLQ-CR29:

| # | Symptom | # | Symptom |
|---|---------|---|---------|
| 1 | Lack of energy | 24 | Problems with sexual interest/activity |
| 2 | Worrying | 25 | Shortness of breath |
| 3 | Feeling sad | 26 | Vomiting |
| 4 | Pain | 27 | Hair loss |
| 5 | Feeling nervous | 28 | Problems with urination |
| 6 | Feeling drowsy | 29 | Mouth sores |
| 7 | Dry mouth | 30 | Difficulty swallowing |
| 8 | Difficulty sleeping | 31 | Changes in skin |
| 9 | Feeling irritable | 32 | Sweats |
| 10 | Nausea | 33 | Urinary frequency (day) |
| 11 | Lack of appetite | 34 | Urinary frequency (night) |
| 12 | Difficulty concentrating | 35 | Urinary incontinence |
| 13 | Feeling bloated | 36 | Dysuria |
| 14 | Change in food taste | 37 | Abdominal pain |
| 15 | Numbness/tingling in hands/feet | 38 | Anal/rectal pain |
| 16 | Constipation | 39 | Blood in stool |
| 17 | Cough | 40 | Mucus in stool |
| 18 | I don't look like myself | 41 | Flatulence/gas |
| 19 | Itching | 42 | Stool leakage |
| 20 | Swelling of arms or legs | 43 | Sore skin around stoma/anal area |
| 21 | Weight loss | 44 | Frequent bowel movements |
| 22 | Diarrhoea | 45 | Erectile dysfunction |
| 23 | Dizziness | 46 | Dyspareunia |

---

## Citation

> Lee Y, Dinov I, Hu X, Kang Y. Extracting Symptoms from Colorectal Cancer Patients Using Large Language Models in MIMIC-IV Clinical Discharge Summaries: A Multi-Method Comparative Study. *(Under review, 2025)*

---

## License

Code: MIT License  
Data: Subject to PhysioNet MIMIC-IV Data Use Agreement
