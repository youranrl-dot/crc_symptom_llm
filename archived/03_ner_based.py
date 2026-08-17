"""
Step 3: NER-based symptom extraction using spaCy EntityRuler
- spaCy blank 'en' + EntityRuler로 clinical phrase 매칭 (tokenization-aware)
- 동일 negation detection 적용
"""

import pandas as pd
import re
import sys
import os

import spacy
from spacy.pipeline import EntityRuler

sys.path.insert(0, "/sessions/stoic-practical-fermi/mnt/MIMIC_LLM/pipeline")
from symptom_dict import SYMPTOM_KEYWORDS
from importlib import import_module

DATA_DIR = "/sessions/stoic-practical-fermi/mnt/MIMIC_LLM/pipeline/data"
ANN_FILE = "/sessions/stoic-practical-fermi/mnt/MIMIC_LLM/files/annotation_sample_1000_cchpi_annotated.csv"

# ── negation (재사용) ─────────────────────────────────────────────────────
NEG_PRE = [
    r"\bno\b", r"\bnot\b", r"\bdenies\b", r"\bdenied\b", r"\bwithout\b",
    r"\babsent\b", r"\bnegative\s+for\b", r"\bnever\b", r"\bfree\s+of\b",
    r"\bno\s+evidence\s+of\b", r"\brunning\s+out\s+of\b",
    r"\brules?\s+out\b", r"\bno\s+complaint\s+of\b",
    r"\bno\s+history\s+of\b", r"\bdenying\b"
]
NEG_POST = [r"\bwas\s+(?:not|absent)\b", r"\bis\s+(?:not|absent)\b"]
NEG_PRE_PAT  = re.compile("|".join(NEG_PRE),  re.IGNORECASE)
NEG_POST_PAT = re.compile("|".join(NEG_POST), re.IGNORECASE)
PRE_WINDOW, POST_WINDOW = 60, 20

def is_negated_span(text: str, start: int, end: int) -> bool:
    pre_ctx  = text[max(0, start - PRE_WINDOW): start]
    post_ctx = text[end: end + POST_WINDOW]
    last_period = max(pre_ctx.rfind("."), pre_ctx.rfind(";"), pre_ctx.rfind("\n"))
    if last_period >= 0:
        pre_ctx = pre_ctx[last_period + 1:]
    return bool(NEG_PRE_PAT.search(pre_ctx) or NEG_POST_PAT.search(post_ctx))

# ── Build spaCy NLP pipeline with EntityRuler ────────────────────────────
print("Building spaCy EntityRuler NLP pipeline ...")
nlp = spacy.blank("en")
ruler = nlp.add_pipe("entity_ruler", config={"phrase_matcher_attr": "LOWER"})

patterns = []
for sym, keywords in SYMPTOM_KEYWORDS.items():
    for kw in keywords:
        # regex 패턴은 EntityRuler에서 직접 지원 (pattern 형식)
        # r"\b...\b" 형태 → regex pattern, 나머지 → phrase pattern
        if kw.startswith(r"\b") or kw.startswith("r\\b") or "\\b" in kw:
            # regex → string으로 변환 후 phrase로 추가
            clean = re.sub(r'\\b|\(\?.*?\)', '', kw).strip("()").strip()
            if clean:
                patterns.append({"label": sym, "pattern": clean})
        else:
            patterns.append({"label": sym, "pattern": kw})

ruler.add_patterns(patterns)
print(f"  Total patterns added: {len(patterns):,}")


def extract_symptoms_ner(text: str) -> dict:
    if not isinstance(text, str) or len(text.strip()) == 0:
        return {sym: 0 for sym in SYMPTOM_KEYWORDS}
    doc = nlp(text)
    detected = set()
    for ent in doc.ents:
        if not is_negated_span(text, ent.start_char, ent.end_char):
            detected.add(ent.label_)
    return {sym: (1 if sym in detected else 0) for sym in SYMPTOM_KEYWORDS}


def run(input_file: str, output_file: str, label: str = ""):
    print(f"\n{'='*60}")
    print(f"NER-based extraction: {label or input_file}")
    df = pd.read_csv(input_file)
    print(f"  Input rows: {len(df):,}")

    preds = df["cc_hpi_text"].apply(extract_symptoms_ner)
    pred_df = pd.DataFrame(list(preds))
    pred_df.columns = [f"pred_{c}" for c in pred_df.columns]

    out = pd.concat([df[["note_id", "subject_id", "hadm_id"]], pred_df], axis=1)
    out.to_csv(output_file, index=False)
    print(f"  Saved → {output_file}")

    sym_cols = list(pred_df.columns)
    pos_counts = pred_df[sym_cols].sum().sort_values(ascending=False)
    print(f"  Top 5 predicted symptoms:")
    for sym, cnt in pos_counts.head(5).items():
        print(f"    {sym.replace('pred_','')}: {cnt} ({cnt/len(df)*100:.1f}%)")
    return out


if __name__ == "__main__":
    crc_file = f"{DATA_DIR}/crc_discharge_cchpi.csv"
    if os.path.exists(crc_file):
        run(crc_file,
            f"{DATA_DIR}/predictions_ner_full.csv",
            label="Full CRC notes")

    run(ANN_FILE,
        f"{DATA_DIR}/predictions_ner_gt1000.csv",
        label="Ground-truth 1000 notes")
