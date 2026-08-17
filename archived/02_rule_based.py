"""
Step 2: Rule-based symptom extraction
- Window-based negation detection
- 46 symptoms × keyword dict
"""

import pandas as pd
import re
import sys
import os

sys.path.insert(0, "/sessions/stoic-practical-fermi/mnt/MIMIC_LLM/pipeline")
from symptom_dict import SYMPTOM_KEYWORDS

DATA_DIR = "/sessions/stoic-practical-fermi/mnt/MIMIC_LLM/pipeline/data"
ANN_FILE = "/sessions/stoic-practical-fermi/mnt/MIMIC_LLM/files/annotation_sample_1000_cchpi_annotated.csv"

# ── Negation cues (pre-keyword window) ────────────────────────────────────
NEG_PRE = [
    r"\bno\b", r"\bnot\b", r"\bdenies\b", r"\bdenied\b", r"\bwithout\b",
    r"\babsent\b", r"\bnegative\s+for\b", r"\bnever\b", r"\bfree\s+of\b",
    r"\bno\s+evidence\s+of\b", r"\brunning\s+out\s+of\b",
    r"\brules?\s+out\b", r"\bno\s+complaint\s+of\b",
    r"\bno\s+history\s+of\b", r"\bdenying\b"
]
NEG_POST = [
    r"\bwas\s+(?:not|absent)\b", r"\bis\s+(?:not|absent)\b"
]
NEG_PRE_PAT  = re.compile("|".join(NEG_PRE),  re.IGNORECASE)
NEG_POST_PAT = re.compile("|".join(NEG_POST), re.IGNORECASE)

PRE_WINDOW  = 60   # chars before keyword
POST_WINDOW = 20   # chars after keyword

def is_negated(text: str, start: int, end: int) -> bool:
    pre_ctx  = text[max(0, start - PRE_WINDOW) : start]
    post_ctx = text[end : end + POST_WINDOW]
    # Check sentence boundary (don't cross a period)
    last_period = max(pre_ctx.rfind("."), pre_ctx.rfind(";"), pre_ctx.rfind("\n"))
    if last_period >= 0:
        pre_ctx = pre_ctx[last_period + 1:]
    if NEG_PRE_PAT.search(pre_ctx):
        return True
    if NEG_POST_PAT.search(post_ctx):
        return True
    return False

# Body-part qualifiers that mean it's NOT generic pain
# (these are already captured by abdominal_pain / anal_rectal_pain etc.)
PAIN_BODY_QUALIFIERS = re.compile(
    r"\b(abdominal|abdomen|stomach|belly|epigastric|periumbilical|peri-umbilical|"
    r"rectal|anal|anorectal|perianal|peri-anal|recti|rectal|"
    r"pelvic|perineal|stoma|colostomy|ileostomy)\s*$",
    re.IGNORECASE
)
PAIN_QUALIFIER_WINDOW = 40  # chars before the pain keyword to check

def is_body_part_qualified_pain(text: str, match_start: int) -> bool:
    """Return True if this pain match is preceded by a body-part qualifier
    (meaning it should be captured by a more specific symptom, not generic pain)."""
    pre = text[max(0, match_start - PAIN_QUALIFIER_WINDOW) : match_start].rstrip()
    # check last word(s) before "pain"
    return bool(PAIN_BODY_QUALIFIERS.search(pre))

def extract_symptoms_rulebased(text: str) -> dict:
    if not isinstance(text, str) or len(text.strip()) == 0:
        return {sym: 0 for sym in SYMPTOM_KEYWORDS}
    results = {}
    for sym, keywords in SYMPTOM_KEYWORDS.items():
        found = 0
        for kw in keywords:
            pattern = re.compile(kw, re.IGNORECASE)
            for m in pattern.finditer(text):
                if not is_negated(text, m.start(), m.end()):
                    # For generic "pain", skip body-part qualified matches
                    if sym == "pain" and is_body_part_qualified_pain(text, m.start()):
                        continue
                    found = 1
                    break
            if found:
                break
        results[sym] = found
    return results


def run(input_file: str, output_file: str, label: str = ""):
    print(f"\n{'='*60}")
    print(f"Rule-based extraction: {label or input_file}")
    df = pd.read_csv(input_file)
    print(f"  Input rows: {len(df):,}")

    preds = df["cc_hpi_text"].apply(extract_symptoms_rulebased)
    pred_df = pd.DataFrame(list(preds))
    pred_df.columns = [f"pred_{c}" for c in pred_df.columns]

    out = pd.concat([df[["note_id", "subject_id", "hadm_id"]], pred_df], axis=1)
    out.to_csv(output_file, index=False)
    print(f"  Saved → {output_file}")

    # Quick prevalence check
    sym_cols = [c for c in pred_df.columns]
    pos_counts = pred_df[sym_cols].sum().sort_values(ascending=False)
    print(f"  Top 5 predicted symptoms:")
    for sym, cnt in pos_counts.head(5).items():
        print(f"    {sym.replace('pred_','')}: {cnt} ({cnt/len(df)*100:.1f}%)")
    return out


if __name__ == "__main__":
    # A) Full CRC notes
    crc_file = f"{DATA_DIR}/crc_discharge_cchpi.csv"
    if os.path.exists(crc_file):
        run(crc_file,
            f"{DATA_DIR}/predictions_rulebased_full.csv",
            label="Full CRC notes")

    # B) Ground truth subset (1000 annotated)
    run(ANN_FILE,
        f"{DATA_DIR}/predictions_rulebased_gt1000.csv",
        label="Ground-truth 1000 notes")
