#!/usr/bin/env python3
"""
NER-based symptom extractor (spaCy / scispaCy) for CRC CC/HPI notes.

Unlike the pure dictionary matcher, this restricts symptom matching to
spans the clinical NER model tags as DISEASE entities, then maps those
entity spans to the 46 symptom keys and applies negation (negspaCy /
ConText). This is the "NER (spaCy)" baseline.

RUN IN COLAB (needs internet to download the model):
    pip install scispacy negspacy spacy pandas
    pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz

    python run_ner_scispacy.py \
        --notes notes_200.csv \
        --synonyms symptom_justification_standard_v2.csv \
        --out preds_ner.csv

notes_200.csv must have columns: note_id, note_text
Output: note_id + pred_<symptom> for all 46 symptoms.

If you already have your own NER pipeline, you do NOT need this script —
just run it on notes_200.csv and emit the same output format, then score
with recompute_41_symptoms.py.
"""
import argparse
import re
import pandas as pd
import spacy
from negspacy.negation import Negex  # noqa: F401  (registers the pipe)


def load_synonyms(path):
    j = pd.read_csv(path)
    d = {}
    for _, r in j.iterrows():
        key = str(r["Symptom (item key)"]).strip()
        terms = [t.strip().lower() for t in str(r["Synonyms / trigger terms"]).split(",") if t.strip()]
        d[key] = set(terms)
    return d


def build_nlp():
    nlp = spacy.load("en_ner_bc5cdr_md")
    # negspaCy ConText negation over detected entities
    nlp.add_pipe("negex", config={"ent_types": ["DISEASE"]})
    return nlp


def map_entity_to_symptom(ent_text, syn):
    """Map a detected DISEASE entity span to a symptom key if its text
    contains any trigger term for that symptom."""
    t = ent_text.lower()
    hits = []
    for key, terms in syn.items():
        for term in terms:
            if re.search(r"\b" + re.escape(term) + r"\b", t):
                hits.append(key)
                break
    return hits


def extract(doc, syn, all_keys):
    out = {k: 0 for k in all_keys}
    for ent in doc.ents:
        if ent.label_ != "DISEASE":
            continue
        if getattr(ent._, "negex", False):   # negated entity -> skip
            continue
        for key in map_entity_to_symptom(ent.text, syn):
            out[key] = 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes", required=True, help="CSV with note_id, note_text")
    ap.add_argument("--synonyms", required=True)
    ap.add_argument("--out", default="preds_ner.csv")
    args = ap.parse_args()

    syn = load_synonyms(args.synonyms)
    all_keys = list(syn.keys())
    nlp = build_nlp()
    notes = pd.read_csv(args.notes)

    rows = []
    for _, r in notes.iterrows():
        doc = nlp(str(r["note_text"]))
        labels = extract(doc, syn, all_keys)
        rows.append({"note_id": r["note_id"], **{f"pred_{k}": v for k, v in labels.items()}})
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} rows, {len(all_keys)} symptoms -> {args.out}")


if __name__ == "__main__":
    main()
