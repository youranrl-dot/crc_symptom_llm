#!/usr/bin/env python3
"""Build the hybrid variants: LLM predictions + post-hoc rule-based negation filtering.

For every symptom an LLM marked present, the original CC/HPI text is re-scanned. If
any trigger term for that symptom appears within ``WINDOW`` characters *after* a
negation cue, the prediction is flipped to absent.

    python run_hybrid.py \
        --pred data/predictions_claude_haiku_full.csv \
        --notes data/crc_discharge_cchpi.csv \
        --out data/preds_claude_hybrid.csv

The filter configuration lives in ``negation_filter_config.py``. Score the output
with ``compare_with_gold.py`` to regenerate the hybrid rows of Tables 2-3.

This approach is reported in the manuscript as a **negative** result: the fixed
character window has no notion of syntactic scope, so it overrides correct LLM
predictions on constructions such as "no relief from nausea", where the symptom is
present despite the nearby negation cue.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from negation_filter_config import NEGATION_PATTERNS, SYMPTOM_KEYWORDS, WINDOW  # noqa: E402


def _compile(term: str) -> re.Pattern:
    """Entries starting with \\b are regexes; everything else is a literal."""
    try:
        return re.compile(term, re.IGNORECASE) if term.startswith(r"\b") \
            else re.compile(re.escape(term), re.IGNORECASE)
    except re.error:
        return re.compile(re.escape(term), re.IGNORECASE)


_PATTERNS = {sym: [_compile(t) for t in terms] for sym, terms in SYMPTOM_KEYWORDS.items()}


def is_negated(text_lower: str, symptom: str) -> bool:
    """True if any trigger term for ``symptom`` sits within WINDOW chars of a cue."""
    for pattern in _PATTERNS.get(symptom, []):
        for match in pattern.finditer(text_lower):
            if NEGATION_PATTERNS.search(text_lower[max(0, match.start() - WINDOW):match.start()]):
                return True
    return False


def apply_filter(preds: pd.DataFrame, notes: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    merged = preds.merge(notes[["note_id", "cc_hpi_text"]], on="note_id", how="left")
    missing_text = int(merged["cc_hpi_text"].isna().sum())
    if missing_text:
        print(f"  warning: {missing_text} note(s) have no CC/HPI text; "
              "their predictions pass through unchanged")

    cols = [f"pred_{k}" for k in labels]
    base = merged[cols].fillna(0).astype(int)
    out = base.copy()

    for row_i, text in enumerate(merged["cc_hpi_text"]):
        if not isinstance(text, str) or not text.strip():
            continue
        low = text.lower()
        for k, col in zip(labels, cols):
            if base.iat[row_i, cols.index(col)] == 1 and is_negated(low, k):
                out.iat[row_i, cols.index(col)] = 0

    before, after = int(base.to_numpy().sum()), int(out.to_numpy().sum())
    print(f"  positives {before:,} -> {after:,}  ({before - after:,} flipped to absent)")

    out.insert(0, "note_id", merged["note_id"].values)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred", type=Path, required=True, help="LLM prediction CSV")
    parser.add_argument("--notes", type=Path, default=HERE / "data" / "crc_discharge_cchpi.csv")
    parser.add_argument("--symptoms", type=Path, default=HERE / "symptoms.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    labels = json.load(open(args.symptoms, encoding="utf-8"))["labels"]
    preds = pd.read_csv(args.pred)
    missing = [k for k in labels if f"pred_{k}" not in preds.columns]
    if missing:
        raise KeyError(f"{args.pred.name} is missing {len(missing)} labels: {missing[:5]}")

    notes = pd.read_csv(args.notes)
    print(f"{args.pred.name} -> {args.out.name}")
    result = apply_filter(preds, notes, labels)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    print(f"  wrote {len(result)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
