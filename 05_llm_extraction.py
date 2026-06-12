"""
Step 5: LLM-based symptom extraction
- Claude Haiku / GPT-4o mini / Gemini 1.5 Flash
- Identical structured prompt for all models
- Binary 0/1 per symptom
- Rate limiting + retry + cost tracking
"""

import os, sys, json, time, re
import pandas as pd
from pathlib import Path

# ── API keys (set via env before running) ─────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY    = os.environ.get("OPENAI_API_KEY",    "")
GOOGLE_API_KEY    = os.environ.get("GOOGLE_API_KEY",    "")

ANN_FILE = "/sessions/stoic-practical-fermi/mnt/MIMIC_LLM/files/annotation_sample_1000_cchpi_annotated.csv"
DATA_DIR = "/sessions/stoic-practical-fermi/mnt/MIMIC_LLM/pipeline/data"

# ── 46 symptoms list ──────────────────────────────────────────────────────
SYMPTOMS = [
    "lack_of_energy", "worrying", "feeling_sad", "pain", "feeling_nervous",
    "feeling_drowsy", "dry_mouth", "difficulty_sleeping", "feeling_irritable",
    "nausea", "lack_of_appetite", "difficulty_concentrating", "feeling_bloated",
    "change_in_the_way_food_tastes", "numbness_tingling_in_hands_feet",
    "constipation", "cough", "i_dont_look_like_myself", "itching",
    "swelling_of_arms_or_legs", "weight_loss", "diarrhoea", "dizziness",
    "problems_with_sexual_interest_or_activity", "shortness_of_breath",
    "vomiting", "hair_loss", "problems_with_urination", "mouth_sores",
    "difficulty_swallowing", "changes_in_skin", "sweats",
    "urinary_frequency_day", "urinary_frequency_night", "urinary_incontinence",
    "dysuria", "abdominal_pain", "anal_rectal_pain", "blood_in_stool",
    "mucus_in_stool", "flatulence_gas", "stool_leakage",
    "sore_skin_around_stoma_anal_area", "frequent_bowel_movements",
    "erectile_dysfunction", "dyspareunia"
]

# ── Prompt template ───────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a clinical NLP system. Your task is to extract symptoms from clinical notes.
For each symptom, output 1 if the symptom is PRESENT (positively mentioned, currently experienced) or 0 if ABSENT (not mentioned, denied, negated, or only mentioned as past/resolved history).
Return ONLY a valid JSON object with symptom names as keys and 0 or 1 as values. No explanation."""

def make_user_prompt(text: str) -> str:
    symptoms_str = "\n".join(f'  "{s}": ?' for s in SYMPTOMS)
    return f"""Clinical note (Chief Complaint + HPI):
\"\"\"{text[:2000]}\"\"\"

Extract the following symptoms (1=present, 0=absent):
{{
{symptoms_str}
}}"""

# ── JSON parser (robust) ──────────────────────────────────────────────────
def parse_response(raw: str) -> dict:
    try:
        # Extract JSON block if surrounded by markdown
        m = re.search(r'\{[\s\S]+\}', raw)
        if m:
            data = json.loads(m.group())
            result = {}
            for s in SYMPTOMS:
                v = data.get(s, 0)
                result[s] = 1 if str(v) in ("1", "true", "True") else 0
            return result
    except Exception:
        pass
    return {s: 0 for s in SYMPTOMS}

# ── Claude Haiku ──────────────────────────────────────────────────────────
def extract_claude(text: str, client) -> dict:
    msg = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": make_user_prompt(text)}]
    )
    return parse_response(msg.content[0].text)

# ── GPT-4o mini ───────────────────────────────────────────────────────────
def extract_gpt(text: str, client) -> dict:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=512,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": make_user_prompt(text)}
        ]
    )
    return parse_response(resp.choices[0].message.content)

# ── Gemini 1.5 Flash ──────────────────────────────────────────────────────
def extract_gemini(text: str, client) -> dict:
    resp = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=SYSTEM_PROMPT + "\n\n" + make_user_prompt(text)
    )
    return parse_response(resp.text)

# ── Batch runner ──────────────────────────────────────────────────────────
def run_batch(df, extractor_fn, model_name, out_path,
              batch_delay=0.1, max_retries=3):
    """
    Run extractor on all rows. Supports resume (skip already-done notes).
    """
    # Resume support
    done = set()
    if Path(out_path).exists():
        existing = pd.read_csv(out_path)
        done = set(existing["note_id"].tolist())
        print(f"  Resuming: {len(done)} already done")

    results = []
    total = len(df)
    errors = 0

    for i, row in df.iterrows():
        if row["note_id"] in done:
            continue
        text = row.get("cc_hpi_text", "")
        for attempt in range(max_retries):
            try:
                pred = extractor_fn(text)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    pred = {s: 0 for s in SYMPTOMS}
                    errors += 1
                    print(f"  ERROR row {i}: {e}")
                else:
                    time.sleep(2 ** attempt)

        row_result = {"note_id": row["note_id"],
                      "subject_id": row["subject_id"],
                      "hadm_id": row["hadm_id"]}
        row_result.update({f"pred_{s}": pred[s] for s in SYMPTOMS})
        results.append(row_result)

        # Progressive save every 50 rows
        if len(results) % 50 == 0:
            temp = pd.DataFrame(results)
            if Path(out_path).exists():
                prev = pd.read_csv(out_path)
                temp = pd.concat([prev, temp], ignore_index=True)
            temp.to_csv(out_path, index=False)
            done_count = len(done) + len(results)
            print(f"  [{model_name}] {done_count}/{total} done, {errors} errors",
                  flush=True)

        time.sleep(batch_delay)

    # Final save
    if results:
        final = pd.DataFrame(results)
        if Path(out_path).exists():
            prev = pd.read_csv(out_path)
            final = pd.concat([prev, final], ignore_index=True)
        final.to_csv(out_path, index=False)

    total_done = len(done) + len(results)
    print(f"  [{model_name}] Complete: {total_done}/{total}, errors={errors}")
    return final if results else pd.read_csv(out_path)


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model_arg = sys.argv[1] if len(sys.argv) > 1 else "all"

    df = pd.read_csv(ANN_FILE)
    print(f"Ground truth loaded: {len(df)} notes")

    if model_arg in ("claude", "all") and ANTHROPIC_API_KEY:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        print("\n--- Claude Haiku ---")
        run_batch(df, lambda t: extract_claude(t, client), "Claude Haiku",
                  f"{DATA_DIR}/predictions_claude_gt1000.csv")

    if model_arg in ("gpt", "all") and OPENAI_API_KEY:
        import openai as oa
        client = oa.OpenAI(api_key=OPENAI_API_KEY)
        print("\n--- GPT-4o mini ---")
        run_batch(df, lambda t: extract_gpt(t, client), "GPT-4o mini",
                  f"{DATA_DIR}/predictions_gpt_gt1000.csv")

    if model_arg in ("gemini", "all") and GOOGLE_API_KEY:
        from google import genai
        client = genai.Client(api_key=GOOGLE_API_KEY)
        print("\n--- Gemini 1.5 Flash ---")
        run_batch(df, lambda t: extract_gemini(t, client), "Gemini 1.5 Flash",
                  f"{DATA_DIR}/predictions_gemini_gt1000.csv")

    print("\nDone. Run 04_evaluate.py to compare all methods.")
