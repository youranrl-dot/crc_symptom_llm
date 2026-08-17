"""
Step 1: CRC patient extraction + CC/HPI section parsing (optimized)
"""
import pandas as pd, re, os, sys

BASE = "/sessions/stoic-practical-fermi/mnt/MIMIC_LLM/files"
DISCHARGE_CSV = f"{BASE}/mimic-iv-note-deidentified-free-text-clinical-notes-2.2/note/discharge.csv/discharge.csv"
DIAGNOSES_CSV = f"{BASE}/diagnoses_icd.csv"
OUT_DIR = "/sessions/stoic-practical-fermi/mnt/MIMIC_LLM/pipeline/data"
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading diagnoses_icd.csv ...", flush=True)
diag = pd.read_csv(DIAGNOSES_CSV, dtype={"icd_code": str})
mask_10 = diag["icd_code"].str.match(r"^C1[89]|^C20", na=False)
mask_9  = (diag["icd_version"] == 9) & diag["icd_code"].str.match(r"^15[34]", na=False)
crc_diag     = diag[mask_10 | mask_9].copy()
crc_subjects = set(crc_diag["subject_id"].unique())
print(f"  CRC unique subjects: {len(crc_subjects):,}", flush=True)

hadm_icd = (
    crc_diag.sort_values("seq_num")
    .groupby("hadm_id")
    .agg(primary_icd=("icd_code","first"), all_icd=("icd_code", lambda x: "|".join(x)))
    .reset_index()
)

CC_PAT = re.compile(
    r"Chief\s*Complaint[:\s]*(.*?)(?=\n\s*(?:Major Surgical|Allergies|History of Present Illness|Attending|Past Medical|Social History|Physical Exam)|$)",
    re.IGNORECASE | re.DOTALL
)
HPI_PAT = re.compile(
    r"History\s+of\s+Present\s+Illness[:\s]*(.*?)(?=\n\s*(?:Past Medical|Social History|Family History|Review of Systems|Physical Exam|Medications|Allergies)|$)",
    re.IGNORECASE | re.DOTALL
)
def extract_cc_hpi(text):
    if not isinstance(text, str): return ""
    cc  = (m := CC_PAT.search(text))  and m.group(1).strip() or ""
    hpi = (m := HPI_PAT.search(text)) and m.group(1).strip() or ""
    return re.sub(r"\s+", " ", " ".join(filter(None,[cc,hpi])).lower()).strip()

print("Filtering discharge notes (3.5GB — this takes a while) ...", flush=True)
CHUNK, rows, n = 5_000, [], 0
for chunk in pd.read_csv(DISCHARGE_CSV, chunksize=CHUNK,
                          usecols=["note_id","subject_id","hadm_id","charttime","text"],
                          dtype={"note_id":str,"subject_id":int,"hadm_id":int}):
    n += len(chunk)
    sub = chunk[chunk["subject_id"].isin(crc_subjects)].copy()
    if len(sub):
        sub["cc_hpi_text"] = sub["text"].apply(extract_cc_hpi)
        rows.append(sub.drop(columns=["text"]))
    if n % 50_000 == 0:
        print(f"  processed {n:,} rows, kept {sum(len(r) for r in rows):,}", flush=True)

df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
df = df.merge(hadm_icd, on="hadm_id", how="left")
out = f"{OUT_DIR}/crc_discharge_cchpi.csv"
df.to_csv(out, index=False)
print(f"\nDone. Saved {len(df):,} notes → {out}", flush=True)
print(f"Unique patients: {df['subject_id'].nunique():,}  admissions: {df['hadm_id'].nunique():,}", flush=True)
