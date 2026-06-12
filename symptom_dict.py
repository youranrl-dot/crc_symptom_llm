"""
46-symptom keyword dictionary
각 symptom별 임상 표현 키워드 목록 (lower-case, regex-safe)
"""

SYMPTOM_KEYWORDS = {
    "lack_of_energy": [
        "fatigue", "fatigued", "tired", "tiredness", "exhausted", "exhaustion",
        "lethargy", "lethargic", "weakness", "weak", "no energy", "low energy",
        "lack of energy", "lack energy", "worn out", "worn-out", "malaise",
        "asthenia", "sluggish", "run down"
    ],
    "worrying": [
        r"\bworr(?:y|ied|ying)\b",      # worry/worried/worrying
        r"\banxi(?:ous|ety)\b",          # anxious/anxiety
        "apprehensive", "apprehension",
        r"\bfearful\b", r"\bdread\b",
        r"\bpanic\b", r"\bpanic attack\b",
        r"\bphobia\b",
        # stress only when explicitly psychological
        r"\bpsychological stress\b", r"\bemotional stress\b",
    ],
    "feeling_sad": [
        r"\bsad\b", "sadness",
        r"\bdepressed mood\b", r"\bfeeling depressed\b",
        r"\bclinical depression\b", r"\bmajor depression\b",
        r"\bdepressive disorder\b",
        "low mood", "tearful", "crying", "grief",
        r"\bhopeless\b", "hopelessness", "despondent", "unhappy",
        r"\bfeeling down\b", r"\bdown in the dumps\b",
        "dysphoria", "dysphoric", "melancholy",
    ],
    "pain": [
        r"\bpain\b", r"\bpains\b", "ache", "aching", "hurt", "hurting",
        "discomfort", "sore", "soreness", "tenderness", "tender", "painful",
        "agony", "burning pain", "stabbing", "throbbing", "cramping pain"
    ],
    "feeling_nervous": [
        "nervous", "nervousness", "jittery", "on edge", "restless", "restlessness",
        r"\bfeel(?:ing)?\s+tense\b", r"\btension\s+(?:and|anxiety)\b",
        r"\banxi(?:ous|ety)\b", "agitated", "agitation", "jumpy", "apprehensive"
    ],
    "feeling_drowsy": [
        "drowsy", "drowsiness", "somnolent", "somnolence", "sleepy", "sleepiness",
        "sedated", "sedation", "hypersomnia", "oversleeping", "excessive sleep",
        # clinical variants seen in GT-positive notes (18/69 = lethargy, 7/69 = somnolent)
        r"\bletharg(?:y|ic)\b",      # lethargy, lethargic
        "groggy", "obtunded", "obtundation",
        r"\baltered\s+mental\s+status\b",  # AMS
        r"\bmental\s+status\s+change\b",
        r"\bencephalopathy\b",
    ],
    "dry_mouth": [
        "dry mouth", "drymouth", "xerostomia", "mouth dryness", "dryness of mouth",
        "decreased saliva", "reduced saliva"
    ],
    "difficulty_sleeping": [
        "insomnia", "difficulty sleeping", "trouble sleeping", "can't sleep",
        "cannot sleep", "sleep disturbance", "poor sleep", "disrupted sleep",
        "sleep difficulty", "unable to sleep", "sleeplessness", "waking up",
        "early awakening"
    ],
    "feeling_irritable": [
        "irritable", "irritability", "agitated", "agitation", "short tempered",
        "easily upset", "mood swings", "grumpy", "impatient"
    ],
    "nausea": [
        "nausea", "nauseated", "nauseous", "queasy", "sick to stomach",
        "feel sick", "feeling sick", "upset stomach", "queasiness"
    ],
    "lack_of_appetite": [
        "anorexia", "poor appetite", "decreased appetite", "reduced appetite",
        "loss of appetite", "not eating", "no appetite", "lack of appetite",
        "not hungry", "unable to eat", "early satiety", "early satiation",
        "poor oral intake", "decreased oral intake", "not tolerating po"
    ],
    "difficulty_concentrating": [
        "difficulty concentrating", "trouble concentrating", "brain fog",
        "cognitive impairment", "confusion", "confused", "forgetful",
        "forgetfulness", "memory loss", "poor concentration", "mental fog",
        "altered mental status", "encephalopathy", "cognitive decline"
    ],
    "feeling_bloated": [
        "bloated", "bloating", "abdominal distension", "distended abdomen",
        "distended", "distension", "fullness", "feeling full", "abdominal fullness",
        "gassiness", "feeling bloated"
    ],
    "change_in_the_way_food_tastes": [
        "dysgeusia", "taste change", "taste changes", "altered taste",
        "food tastes different", "metallic taste", "taste disturbance",
        "loss of taste", "ageusia", "hypogeusia"
    ],
    "numbness_tingling_in_hands_feet": [
        "numbness", "numb", "tingling", "paresthesia", "paresthesias",
        "peripheral neuropathy", "neuropathy", "pins and needles",
        "burning sensation", "hand numbness", "foot numbness",
        "sensory changes", "sensory loss", "decreased sensation"
    ],
    "constipation": [
        "constipation", "constipated", "hard stool", "hard stools",
        "straining to defecate", "straining with bowel", "infrequent bowel",
        "no bowel movement", "unable to defecate", "obstipation",
        "difficulty passing stool"
    ],
    "cough": [
        r"\bcough\b", "coughing", "dry cough", "productive cough",
        "chronic cough", "persistent cough", "hacking cough"
    ],
    "i_dont_look_like_myself": [
        "does not look like himself", "does not look like herself",
        "doesn't look like himself", "doesn't look like herself",
        "appearance changed", "looks different", "not look like myself"
    ],
    "itching": [
        "pruritus", "itching", r"\bitch\b", "itchy", "scratching",
        "generalized itching", "skin itching"
    ],
    "swelling_of_arms_or_legs": [
        "edema", "swelling", "swollen", "pitting edema", "peripheral edema",
        "leg swelling", "ankle swelling", "arm swelling", "bilateral edema",
        "lymphedema", "lower extremity edema", "upper extremity edema",
        "fluid retention", "swollen legs", "swollen ankles", "swollen feet"
    ],
    "weight_loss": [
        "weight loss", "losing weight", "lost weight", "unintentional weight loss",
        "weight decreased", "losing pounds", "cachexia", "wasting",
        "significant weight loss", "intentional weight loss"
    ],
    "diarrhoea": [
        "diarrhea", "diarrhoea", "loose stool", "loose stools", "watery stool",
        "watery stools", "frequent loose", "watery bowel", "liquid stool",
        "liquid stools", "profuse diarrhea", "loose bowel", "soft stool"
    ],
    "dizziness": [
        "dizziness", "dizzy", "lightheaded", "light-headed", "lightheadedness",
        "vertigo", "unsteady", "unsteadiness", "pre-syncopal", "presyncopal",
        "near fainting", "near syncope", "room spinning"
    ],
    "problems_with_sexual_interest_or_activity": [
        "sexual dysfunction", "decreased libido", "low libido", "loss of libido",
        "sexual interest", "sexual activity", "impotence"
    ],
    "shortness_of_breath": [
        "shortness of breath", "sob", "dyspnea", "dyspnoea", "breathless",
        "breathlessness", "difficulty breathing", "trouble breathing",
        "respiratory distress", "air hunger", "winded", "worsening dyspnea",
        "exertional dyspnea", "orthopnea"
    ],
    "vomiting": [
        "vomiting", r"\bvomit\b", "vomited", "emesis", "throwing up",
        "threw up", "retching", r"\bretch\b", "hematemesis", "bile vomiting",
        "projectile vomiting", "nausea and vomiting", "n/v"
    ],
    "hair_loss": [
        "hair loss", "alopecia", "losing hair", "thinning hair",
        "hair thinning", "bald", "baldness"
    ],
    "problems_with_urination": [
        "urinary symptoms", "urinary problem", "difficulty urinating",
        "trouble urinating", "unable to urinate", "urinary retention",
        "voiding difficulty", "weak urine stream", "incomplete bladder emptying"
    ],
    "mouth_sores": [
        "mouth sores", "oral ulcer", "oral ulcers", "mucositis", "stomatitis",
        "canker sore", "canker sores", "oral lesion", "painful mouth",
        "mouth pain", "aphthous"
    ],
    "difficulty_swallowing": [
        "dysphagia", "difficulty swallowing", "trouble swallowing",
        "unable to swallow", "odynophagia", "painful swallowing",
        "swallowing difficulty", "can't swallow", "globus"
    ],
    "changes_in_skin": [
        "rash", "skin rash", "skin change", "skin changes", "skin lesion",
        "skin discoloration", "erythema", "jaundice", "yellowing",
        "skin breakdown", "dermatitis", "pruritic rash", "maculopapular"
    ],
    "sweats": [
        "sweating", "diaphoresis", "diaphoretic", "night sweats",
        "drenching sweats", "hyperhidrosis", "profuse sweating",
        "sweaty", "hot flashes", "flushing"
    ],
    "urinary_frequency_day": [
        "urinary frequency", "frequent urination", "polyuria",
        "urinating frequently", "frequent urge to urinate",
        "increased urinary frequency", "overactive bladder"
    ],
    "urinary_frequency_night": [
        "nocturia", "waking up to urinate", "nocturnal urination",
        "urinating at night", "getting up to void", "nighttime urination"
    ],
    "urinary_incontinence": [
        "urinary incontinence", "urine leakage", "leaking urine",
        "bladder leakage", "loss of bladder control", "incontinence of urine",
        "urge incontinence", "stress incontinence"
    ],
    "dysuria": [
        "dysuria", "painful urination", "burning urination", "burning with urination",
        "pain with urination", "pain on urination", "stinging urination"
    ],
    "abdominal_pain": [
        "abdominal pain", "stomach pain", "belly pain", "abd pain",
        "epigastric pain", "periumbilical pain", "lower abdominal pain",
        "upper abdominal pain", "abdominal cramping", "abdominal cramps",
        "abdominal discomfort", "abdominal tenderness", "right lower quadrant",
        "left lower quadrant", "rlq pain", "llq pain", "peritoneal"
    ],
    "anal_rectal_pain": [
        "rectal pain", "anal pain", "anorectal pain", "pain in rectum",
        "pain in anus", "rectal discomfort", "anal discomfort",
        "proctalgia", "tenesmus", "rectal pressure", "perianal pain"
    ],
    "blood_in_stool": [
        "blood in stool", "bloody stool", "hematochezia", "melena",
        "rectal bleeding", "bright red blood", "brbpr",
        "blood per rectum", "bloody diarrhea", "blood in bowel",
        "hemoccult positive", "guaiac positive", "occult blood",
        "blood mixed with stool", "bleeding per rectum"
    ],
    "mucus_in_stool": [
        # original phrases
        "mucus in stool", "mucoid stool", "mucus with stool",
        "slimy stool", "mucousy stool", "mucous in stool",
        "mucus discharge per rectum",
        # expanded from actual clinical text variants in GT notes
        "mucusy",                   # "mucusy discharge"
        "mucous per rectum",        # "passing pink bloody mucous per rectum"
        "mucus per rectum",         # variant
        "mucus stools",             # "frequent mucus stools"
        "mucous stools",            # variant
        "mucous clot",              # "blood/mucous clot"
        "mucus and blood",          # "mucus and blood on her stool"
        "blood and mucus",          # reverse order
        "with mucus",               # "slightly loose and with mucus"
        "mucous in her stool",      # direct match
        "mucous in his stool",      # variant
    ],
    "flatulence_gas": [
        "flatulence", "flatus", r"\bgas\b", "gassy", "excessive gas",
        "passing gas", "belching", "burping", "eructation",
        "abdominal gas", "intestinal gas"
    ],
    "stool_leakage": [
        "fecal incontinence", "stool leakage", "bowel incontinence",
        "leaking stool", "losing stool", "loss of bowel control",
        "anal incontinence", "soiling", "accidental bowel leakage"
    ],
    "sore_skin_around_stoma_anal_area": [
        "peristomal skin", "stoma site", "around stoma", "stoma irritation",
        "perianal skin", "perianal irritation", "skin around stoma",
        "stoma wound", "peristomal wound"
    ],
    "frequent_bowel_movements": [
        "frequent bowel movements", "increased bowel frequency",
        "multiple bowel movements", "bowel frequency", "frequent stools",
        "several bowel movements", "many bowel movements"
    ],
    "erectile_dysfunction": [
        "erectile dysfunction", "impotence", r"\b(?:ed)\b.*(?:sexual|erect)",
        "unable to achieve erection", "erection problem"
    ],
    "dyspareunia": [
        "dyspareunia", "painful intercourse", "painful sex",
        "pain during sex", "pain with intercourse", "pain during intercourse"
    ]
}
