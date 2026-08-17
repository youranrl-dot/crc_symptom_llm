"""Negation-filter configuration for the hybrid variants.

Extracted verbatim from the project's run_hybrid_pipeline.py so that the hybrid
rows of Tables 2-3 are reproducible from this repository alone. Do not edit these
values without re-running compare_with_gold.py and updating the manuscript.
"""

import re

#: Characters before a matched keyword that are scanned for a negation cue.
WINDOW = 100

#: Negation cues. A keyword preceded by any of these within WINDOW characters
#: flips an LLM "present" prediction to "absent".
NEGATION_PATTERNS = re.compile(
    r"\b(no|not|without|denies|denied|deny|absent|absence|free of|negative for|"
    r"never|none|neither|nor|cannot|can't|did not|does not|do not|"
    r"resolved|resolving|improved|improving|no longer|no evidence of|"
    r"no complaint of|no complaints of)\b",
    re.IGNORECASE,
)

#: Trigger terms per symptom. Entries starting with \b are treated as regexes;
#: everything else is matched literally (case-insensitive).
SYMPTOM_KEYWORDS = {
    'lack_of_energy': ['fatigue', 'fatigued', 'tired', 'tiredness', 'exhausted', 'exhaustion', 'lethargy', 'lethargic', 'weakness', 'weak', 'no energy', 'low energy', 'lack of energy', 'worn out', 'malaise', 'asthenia'],
    'worrying': ['worry', 'worrying', 'worried', 'anxiety', 'anxious', 'apprehensive', 'apprehension', 'concerned', 'fearful', 'dread', 'panic'],
    'feeling_sad': ['sad', 'sadness', 'depressed', 'depression', 'low mood', 'tearful', 'crying', 'hopeless', 'despondent', 'dysphoria', 'dysphoric'],
    'pain': ['\\bpain\\b', '\\bpains\\b', 'ache', 'aching', 'hurt', 'hurting', 'discomfort', 'sore', 'soreness', 'tenderness', 'painful'],
    'feeling_nervous': ['nervous', 'nervousness', 'jittery', 'on edge', 'restless', 'tense', 'tension'],
    'feeling_drowsy': ['drowsy', 'drowsiness', 'somnolent', 'somnolence', 'sleepy', 'sleepiness', 'sedated', 'sedation'],
    'dry_mouth': ['dry mouth', 'xerostomia', 'mouth dryness', 'decreased saliva'],
    'difficulty_sleeping': ['insomnia', 'difficulty sleeping', 'trouble sleeping', "can't sleep", 'cannot sleep', 'sleep disturbance', 'poor sleep', 'sleeplessness'],
    'feeling_irritable': ['irritable', 'irritability', 'short tempered', 'mood swings', 'grumpy', 'impatient'],
    'nausea': ['nausea', 'nauseated', 'nauseous', 'queasy', 'sick to stomach', 'upset stomach'],
    'lack_of_appetite': ['anorexia', 'poor appetite', 'decreased appetite', 'loss of appetite', 'not eating', 'no appetite', 'early satiety', 'poor oral intake', 'decreased oral intake', 'not tolerating po'],
    'difficulty_concentrating': ['difficulty concentrating', 'brain fog', 'cognitive impairment', 'confusion', 'confused', 'forgetful', 'memory loss', 'poor concentration', 'altered mental status', 'encephalopathy'],
    'feeling_bloated': ['bloated', 'bloating', 'abdominal distension', 'distended abdomen', 'distended', 'distension', 'abdominal fullness'],
    'change_in_the_way_food_tastes': ['dysgeusia', 'taste change', 'taste changes', 'altered taste', 'metallic taste', 'loss of taste', 'ageusia'],
    'numbness_tingling_in_hands_feet': ['numbness', 'numb', 'tingling', 'paresthesia', 'paresthesias', 'peripheral neuropathy', 'neuropathy', 'pins and needles', 'burning sensation', 'decreased sensation'],
    'constipation': ['constipation', 'constipated', 'hard stool', 'hard stools', 'straining to defecate', 'no bowel movement', 'obstipation'],
    'cough': ['\\bcough\\b', 'coughing', 'dry cough', 'productive cough', 'persistent cough'],
    'i_dont_look_like_myself': ['does not look like himself', 'does not look like herself', 'appearance changed', 'looks different'],
    'itching': ['pruritus', 'itching', '\\bitch\\b', 'itchy', 'scratching'],
    'swelling_of_arms_or_legs': ['edema', 'swelling', 'swollen', 'pitting edema', 'peripheral edema', 'leg swelling', 'ankle swelling', 'arm swelling', 'lymphedema', 'lower extremity edema'],
    'weight_loss': ['weight loss', 'losing weight', 'lost weight', 'unintentional weight loss', 'cachexia', 'wasting'],
    'diarrhoea': ['diarrhea', 'diarrhoea', 'loose stool', 'loose stools', 'watery stool', 'watery stools', 'liquid stool'],
    'dizziness': ['dizziness', 'dizzy', 'lightheaded', 'light-headed', 'vertigo', 'unsteady', 'near syncope'],
    'problems_with_sexual_interest_or_activity': ['sexual dysfunction', 'decreased libido', 'low libido', 'loss of libido', 'sexual interest', 'impotence'],
    'shortness_of_breath': ['shortness of breath', 'sob', 'dyspnea', 'dyspnoea', 'breathless', 'breathlessness', 'difficulty breathing', 'orthopnea'],
    'vomiting': ['vomiting', '\\bvomit\\b', 'vomited', 'emesis', 'throwing up', 'threw up', 'retching', 'hematemesis', 'n/v'],
    'hair_loss': ['hair loss', 'alopecia', 'losing hair', 'thinning hair'],
    'problems_with_urination': ['difficulty urinating', 'trouble urinating', 'urinary retention', 'voiding difficulty', 'weak urine stream'],
    'mouth_sores': ['mouth sores', 'oral ulcer', 'oral ulcers', 'mucositis', 'stomatitis', 'aphthous'],
    'difficulty_swallowing': ['dysphagia', 'difficulty swallowing', 'trouble swallowing', 'odynophagia', 'painful swallowing'],
    'changes_in_skin': ['rash', 'skin rash', 'skin change', 'skin changes', 'erythema', 'jaundice', 'yellowing', 'dermatitis'],
    'sweats': ['sweating', 'diaphoresis', 'diaphoretic', 'night sweats', 'drenching sweats', 'hyperhidrosis', 'hot flashes'],
    'urinary_frequency_day': ['urinary frequency', 'frequent urination', 'polyuria', 'urinating frequently', 'overactive bladder'],
    'urinary_frequency_night': ['nocturia', 'waking up to urinate', 'nocturnal urination', 'nighttime urination'],
    'urinary_incontinence': ['urinary incontinence', 'urine leakage', 'leaking urine', 'bladder leakage', 'loss of bladder control'],
    'dysuria': ['dysuria', 'painful urination', 'burning urination', 'burning with urination', 'pain with urination'],
    'abdominal_pain': ['abdominal pain', 'stomach pain', 'belly pain', 'abd pain', 'epigastric pain', 'abdominal cramping', 'abdominal cramps', 'abdominal discomfort', 'abdominal tenderness'],
    'anal_rectal_pain': ['rectal pain', 'anal pain', 'anorectal pain', 'rectal discomfort', 'proctalgia', 'tenesmus', 'rectal pressure', 'perianal pain'],
    'blood_in_stool': ['blood in stool', 'bloody stool', 'hematochezia', 'melena', 'rectal bleeding', 'bright red blood', 'brbpr', 'blood per rectum', 'bloody diarrhea', 'guaiac positive'],
    'mucus_in_stool': ['mucus in stool', 'mucoid stool', 'mucus with stool', 'slimy stool', 'mucous in stool'],
    'flatulence_gas': ['flatulence', 'flatus', '\\bgas\\b', 'gassy', 'excessive gas', 'passing gas', 'belching', 'burping'],
    'stool_leakage': ['fecal incontinence', 'stool leakage', 'bowel incontinence', 'leaking stool', 'anal incontinence'],
    'sore_skin_around_stoma_anal_area': ['peristomal skin', 'stoma site', 'around stoma', 'stoma irritation', 'perianal skin', 'perianal irritation'],
    'frequent_bowel_movements': ['frequent bowel movements', 'increased bowel frequency', 'multiple bowel movements', 'bowel frequency', 'frequent stools'],
    'erectile_dysfunction': ['erectile dysfunction', 'unable to achieve erection', 'erection problem'],
    'dyspareunia': ['dyspareunia', 'painful intercourse', 'painful sex', 'pain during sex', 'pain with intercourse'],
}

assert len(SYMPTOM_KEYWORDS) == 46, len(SYMPTOM_KEYWORDS)