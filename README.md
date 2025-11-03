# Detecting-PSI
The code to detect PSI online in Python: Based in theory
# broadened_pass.py
# High-recall, slang-aware detection of parasocial / fandom / shielding cues.

import pandas as pd
import re
from pathlib import Path

# ---- Load ----
INPUT = Path('/mnt/data/carditweets.csv')  # change if needed
df = pd.read_csv(INPUT)
df.columns = [c.strip() for c in df.columns]

# Pick a text column heuristically
TEXT_CANDIDATES = ['clean_text','text','tweet','tweet_text','full_text','content','body','message']
text_col = next((c for c in TEXT_CANDIDATES if c in df.columns), None)
if text_col is None:
    df['clean_text'] = ""
    text_col = 'clean_text'

# ---- Normalize ----
def normalize_text(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r'https?://\S+|www\.\S+', ' ', s)  # URLs
    s = re.sub(r'@\w+', ' ', s)                  # mentions
    s = s.replace('#', ' ')                      # keep tokens, drop '#'
    s = re.sub(r'\s+', ' ', s).strip()
    return s

df['norm_text'] = df[text_col].map(normalize_text)

# ---- Expanded lexicons (ASCII apostrophes only) ----
# Keep these modest to avoid overfitting; add more phrases as you iterate.

parasocial_patterns = {
    'affective': [
        'i love her','i love cardi','i care about','i feel for','my heart',
        'she makes me','so proud of her','im proud of her','she deserves happiness',
        'she dont even know me but','she does not even know me but',
        'protect cardi at all costs','i miss her','she my queen','shes my queen'
    ],
    'empathy': [
        'poor cardi','let her heal','shes human','we all make mistakes',
        'she owned up','shes real for that','shes honest','she told the truth'
    ]
}

fandom_patterns = {
    'collective_identity': [
        'we stan','bardigang','real fans know','we ride for cardi','we protect our own',
        'our queen','we got her back','we support her no matter what','true fans',
        'the fandom','she represents us'
    ],
    'boundary_policing': [
        'fake fans','yall not real fans','pick a side','bandwagon',
        'yall flip quick','stay out the fandom','real ones know',
        'yall switch up','you wasnt here before','you werent here before',
        'outsiders dont get it'
    ]
}

shielding_patterns = {
    'contextual_defense': [
        'she did what she had to do','thats survival','yall dont know the struggle',
        'she came from nothing','she from the hood','stop comparing her',
        'not the same as','men do worse','context matters','back then was different',
        'she apologized already','she apologised already','let her live'
    ],
    'counterattack': [
        'yall mad','yall reaching','keep that same energy','where was this energy',
        'yall weird','yall hating','yall always tear down women',
        'protect black women','lets talk about the men','you just hate her',
        'this hate is weird'
    ]
}

# ---- Detection helpers ----
def detect_patterns(text: str, patterns: dict) -> list:
    found = []
    for cat, phrases in patterns.items():
        for p in phrases:
            if p in text:
                found.append(cat)
                break  # one hit per category is enough
    return found

# ---- Apply detection (lenient, includes RTs/quotes/replies) ----
df['parasocial_hits'] = df['norm_text'].apply(lambda x: detect_patterns(x, parasocial_patterns))
df['fandom_hits']     = df['norm_text'].apply(lambda x: detect_patterns(x, fandom_patterns))
df['shielding_hits']  = df['norm_text'].apply(lambda x: detect_patterns(x, shielding_patterns))

df['has_parasocial'] = df['parasocial_hits'].str.len() > 0
df['has_fandom']     = df['fandom_hits'].str.len() > 0
df['has_shielding']  = df['shielding_hits'].str.len() > 0

# ---- Summary counts & percentages ----
total = len(df)
parasocial_any = int(df['has_parasocial'].sum())
fandom_any = int(df['has_fandom'].sum())
shielding_any = int(df['has_shielding'].sum())

summary = {
    "total_rows": total,
    "parasocial_any": parasocial_any,
    "parasocial_pct": round(100*parasocial_any/total, 2),
    "fandom_any": fandom_any,
    "fandom_pct": round(100*fandom_any/total, 2),
    "shielding_any": shielding_any,
    "shielding_pct": round(100*shielding_any/total, 2),
    "overlap_pf": int((df['has_parasocial'] & df['has_fandom']).sum()),
    "overlap_ps": int((df['has_parasocial'] & df['has_shielding']).sum()),
    "overlap_fs": int((df['has_fandom'] & df['has_shielding']).sum()),
}
print("=== Broadened Pass Summary ===")
for k, v in summary.items():
    print(f"{k}: {v}")

# ---- Optional: save per-tweet flags for auditing ----
OUT = INPUT.with_name('carditweets_broadened_scored.csv')
df.to_csv(OUT, index=False)
print(f"\nSaved scored file to: {OUT}")
