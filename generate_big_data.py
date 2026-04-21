import pandas as pd
import numpy as np
 
random = np.random.default_rng(42)
 
NUM_ROWS = 2000
 
SYMPTOMS = [
    "fatigue", "pale_skin", "dry_skin", "bone_pain",
    "vision_blur", "mouth_ulcers", "bleeding_gums",
    "muscle_cramps", "headache", "numbness_tingling",
    "brittle_nails", "easy_bruising", "hair_loss",
    "swollen_joints", "night_sweats"
]
 
PROFILES = {
    "Iron Deficiency": {
        "primary":       ["fatigue", "pale_skin"],
        "secondary":     ["dry_skin", "mouth_ulcers", "hair_loss"],
        "primary_prob":   0.85,
        "secondary_prob": 0.30,
    },
    "Vitamin A Deficiency": {
        "primary":       ["dry_skin", "vision_blur"],
        "secondary":     ["fatigue", "hair_loss"],
        "primary_prob":   0.85,
        "secondary_prob": 0.25,
    },
    "Vitamin D Deficiency": {
        "primary":       ["bone_pain", "fatigue"],
        "secondary":     ["dry_skin", "muscle_cramps", "night_sweats"],
        "primary_prob":   0.85,
        "secondary_prob": 0.30,
    },
    "Vitamin B12 Deficiency": {
        "primary":       ["mouth_ulcers", "fatigue"],
        "secondary":     ["pale_skin", "numbness_tingling"],
        "primary_prob":   0.85,
        "secondary_prob": 0.25,
    },
    "Vitamin C Deficiency": {
        "primary":       ["bleeding_gums", "mouth_ulcers"],
        "secondary":     ["fatigue", "dry_skin", "easy_bruising"],
        "primary_prob":   0.85,
        "secondary_prob": 0.20,
    },
    "Vitamin K Deficiency": {
        "primary":       ["bleeding_gums", "easy_bruising"],
        "secondary":     ["bone_pain", "swollen_joints"],
        "primary_prob":   0.85,
        "secondary_prob": 0.25,
    },
    "Vitamin B6 Deficiency": {
        "primary":       ["mouth_ulcers", "numbness_tingling"],
        "secondary":     ["fatigue", "dry_skin", "headache"],
        "primary_prob":   0.85,
        "secondary_prob": 0.25,
    },
    "Magnesium Deficiency": {
        "primary":       ["muscle_cramps", "fatigue"],
        "secondary":     ["numbness_tingling", "headache", "night_sweats"],
        "primary_prob":   0.85,
        "secondary_prob": 0.30,
    },
    "Calcium Deficiency": {
        "primary":       ["bone_pain", "muscle_cramps"],
        "secondary":     ["brittle_nails", "dry_skin", "numbness_tingling"],
        "primary_prob":   0.85,
        "secondary_prob": 0.25,
    },
    "Folate B9 Deficiency": {
        "primary":       ["fatigue", "pale_skin"],
        "secondary":     ["mouth_ulcers", "headache", "hair_loss"],
        "primary_prob":   0.85,
        "secondary_prob": 0.25,
    },
    "Healthy": {
        "primary":       [],
        "secondary":     ["fatigue", "headache"],
        "primary_prob":   0.0,
        "secondary_prob": 0.08,
    },
}
 
CLASSES = list(PROFILES.keys())
data = []
 
for _ in range(NUM_ROWS):
    diagnosis = random.choice(CLASSES)
    profile   = PROFILES[diagnosis]
    row       = {s: 0 for s in SYMPTOMS}
 
    for s in profile["primary"]:
        if random.random() < profile["primary_prob"]:
            row[s] = 1
 
    for s in profile["secondary"]:
        if random.random() < profile["secondary_prob"]:
            row[s] = 1
 
    # 5% random noise on any symptom
    for s in SYMPTOMS:
        if row[s] == 0 and random.random() < 0.05:
            row[s] = 1
 
    row["disease_diagnosis"] = diagnosis
    data.append(row)
 
df = pd.DataFrame(data)
df.to_csv("vitamin_dataset.csv", index=False)
 
print("=" * 55)
print("  Dataset generated: vitamin_dataset.csv")
print("=" * 55)
print(f"  Rows     : {df.shape[0]}")
print(f"  Features : {df.shape[1] - 1} symptoms")
print(f"  Classes  : {len(CLASSES)} conditions")
print()
print("  Class distribution:")
for cls, count in df["disease_diagnosis"].value_counts().items():
    print(f"    {cls:<30} {count} rows")
print()
print("  Symptom frequency (% of rows with symptom=1):")
for sym, freq in (df[SYMPTOMS].mean() * 100).items():
    print(f"    {sym:<22} {freq:.1f}%")
print()
print("  Next step: run train_models.py")
 