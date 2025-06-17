import pandas as pd
import os

# Paths
merged_raw_path = "data/raw/merged.csv"
demo_path = "data/raw/DEMO_L.csv"
output_dir = "data/seqn_demo"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# Load merged and DEMO files
# -----------------------------
merged = pd.read_csv(merged_raw_path, dtype={"SEQN": str})
demo = pd.read_csv(demo_path, dtype={"SEQN": str})

# -----------------------------
# Filter merged like you did for merged_clean
# -----------------------------
filtered = merged[(merged["RIDAGEYR"] >= 18) & (merged["RIDSTATR"] == 2)].copy()

# PHQ-9 filter: at least 6 valid responses
phq9_items = [
    "DPQ010", "DPQ020", "DPQ030", "DPQ040", "DPQ050",
    "DPQ060", "DPQ070", "DPQ080", "DPQ090"
]
valid_counts = filtered[phq9_items].isin([0, 1, 2, 3]).sum(axis=1)
filtered = filtered[valid_counts >= 6].copy()

# Save SEQN list to ensure alignment
seqns = filtered["SEQN"]

# -----------------------------
# Create ul_seqn_demo.csv
# (original filtered version + demo)
# -----------------------------
ul = filtered.merge(demo, on="SEQN", how="left")
ul.to_csv(os.path.join(output_dir, "ul_seqn_demo.csv"), index=False)

# -----------------------------
# Create sl_seqn_demo.csv
# (apply modeling prep logic to ul)
# -----------------------------
sl = ul.copy()

# Recode yes/no binary
binary_vars = [
    'Covered by health insurance',
    'Time when no insurance in past year?',
    'Routine place to go for healthcare',
    'Past 12 months had video conf w/Dr?',
    'Seen mental health professional/past yr'
]
for col in binary_vars:
    if col in sl.columns:
        sl[col] = sl[col].map({1: 1, 2: 0})

# Insurance binary: filled = 1, blank = 0
insurance_cols = [
    'Covered by private insurance', 'Covered by Medicare',
    'Covered by Medi-Gap', 'Covered by Medicaid', 'Covered by CHIP',
    'Covered by military health care', 'Covered by state-sponsored health plan',
    'Covered by other government insurance'
]
for col in insurance_cols:
    if col in sl.columns:
        sl[col] = sl[col].notna().astype(int)

# Gender: 1 = male → 0, 2 = female → 1
if 'Gender' in sl.columns:
    sl['Gender'] = sl['Gender'].map({1: 0, 2: 1})

# Drop unused
sl = sl.drop(columns=[
    'Covered by CHIP', 'Covered by other government insurance'
], errors='ignore')

# Convert relevant cols to int
columns_to_convert = binary_vars + insurance_cols + [
    'Gender',
    'Education level - Adults 20+',
    'Difficulty these problems have caused',
    'Difficulty with self-care',
    'How often feel worried/nervous/anxious',
    'Level of feeling worried/nervous/anxious',
    'Type place most often go for healthcare',
    'Age in years at screening',
    'Total number of people in the Household'
]
for col in columns_to_convert:
    if col in sl.columns:
        sl[col] = sl[col].fillna(0).astype(int)

sl.to_csv(os.path.join(output_dir, "sl_seqn_demo.csv"), index=False)

print("✅ Done!")
print("→ data/seqn_demo/ul_seqn_demo.csv")
print("→ data/seqn_demo/sl_seqn_demo.csv")
