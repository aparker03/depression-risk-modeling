import pandas as pd
import os
import numpy as np

# Set paths
merged_raw_path = "data/raw/merged.csv"
output_dir = "data/clean"  # New target location
os.makedirs(output_dir, exist_ok=True)

# Load merged dataset (already includes DEMO)
df = pd.read_csv(merged_raw_path, dtype={"SEQN": str})

# Replace placeholder float with NaN
placeholder = 5.397605346934028e-79
df.replace(placeholder, np.nan, inplace=True)

# Filter: Adults (18+) who completed both interview and exam
df = df[(df["RIDAGEYR"] >= 18) & (df["RIDSTATR"] == 2)]

# Remove invalid codes: 7, 9, 77, 99
invalid_vals = [7, 9, 77, 99]
for col in df.columns:
    if df[col].dtype in [np.int64, np.float64]:
        df = df[~df[col].isin(invalid_vals)]

# Drop columns not used in modeling — keep all DEMO columns
drop_cols = [
    # Functioning (FNQ_L)
    "FNQ021", "FNQ041", "FNQ050", "FNQ060", "FNQ080", "FNQ100", "FNQ110", "FNQ120",
    "FNQ130", "FNQ140", "FNQ150", "FNQ160", "FNQ170", "FNQ180", "FNQ190", "FNQ200",
    "FNQ410", "FNQ430", "FNQ440", "FNQ450", "FNQ460", "FNQ480", "FNQ490", "FNQ530", "FNQ540",
    "FNDADI", "FNDAEDI", "FNDCDI",
    # Hospital Utilization and Access to Care (HUQ_L)
    "HUQ010", "HUQ085",
    # Income (INQ_L)
    "INQ300", "IND310", "INDFMMPC", "INDFMPIR",
    # Sleep (SLQ_L)
    "SLQ300", "SLQ310", "SLQ320", "SLQ330"
]
df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

# Rename columns for readability and modeling
column_mapping = {
    "DMDHHSIZ": "Total number of people in the Household",
    "DMDEDUC2": "Education level - Adults 20+",
    "RIAGENDR": "Gender",
    "RIDAGEYR": "Age in years at screening",
    "FNQ470": "Difficulty with self-care",
    "FNQ510": "How often feel worried/nervous/anxious",
    "FNQ520": "Level of feeling worried/nervous/anxious",
    "HUQ030": "Routine place to go for healthcare",
    "HUQ042": "Type place most often go for healthcare",
    "HUQ055": "Past 12 months had video conf w/Dr?",
    "HUQ090": "Seen mental health professional/past yr",
    "HIQ011": "Covered by health insurance",
    "HIQ032A": "Covered by private insurance",
    "HIQ032B": "Covered by Medicare",
    "HIQ032C": "Covered by Medi-Gap",
    "HIQ032D": "Covered by Medicaid",
    "HIQ032E": "Covered by CHIP",
    "HIQ032F": "Covered by military health care",
    "HIQ032H": "Covered by state-sponsored health plan",
    "HIQ032I": "Covered by other government insurance",
    "HIQ210": "Time when no insurance in past year?",
    "INDFMMPI": "Monthly poverty index",
    "SLD012": "Sleep hours - weekdays or workdays",
    "SLD013": "Sleep hours - weekends",
}
df.rename(columns=column_mapping, inplace=True)

# Filter: PHQ-9 valid (≥6 non-null answers)
phq9_items = [
    "DPQ010", "DPQ020", "DPQ030", "DPQ040", "DPQ050",
    "DPQ060", "DPQ070", "DPQ080", "DPQ090"
]
valid_mask = df[phq9_items].isin([0, 1, 2, 3])
valid_counts = valid_mask.sum(axis=1)
df = df[valid_counts >= 6].copy()

# Nullify invalid PHQ-9 responses
df.loc[:, phq9_items] = df[phq9_items].where(valid_mask)

# Compute total PHQ-9 score
df["PHQ9_TOTAL"] = df[phq9_items].sum(axis=1)

# Drop individual PHQ-9 items; rename DPQ100
df.drop(columns=phq9_items, inplace=True)
df.rename(columns={"DPQ100": "Difficulty these problems have caused"}, inplace=True)

# --- Apply recoding for modeling ---

# Binary yes/no variables (1 = yes; else = no)
binary_vars = [
    "Covered by health insurance",
    "Time when no insurance in past year?",
    "Routine place to go for healthcare",
    "Past 12 months had video conf w/Dr?",
    "Seen mental health professional/past yr"
]
for col in binary_vars:
    if col in df.columns:
        df[col] = df[col].map({1: 1, 2: 0})

# Insurance type flags: filled = 1, blank = 0
insurance_cols = [
    "Covered by private insurance", "Covered by Medicare",
    "Covered by Medi-Gap", "Covered by Medicaid", "Covered by CHIP",
    "Covered by military health care", "Covered by state-sponsored health plan",
    "Covered by other government insurance"
]
for col in insurance_cols:
    if col in df.columns:
        df[col] = df[col].notna().astype(int)

# Gender: 1 = male → 0, 2 = female → 1
if "Gender" in df.columns:
    df["Gender"] = df["Gender"].map({1: 0, 2: 1})

# Drop unused insurance columns
df.drop(columns=["Covered by CHIP", "Covered by other government insurance"], inplace=True, errors="ignore")

# Convert fields to int for modeling
columns_to_convert = binary_vars + insurance_cols + [
    "Gender",
    "Education level - Adults 20+",
    "Difficulty these problems have caused",
    "Difficulty with self-care",
    "How often feel worried/nervous/anxious",
    "Level of feeling worried/nervous/anxious",
    "Type place most often go for healthcare",
    "Age in years at screening",
    "Total number of people in the Household"
]
for col in columns_to_convert:
    if col in df.columns:
        df[col] = df[col].fillna(0).astype(int)

# Save to new location
output_path = os.path.join(output_dir, "sl_seqn_demo.csv")
df.to_csv(output_path, index=False)
print(f"✅ sl_seqn_demo.csv saved to {output_path}")
