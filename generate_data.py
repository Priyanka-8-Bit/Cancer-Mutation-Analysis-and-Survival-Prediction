# generate_data.py
# Generates synthetic ALIVE patients + keeps Dead patients
# Creates balanced Dead vs Alive dataset
# Usage: python3 generate_data.py

import pandas as pd
import numpy as np

print("=" * 55)
print("  SYNTHETIC DATA GENERATOR — DEAD + ALIVE")
print("=" * 55)

# ── Load original dataset ──
try:
    original_df = pd.read_csv("data/survival_original_backup.csv")
    print(f"\nOriginal backup loaded: {len(original_df)} patients")
except FileNotFoundError:
    original_df = pd.read_csv("data/survival.csv")
    print(f"\nOriginal dataset loaded: {len(original_df)} patients")

# ── Get gene columns ──
gene_cols = []
for col in original_df.columns:
    if col in ['ID', 'Age at Diagnosis ',
               'Life', 'Race', 'Gender',
               'Days Till Death']:
        continue
    unique_vals = set(original_df[col].dropna().unique())
    if unique_vals.issubset({0, 1}):
        gene_cols.append(col)

print(f"Gene columns found: {len(gene_cols)}")

# ── Real statistics from original Dead patients ──
age_mean    = original_df['Age at Diagnosis '].mean()
age_std     = original_df['Age at Diagnosis '].std()
age_min     = original_df['Age at Diagnosis '].min()
age_max     = original_df['Age at Diagnosis '].max()

days_mean   = original_df['Days Till Death'].mean()
days_std    = original_df['Days Till Death'].std()
days_min    = original_df['Days Till Death'].min()
days_max    = original_df['Days Till Death'].max()

gene_probs  = original_df[gene_cols].mean()
race_dist   = original_df['Race'].value_counts(normalize=True)
gender_dist = original_df['Gender'].value_counts(normalize=True)

print(f"\n Original Dead Patient Statistics:")
print(f"   Age    : {age_mean:.1f} ± {age_std:.1f} years")
print(f"   Days   : {days_mean:.1f} ± {days_std:.1f}")

np.random.seed(42)

# ══════════════════════════════════════════
# PART 1: Generate Synthetic DEAD Patients
# ══════════════════════════════════════════
# Keep original 300 + add 200 more dead patients
# for better coverage of dead patient patterns

print(f"\n── Generating Synthetic Dead Patients ──")
N_DEAD = 200

dead_rows = []
for i in range(N_DEAD):

    # Age — same distribution as real dead patients
    age = np.random.normal(age_mean, age_std)
    age = np.clip(age, age_min, age_max)

    # Gender and Race — same distribution
    gender = np.random.choice(gender_dist.index, p=gender_dist.values)
    race   = np.random.choice(race_dist.index,   p=race_dist.values)

    # Days Till Death — dead patients die within dataset range
    days = np.random.lognormal(
        mean=np.log(days_mean),
        sigma=0.8
    )
    days = np.clip(days, days_min, days_max)

    # Gene mutations — dead patients have higher mutation burden
    gene_values = {}
    for gene in gene_cols:
        prob = gene_probs[gene] * 1.1   # slightly higher mutations
        prob = np.clip(prob, 0, 1)
        gene_values[gene] = int(np.random.random() < prob)

    row = {
        'ID'               : f"syn_dead_{i:04d}",
        'Age at Diagnosis ': round(age, 1),
        'Life'             : 'Dead',
        'Race'             : race,
        'Gender'           : gender,
        'Days Till Death'  : round(days)
    }
    row.update(gene_values)
    dead_rows.append(row)

dead_synthetic_df = pd.DataFrame(dead_rows)
print(f"Synthetic dead patients: {len(dead_synthetic_df)}")


# ══════════════════════════════════════════
# PART 2: Generate Synthetic ALIVE Patients
# ══════════════════════════════════════════
# Alive patients have different characteristics:
# - Younger age at diagnosis
# - Longer survival (Days Till Death = 0 or very high)
# - Lower mutation burden
# - More likely to be in better health

print(f"\n── Generating Synthetic Alive Patients ──")
N_ALIVE = 300

alive_rows = []
for i in range(N_ALIVE):

    # Alive patients tend to be younger
    # Glioblastoma survivors are typically diagnosed younger
    age = np.random.normal(age_mean - 8, age_std)
    age = np.clip(age, age_min, age_max)

    # Gender and Race — same distribution
    gender = np.random.choice(gender_dist.index, p=gender_dist.values)
    race   = np.random.choice(race_dist.index,   p=race_dist.values)

    # Alive patients have longer survival
    # We set Days Till Death to a high value
    # representing they are still alive at last follow-up
    days = np.random.normal(1200, 400)
    days = np.clip(days, 600, 3881)

    # Gene mutations — alive patients have LOWER mutation burden
    # This is biologically realistic:
    # fewer harmful mutations = better prognosis
    gene_values = {}
    for gene in gene_cols:
        prob = gene_probs[gene] * 0.7   # 30% lower mutation rate
        prob = np.clip(prob, 0, 1)
        gene_values[gene] = int(np.random.random() < prob)

    row = {
        'ID'               : f"syn_alive_{i:04d}",
        'Age at Diagnosis ': round(age, 1),
        'Life'             : 'Alive',
        'Race'             : race,
        'Gender'           : gender,
        'Days Till Death'  : round(days)
    }
    row.update(gene_values)
    alive_rows.append(row)

alive_df = pd.DataFrame(alive_rows)
print(f"Synthetic alive patients: {len(alive_df)}")
print(f"\n Alive Patient Statistics:")
print(f"   Age    : {alive_df['Age at Diagnosis '].mean():.1f} ± "
      f"{alive_df['Age at Diagnosis '].std():.1f} years")
print(f"   Days   : {alive_df['Days Till Death'].mean():.1f} ± "
      f"{alive_df['Days Till Death'].std():.1f}")


# ══════════════════════════════════════════
# PART 3: Combine All Data
# ══════════════════════════════════════════

print(f"\n── Combining All Datasets ──")

# Make sure all DataFrames have same columns
dead_synthetic_df = dead_synthetic_df[original_df.columns]
alive_df          = alive_df[original_df.columns]

# Combine original + synthetic dead + synthetic alive
combined_df = pd.concat(
    [original_df, dead_synthetic_df, alive_df],
    ignore_index=True
)

# Shuffle the dataset
combined_df = combined_df.sample(
    frac=1, random_state=42
).reset_index(drop=True)

print(f"\n Final Dataset Breakdown:")
print(f"   Original Dead patients    : {len(original_df)}")
print(f"   Synthetic Dead patients   : {len(dead_synthetic_df)}")
print(f"   Synthetic Alive patients  : {len(alive_df)}")
print(f"   ─────────────────────────────")
print(f"   Total patients            : {len(combined_df)}")
print(f"   Dead  : {(combined_df['Life']=='Dead').sum()} "
      f"({(combined_df['Life']=='Dead').mean()*100:.1f}%)")
print(f"   Alive : {(combined_df['Life']=='Alive').sum()} "
      f"({(combined_df['Life']=='Alive').mean()*100:.1f}%)")

# Save combined dataset
combined_df.to_csv("data/survival.csv", index=False)
print(f"\nCombined dataset saved: data/survival.csv")
print(f"\n Done! Now run: python3 main.py")