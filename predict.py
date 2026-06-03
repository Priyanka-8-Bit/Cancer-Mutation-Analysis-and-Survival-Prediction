# predict.py
# Predict survival for a new patient
# Usage: python3 predict.py

import numpy as np
import pandas as pd
from gtts import gTTS
from ml.utils import load_model, print_section, print_step


def get_patient_input():
    """Get new patient data from user input"""
    print_section("NEW PATIENT SURVIVAL PREDICTION")
    print("\nEnter patient details below:")
    print("(Press Enter to use default/average values)\n")

    # Age
    age_input = input("Age at Diagnosis (default=55): ").strip()
    age       = float(age_input) if age_input else 55.0

    # Gender
    print("\nGender options: Male / Female")
    gender_input = input("Gender (default=Male): ").strip().lower()
    gender       = gender_input if gender_input in ['male', 'female'] else 'male'

    # Race
    print("\nRace options:")
    print("  1. Caucasian")
    print("  2. African American")
    print("  3. Asian")
    print("  4. Other")
    race_map = {
        '1'               : 'Caucasian',
        '2'               : 'African American',
        '3'               : 'Asian',
        '4'               : 'Other',
        'caucasian'       : 'Caucasian',
        'african american': 'African American',
        'asian'           : 'Asian',
        'other'           : 'Other'
    }
    race_input = input("Race (default=Caucasian): ").strip().lower()
    race       = race_map.get(race_input, 'Caucasian')

    # Load overlap genes list
    try:
        overlap_df    = pd.read_csv("outputs/results/harmful_genes.csv")
        overlap_genes = overlap_df['Gene_Symbol'].tolist()
    except FileNotFoundError:
        print(" Run main.py first to generate harmful genes list!")
        return None

    # Gene mutations
    print(f"\nEnter gene mutation status (1=mutated, 0=not mutated)")
    print("Press Enter to skip (defaults to 0)\n")
    print(f"There are {len(overlap_genes)} harmful genes to check.")

    check_all = input(
        "Do you want to enter each gene? (y/n, default=n): "
    ).strip().lower()

    gene_mutations = {}
    if check_all == 'y':
        for gene in overlap_genes:
            val = input(f"  {gene} mutated? (1/0, default=0): ").strip()
            gene_mutations[gene] = int(val) if val in ['0', '1'] else 0
    else:
        print("Using average mutation score from dataset.")
        gene_mutations = {g: 0 for g in overlap_genes}

    return {
        'age'           : age,
        'gender'        : gender,
        'race'          : race,
        'gene_mutations': gene_mutations,
        'overlap_genes' : overlap_genes
    }


def predict_survival(patient):
    """Run prediction for a new patient"""
    print_step("Running Prediction")

    # ── Load all saved models ──
    try:
        scaler     = load_model("survival_scaler.pkl")
        pca        = load_model("pca.pkl")
        regressor  = load_model("regression_model.pkl")
        classifier = load_model("classifier.pkl")
        le_gender  = load_model("label_encoder_gender.pkl")
        le_race    = load_model("label_encoder_race.pkl")
    except FileNotFoundError:
        print(" Models not found! Run main.py first.")
        return None

    # ── Load survival data ──
    surv_df    = pd.read_csv("data/survival.csv")
    surv_clean = surv_df.drop(
        columns=['ID', 'Life', 'Days Till Death']
    ).copy()
    surv_clean.fillna(surv_clean.median(numeric_only=True), inplace=True)
    surv_clean['Gender'] = le_gender.transform(
        surv_clean['Gender'].str.strip()
    )
    surv_clean['Race'] = le_race.transform(
        surv_clean['Race'].str.strip()
    )

    # ── Compute Mutation Score ──
    # Only use genes that actually exist in survival dataset
    overlap_genes  = patient['overlap_genes']
    valid_genes    = [g for g in overlap_genes if g in surv_df.columns]

    # If user entered gene values use them, else use dataset average
    if all(v == 0 for v in patient['gene_mutations'].values()):
        # No genes entered — use average mutation score from dataset
        mutation_score = surv_df[valid_genes].sum(axis=1).mean()
    else:
        mutation_score = sum(
            patient['gene_mutations'].get(g, 0)
            for g in valid_genes
        )

    print(f"   Mutation Score : {mutation_score:.1f}")

    # ── Build patient feature row ──
    new_row = surv_clean.iloc[0:1].copy() * 0

    # Fill patient values
    new_row['Age at Diagnosis '] = patient['age']
    new_row['Mutation_Score']    = mutation_score

    # Encode gender safely
    gender_str = patient['gender'].capitalize()
    if gender_str in le_gender.classes_:
        new_row['Gender'] = le_gender.transform([gender_str])[0]
    else:
        new_row['Gender'] = 0

    # Encode race safely
    race_str = patient['race'].strip()
    if race_str in le_race.classes_:
        new_row['Race'] = le_race.transform([race_str])[0]
    else:
        new_row['Race'] = 0

    # ── Scale ──
    new_scaled = scaler.transform(new_row)

    # ── PCA ──
    new_pca = pca.transform(new_scaled)

    # ── Add interaction feature ──
    # Classifier was trained with 95 features:
    # 94 PCA components + 1 interaction (Age x Mutation_Score)
    # We must recreate this exact feature here
    age_mean    = surv_df['Age at Diagnosis '].mean()
    age_std     = surv_df['Age at Diagnosis '].std() + 1e-8
    mut_scores  = surv_df[valid_genes].sum(axis=1)
    mut_mean    = mut_scores.mean()
    mut_std     = mut_scores.std() + 1e-8

    age_norm    = (patient['age'] - age_mean) / age_std
    mut_norm    = (mutation_score - mut_mean) / mut_std
    interaction = np.array([[age_norm * mut_norm]])

    # Final feature vector for classifier
    new_pca_enhanced = np.hstack([new_pca, interaction])

    # ── Predict survival days (regression) ──
    predicted_days   = max(0, regressor.predict(new_pca)[0])
    predicted_months = predicted_days / 30.44

    # ── Predict survival group (classification) ──
    group_pred  = classifier.predict(new_pca_enhanced)[0]
    group_names = {
        0: 'Short Survival  (< 6 months)',
        1: 'Long Survival   (> 18 months)'
    }
    group_label = group_names.get(int(group_pred), 'Unknown')

    return {
        'age'           : patient['age'],
        'gender'        : patient['gender'].capitalize(),
        'race'          : patient['race'],
        'mutation_score': mutation_score,
        'days'          : predicted_days,
        'months'        : predicted_months,
        'group'         : group_label
    }


def print_report(result):
    """Print prediction report"""
    print("\n" + "─" * 55)
    print("       PATIENT SURVIVAL PREDICTION REPORT")
    print("─" * 55)
    print(f"  Age at Diagnosis  : {result['age']:.0f} years")
    print(f"  Gender            : {result['gender']}")
    print(f"  Race              : {result['race']}")
    print(f"  Mutation Score    : {result['mutation_score']:.1f} harmful genes mutated")
    print(f"  Predicted Days    : {result['days']:.0f} days")
    print(f"  Predicted Months  : {result['months']:.1f} months")
    print(f"  Survival Group    : {result['group']}")
    print("─" * 55)


def generate_audio(result):
    """Generate audio report using gTTS"""
    print_step("Generating Audio Report")

    text = f"""
    Patient Survival Prediction Report.
    Age at diagnosis: {result['age']:.0f} years.
    Gender: {result['gender']}.
    Race: {result['race']}.
    Mutation Score: {result['mutation_score']:.1f},
    meaning {result['mutation_score']:.0f} harmful genes
    are mutated in this patient.
    Predicted survival time: {result['days']:.0f} days,
    which is approximately {result['months']:.1f} months.
    Survival group prediction: {result['group']}.
    End of report.
    """

    try:
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save("outputs/prediction_audio.mp3")
        print("   Audio saved : outputs/prediction_audio.mp3")
        print("   Open the file to listen!")
    except Exception as e:
        print(f"   Audio error: {e}")
        print("   Check internet connection — gTTS needs internet")


def main():
    # Step 1: Get patient data
    patient = get_patient_input()
    if patient is None:
        return

    # Step 2: Predict
    result = predict_survival(patient)
    if result is None:
        return

    # Step 3: Print report
    print_report(result)

    # Step 4: Audio
    audio = input("\nGenerate audio report? (y/n): ").strip().lower()
    if audio == 'y':
        generate_audio(result)

    print("\n Prediction complete!\n")


if __name__ == "__main__":
    main()