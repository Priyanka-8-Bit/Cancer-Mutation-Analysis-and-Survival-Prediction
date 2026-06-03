# main.py
# Run the complete Cancer Mutation Analysis Pipeline
# Usage: python main.py

import os
import time
import pandas as pd

from ml.stage1_genes    import run_stage1
from ml.stage2_survival import run_stage2
from ml.utils           import print_section, RESULTS_DIR


def print_banner():
    print("""
╔══════════════════════════════════════════════════════╗
║     CANCER MUTATION ANALYSIS & SURVIVAL PREDICTION  ║
║     B.Tech TY CSE — ML Project                      ║
║     Two-Stage Genomic ML Pipeline                   ║
╚══════════════════════════════════════════════════════╝
    """)


def print_final_summary(harmful_genes, stage2):
    print_section("FINAL PROJECT SUMMARY")

    clf_path = os.path.join(RESULTS_DIR, "classification_results.csv")
    reg_path = os.path.join(RESULTS_DIR, "regression_results.csv")

    clf_df = pd.read_csv(clf_path)
    reg_df = pd.read_csv(reg_path)

    print(f"""
STAGE 1 — Gene Classification
  • Total genes analyzed  : 711
  • Harmful genes found   : {len(harmful_genes)}
  • Models trained        : KNN, NB, DT, SVM, RF, Bagging, AdaBoost

BRIDGE
  • Overlapping genes     : {len(stage2['overlap_genes'])}
  • Mutation Score range  : {stage2['df']['Mutation_Score'].min():.0f} — {stage2['df']['Mutation_Score'].max():.0f}

STAGE 2 — Survival Prediction
  • Total patients        : {stage2['df'].shape[0]}
  • Features before PCA   : {stage2['X'].shape[1]}
  • Features after PCA    : {stage2['X_pca'].shape[1]}
  • Best Classifier       : {stage2['best_clf']} ({stage2['clf_results'][stage2['best_clf']]['acc']:.2%})
  • Best Regressor        : {stage2['best_reg']} (R²={stage2['reg_results'][stage2['best_reg']]['R2']:.4f})

CLASSIFICATION RESULTS (Short vs Long Survival)
{clf_df.to_string(index=False)}

REGRESSION RESULTS (Predict Days Till Death)
{reg_df.to_string(index=False)}  """)


def generate_audio(harmful_genes, stage2):
    print_section("GENERATING AUDIO SUMMARY")

    try:
        from gtts import gTTS

        best_clf     = stage2['best_clf']
        best_clf_acc = stage2['clf_results'][best_clf]['acc']
        best_reg     = stage2['best_reg']
        best_reg_r2  = stage2['reg_results'][best_reg]['R2']

        summary_text = f"""
        Cancer Mutation Analysis and Survival Prediction.
        B Tech Third Year CSE Machine Learning Project.

        This project uses a two stage machine learning pipeline
        on real Glioblastoma cancer data.

        Stage One — Gene Classification.
        Seven hundred and eleven frequently mutated genes were analyzed.
        {len(harmful_genes)} harmful genes were identified.
        Models trained include K Nearest Neighbors, Naive Bayes,
        Decision Tree, Support Vector Machine,
        Random Forest, Bagging and AdaBoost.

        Bridge — Mutation Score.
        {len(stage2['overlap_genes'])} harmful genes overlapped
        between both datasets.
        A custom feature called Mutation Score was computed
        for each patient.

        Stage Two — Survival Prediction.
        Three hundred Glioblastoma patients were analyzed.
        Principal Component Analysis reduced features
        to {stage2['X_pca'].shape[1]} components.
        Clustering was performed using K Means, K Medoid and DBSCAN.
        Best regression model was {best_reg}
        with R squared of {best_reg_r2:.4f}.
        Best classification model was {best_clf}
        with accuracy of {best_clf_acc:.2%}.
        All seven lab assignments are covered in this pipeline.
        Pipeline complete.
        """

        tts = gTTS(text=summary_text, lang='en', slow=False)
        tts.save("outputs/pipeline_summary.mp3")
        print("   Audio saved : outputs/pipeline_summary.mp3")
        print("   Open the file to listen!")

    except Exception as e:
        print(f"    Audio error: {e}")
        print("   Check internet connection — gTTS needs internet")


def main():
    start_time = time.time()

    print_banner()

    harmful_genes, best_gene_model, gene_results = run_stage1()

    stage2 = run_stage2(harmful_genes)

    print_final_summary(harmful_genes, stage2)

    generate_audio(harmful_genes, stage2)

    elapsed = time.time() - start_time
    print(f"\n Total time: {elapsed:.1f} seconds")
    print(" Pipeline complete! Check outputs/ folder.\n")


if __name__ == "__main__":
    main()