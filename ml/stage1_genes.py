# stage1_genes.py
# Stage 1 — Gene Classification Pipeline
# Goal: Identify harmful genes from mutation data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier,
                               BaggingClassifier,
                               AdaBoostClassifier)
from sklearn.metrics import accuracy_score, classification_report

from ml.utils import (save_plot, save_model, save_results,
                      print_section, print_step, GENE_FILE)


def load_gene_data():
    """Load and return the gene dataset"""
    print_step("Loading Gene Dataset")
    df = pd.read_csv(GENE_FILE, sep='\t')
    print(f"Loaded: {df.shape[0]} genes, {df.shape[1]} columns")
    return df


def preprocess_genes(df):
    """Clean and prepare gene features"""
    print_step("Preprocessing Gene Data")

    # Keep only useful columns
    clean = df[[
        'symbol',
        'num_cohort_ssm_affected_cases',
        'cohort_ssm_affected_cases_percentage',
        'num_mutations'
    ]].copy()

    # Fill missing values
    clean.fillna(clean.median(numeric_only=True), inplace=True)

    # Create harmful label
    count_thresh = clean['num_cohort_ssm_affected_cases'].quantile(0.75)
    freq_thresh  = clean['cohort_ssm_affected_cases_percentage'].quantile(0.75)
    mut_thresh   = clean['num_mutations'].quantile(0.75)

    clean['is_harmful'] = (
        (clean['num_cohort_ssm_affected_cases'] > count_thresh) &
        (clean['cohort_ssm_affected_cases_percentage'] > freq_thresh) &
        (clean['num_mutations'] > mut_thresh)
    ).astype(int)

    print(f"   Harmful genes  : {clean['is_harmful'].sum()}")
    print(f"   Harmless genes : {(clean['is_harmful']==0).sum()}")

    # Scale features
    scaler   = StandardScaler()
    features = ['num_cohort_ssm_affected_cases',
                'cohort_ssm_affected_cases_percentage',
                'num_mutations']
    X_scaled = scaler.fit_transform(clean[features])
    y        = clean['is_harmful']

    save_model(scaler, "gene_scaler.pkl")

    return clean, X_scaled, y


def train_gene_models(X, y):
    """Train all classifiers on gene data"""
    print_step("Training Gene Classification Models")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models = {
        'KNN'           : KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes'   : GaussianNB(),
        'Decision Tree' : DecisionTreeClassifier(random_state=42),
        'SVM'           : SVC(kernel='rbf', random_state=42),
        'Random Forest' : RandomForestClassifier(n_estimators=100,
                                                  random_state=42),
        'Bagging'       : BaggingClassifier(random_state=42),
        'AdaBoost'      : AdaBoostClassifier(random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc   = accuracy_score(y_test, preds)
        results[name] = acc
        print(f"   {name:<20} → {acc:.2%}")

    # Best model
    best_name  = max(results, key=results.get)
    best_model = models[best_name]
    print(f"\n    Best: {best_name} ({results[best_name]:.2%})")

    # Save best model
    save_model(best_model, "gene_classifier.pkl")

    return models, results, best_name, best_model


def get_harmful_genes(df, clean, X_scaled, best_model):
    """Extract list of harmful gene symbols"""
    print_step("Extracting Harmful Genes")

    all_preds = best_model.predict(X_scaled)
    clean['predicted_harmful'] = all_preds

    harmful_list = clean[
        clean['predicted_harmful'] == 1
    ]['symbol'].tolist()

    print(f"   Harmful genes identified: {len(harmful_list)}")
    print(f"   Examples: {harmful_list[:8]}")

    # Save list
    pd.DataFrame({'Gene_Symbol': harmful_list}).to_csv(
        "outputs/results/harmful_genes.csv", index=False
    )

    return harmful_list


def plot_gene_results(results):
    """Plot gene model comparison"""
    print_step("Saving Gene Model Comparison Plot")

    plt.figure(figsize=(10, 5))
    names  = list(results.keys())
    accs   = [v * 100 for v in results.values()]
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))

    bars = plt.barh(names, accs, color=colors, edgecolor='white')
    plt.xlabel("Accuracy (%)")
    plt.title("Gene Classification — Model Comparison")
    plt.xlim(0, 110)

    for bar, acc in zip(bars, accs):
        plt.text(bar.get_width() + 1,
                 bar.get_y() + bar.get_height()/2,
                 f'{acc:.1f}%', va='center')

    save_plot("gene_model_comparison.png")


def run_stage1():
    """Run the complete Stage 1 pipeline"""
    print_section("STAGE 1 — GENE CLASSIFICATION")

    # Load
    df = load_gene_data()

    # Preprocess
    clean, X_scaled, y = preprocess_genes(df)

    # Train
    models, results, best_name, best_model = train_gene_models(
        X_scaled, y
    )

    # Extract harmful genes
    harmful_list = get_harmful_genes(df, clean, X_scaled, best_model)

    # Plot
    plot_gene_results(results)

    print(f"\n Stage 1 Complete!")
    print(f"   Harmful genes found: {len(harmful_list)}")

    return harmful_list, best_name, results