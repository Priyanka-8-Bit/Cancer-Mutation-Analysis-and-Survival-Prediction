# utils.py
# Shared helper functions used across the project

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── Paths ──────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data")
PLOTS_DIR   = os.path.join(BASE_DIR, "outputs", "plots")
MODELS_DIR  = os.path.join(BASE_DIR, "outputs", "models")
RESULTS_DIR = os.path.join(BASE_DIR, "outputs", "results")

GENE_FILE     = os.path.join(DATA_DIR, "genes.tsv")
SURVIVAL_FILE = os.path.join(DATA_DIR, "survival.csv")


def save_plot(filename):
    """Save current matplotlib figure to plots folder"""
    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   📊 Saved: outputs/plots/{filename}")


def save_model(model, filename):
    """Save trained ML model to models folder"""
    path = os.path.join(MODELS_DIR, filename)
    joblib.dump(model, path)
    print(f"   💾 Saved: outputs/models/{filename}")


def load_model(filename):
    """Load a saved ML model"""
    path = os.path.join(MODELS_DIR, filename)
    return joblib.load(path)


def save_results(df, filename):
    """Save a DataFrame as CSV to results folder"""
    path = os.path.join(RESULTS_DIR, filename)
    df.to_csv(path, index=False)
    print(f"   📄 Saved: outputs/results/{filename}")


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 55)
    print(f"  {title}")
    print("=" * 55)


def print_step(title):
    """Print a formatted step header"""
    print(f"\n── {title} ──")