# stage2_survival.py
# Stage 2 — Survival Prediction Pipeline
# Updated: Dead vs Alive classification + Days Till Death regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier,
                               BaggingClassifier,
                               AdaBoostClassifier,
                               StackingClassifier)
from sklearn.linear_model import (LinearRegression,
                                   LogisticRegression,
                                   Ridge, Lasso)
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (accuracy_score,
                              classification_report,
                              confusion_matrix,
                              mean_absolute_error,
                              mean_squared_error,
                              r2_score)
from sklearn.metrics import pairwise_distances

from ml.utils import (save_plot, save_model, save_results,
                      print_section, print_step, SURVIVAL_FILE)


# ══════════════════════════════════════════
# LOAD + PREPROCESS
# ══════════════════════════════════════════

def load_survival_data():
    print_step("Loading Survival Dataset")
    df = pd.read_csv(SURVIVAL_FILE)
    print(f"    Loaded: {df.shape[0]} patients, {df.shape[1]} columns")
    print(f"    Dead  : {(df['Life']=='Dead').sum()} patients")
    print(f"   Alive : {(df['Life']=='Alive').sum()} patients")
    return df


def preprocess_survival(df, harmful_genes):
    print_step("Preprocessing Survival Data")

    # Detect gene columns
    gene_cols = []
    for col in df.columns:
        if col in ['ID', 'Age at Diagnosis ',
                   'Life', 'Race', 'Gender',
                   'Days Till Death']:
            continue
        unique_vals = set(df[col].dropna().unique())
        if unique_vals.issubset({0, 1}):
            gene_cols.append(col)

    # Compute Mutation Score
    overlap = [g for g in harmful_genes if g in gene_cols]
    df      = df.copy()
    df['Mutation_Score'] = df[overlap].sum(axis=1)

    print(f"    Gene columns found    : {len(gene_cols)}")
    print(f"   Overlapping genes     : {len(overlap)}")
    print(f"    Mutation Score range  : "
          f"{df['Mutation_Score'].min()} - "
          f"{df['Mutation_Score'].max()}")

    # Encode Life → Vital_Status (Dead=1, Alive=0)
    df['Vital_Status'] = (df['Life'] == 'Dead').astype(int)
    print(f"    Vital Status encoded  : Dead=1, Alive=0")

    # Drop unnecessary columns
    clean = df.drop(columns=['ID', 'Life']).copy()
    clean.fillna(clean.median(numeric_only=True), inplace=True)

    # Encode categorical columns
    le_gender       = LabelEncoder()
    le_race         = LabelEncoder()
    clean['Gender'] = le_gender.fit_transform(clean['Gender'].str.strip())
    clean['Race']   = le_race.fit_transform(clean['Race'].str.strip())

    # Targets
    y_days   = clean['Days Till Death'].copy()   # Regression target
    y_status = clean['Vital_Status'].copy()      # Classification target

    # Features
    X = clean.drop(
        columns=['Days Till Death', 'Vital_Status']
    ).copy()

    # Scale
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    save_model(scaler,    "survival_scaler.pkl")
    save_model(le_gender, "label_encoder_gender.pkl")
    save_model(le_race,   "label_encoder_race.pkl")

    print(f"   Features  : {X.shape[1]} columns")
    print(f"   Patients  : {X.shape[0]}")

    return df, X, X_scaled, y_days, y_status, scaler, overlap


# ══════════════════════════════════════════
# PCA
# ══════════════════════════════════════════

def apply_pca(X_scaled):
    print_step("Applying PCA")

    pca = PCA(n_components=0.70, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    print(f"   Before PCA : {X_scaled.shape[1]} features")
    print(f"   After PCA  : {X_pca.shape[1]} components")
    print(f"   Variance   : {sum(pca.explained_variance_ratio_)*100:.2f}%")

    save_model(pca, "pca.pkl")

    plt.figure(figsize=(8, 4))
    cumulative = np.cumsum(pca.explained_variance_ratio_) * 100
    plt.plot(range(1, len(cumulative)+1), cumulative,
             color='steelblue', linewidth=2, marker='o', markersize=3)
    plt.axhline(y=80, color='red', linestyle='--', label='80% threshold')
    plt.title("PCA — Cumulative Variance Explained")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Variance (%)")
    plt.legend()
    save_plot("pca_variance.png")

    return pca, X_pca


# ══════════════════════════════════════════
# CLUSTERING
# ══════════════════════════════════════════

def run_clustering(X_pca, y_status):
    print_step("Clustering Patients (A6)")

    kmeans        = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_pca)
    save_model(kmeans, "kmeans.pkl")

    def kmedoid(X, k, random_state=42):
        np.random.seed(random_state)
        n          = X.shape[0]
        medoid_idx = np.random.choice(n, k, replace=False)
        for _ in range(100):
            distances   = pairwise_distances(X, X[medoid_idx])
            labels      = np.argmin(distances, axis=1)
            new_medoids = []
            for i in range(k):
                pts = np.where(labels == i)[0]
                if len(pts) == 0:
                    new_medoids.append(medoid_idx[i])
                    continue
                sub  = pairwise_distances(X[pts])
                best = np.argmin(sub.sum(axis=1))
                new_medoids.append(pts[best])
            if set(new_medoids) == set(medoid_idx):
                break
            medoid_idx = np.array(new_medoids)
        return labels

    kmedoid_labels = kmedoid(X_pca, k=3)

    dbscan        = DBSCAN(eps=2.0, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_pca)
    n_clusters    = len(set(dbscan_labels)) - (
        1 if -1 in dbscan_labels else 0
    )

    print(f"    K-Means  : 3 clusters")
    print(f"   K-Medoid : 3 clusters")
    print(f"   DBSCAN   : {n_clusters} clusters found")

    # Visualize
    pca_2d = PCA(n_components=2, random_state=42)
    X_vis  = pca_2d.fit_transform(X_pca)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Clustering Results (A6)", fontweight='bold')

    for ax, labels, title, cmap in zip(
        axes,
        [kmeans_labels, kmedoid_labels, dbscan_labels],
        ['K-Means (K=3)', 'K-Medoid (K=3)', 'DBSCAN'],
        ['Set1', 'Set2', 'Set3']
    ):
        sc = ax.scatter(X_vis[:, 0], X_vis[:, 1],
                        c=labels, cmap=cmap,
                        alpha=0.6, edgecolor='white', s=40)
        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        plt.colorbar(sc, ax=ax)

    save_plot("clustering_results.png")

    # Also plot Dead vs Alive
    plt.figure(figsize=(7, 5))
    colors = {0: 'steelblue', 1: 'coral'}
    labels_names = {0: 'Alive', 1: 'Dead'}
    for status in [0, 1]:
        mask = y_status == status
        plt.scatter(
            X_vis[mask, 0], X_vis[mask, 1],
            c=colors[status],
            label=labels_names[status],
            alpha=0.6, edgecolor='white', s=40
        )
    plt.title("PCA — Dead vs Alive Patients")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    save_plot("dead_vs_alive_pca.png")

    return kmeans_labels


# ══════════════════════════════════════════
# REGRESSION
# ══════════════════════════════════════════

def run_regression(X_pca, y):
    print_step("Regression — Predict Days Till Death (A7)")

    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.2, random_state=42
    )

    models = {
        'Linear Regression' : LinearRegression(),
        'Ridge Regression'  : Ridge(alpha=1.0),
        'Lasso Regression'  : Lasso(alpha=1.0),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds         = model.predict(X_test)
        results[name] = {
            'MAE'  : mean_absolute_error(y_test, preds),
            'RMSE' : np.sqrt(mean_squared_error(y_test, preds)),
            'R2'   : r2_score(y_test, preds),
            'preds': preds,
            'model': model
        }
        print(f"   {name:<20} → "
              f"MAE: {results[name]['MAE']:.1f} days | "
              f"R²: {results[name]['R2']:.4f}")

    best_name  = max(results, key=lambda x: results[x]['R2'])
    best_model = results[best_name]['model']
    save_model(best_model, "regression_model.pkl")
    print(f"\n  Best: {best_name} (R²={results[best_name]['R2']:.4f})")

    reg_df = pd.DataFrame([{
        'Model': k,
        'MAE'  : f"{v['MAE']:.1f}",
        'RMSE' : f"{v['RMSE']:.1f}",
        'R2'   : f"{v['R2']:.4f}"
    } for k, v in results.items()])
    save_results(reg_df, "regression_results.csv")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Regression — Predicted vs Actual", fontweight='bold')
    colors = ['steelblue', 'coral', 'green']

    for idx, (name, res) in enumerate(results.items()):
        axes[idx].scatter(y_test, res['preds'],
                          alpha=0.5, color=colors[idx],
                          edgecolor='white', s=40)
        mn = min(y_test.min(), res['preds'].min())
        mx = max(y_test.max(), res['preds'].max())
        axes[idx].plot([mn, mx], [mn, mx], 'k--', label='Perfect')
        axes[idx].set_title(
            f"{name}\nR²={res['R2']:.4f} | MAE={res['MAE']:.0f}d"
        )
        axes[idx].set_xlabel("Actual Days")
        axes[idx].set_ylabel("Predicted Days")
        axes[idx].legend()

    save_plot("regression_results.png")

    return results, best_name, best_model


# ══════════════════════════════════════════
# CLASSIFICATION — DEAD VS ALIVE
# ══════════════════════════════════════════

def run_classification(X_pca, y_status, X, df):
    print_step("Classification — Dead vs Alive (A2-A5)")

    print(f"   Dead  patients: {(y_status==1).sum()}")
    print(f"   Alive patients: {(y_status==0).sum()}")

    # Add interaction feature
    age_col     = df['Age at Diagnosis '].values
    mut_col     = df['Mutation_Score'].values
    age_norm    = (age_col - age_col.mean()) / (age_col.std() + 1e-8)
    mut_norm    = (mut_col - mut_col.mean()) / (mut_col.std() + 1e-8)
    interaction = (age_norm * mut_norm).reshape(-1, 1)
    X_enhanced  = np.hstack([X_pca, interaction])

    print(f"    Enhanced features: {X_enhanced.shape[1]} total")

    # Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_enhanced, y_status,
        test_size=0.2,
        random_state=42,
        stratify=y_status
    )

    print(f"   Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    # Find best K
    best_k = max(
        range(1, 11),
        key=lambda k: KNeighborsClassifier(
            n_neighbors=k
        ).fit(X_train, y_train).score(X_test, y_test)
    )
    print(f"   Best K for KNN: {best_k}")

    # All models
    models = {
        'KNN'           : KNeighborsClassifier(n_neighbors=best_k),
        'Naive Bayes'   : GaussianNB(),
        'Decision Tree' : DecisionTreeClassifier(
                            max_depth=4,
                            random_state=42,
                            class_weight='balanced'),
        'SVM'           : SVC(kernel='rbf',
                              C=10,
                              gamma='scale',
                              random_state=42,
                              probability=True,
                              max_iter=2000,
                              class_weight='balanced'),
        'Random Forest' : RandomForestClassifier(
                            n_estimators=200,
                            max_depth=8,
                            min_samples_split=4,
                            random_state=42,
                            class_weight='balanced'),
        'Bagging'       : BaggingClassifier(
                            n_estimators=100,
                            random_state=42),
        'AdaBoost'      : AdaBoostClassifier(
                            n_estimators=100,
                            learning_rate=0.5,
                            random_state=42),
        'Stacking'      : StackingClassifier(
                            estimators=[
                                ('dt', DecisionTreeClassifier(
                                    max_depth=3,
                                    random_state=42)),
                                ('nb', GaussianNB())
                            ],
                            final_estimator=LogisticRegression(
                                random_state=42,
                                class_weight='balanced'),
                            cv=3)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds         = model.predict(X_test)
        acc           = accuracy_score(y_test, preds)
        results[name] = {
            'acc'  : acc,
            'preds': preds,
            'model': model
        }
        print(f"   {name:<20} → {acc:.2%}")

    best_name  = max(results, key=lambda x: results[x]['acc'])
    best_model = results[best_name]['model']
    save_model(best_model, "classifier.pkl")
    print(f"\n    Best: {best_name} ({results[best_name]['acc']:.2%})")

    # Save results
    rows = []
    for name, res in results.items():
        report = classification_report(
            y_test, res['preds'],
            target_names=['Alive', 'Dead'],
            output_dict=True,
            zero_division=0
        )
        rows.append({
            'Model'    : name,
            'Accuracy' : f"{res['acc']:.2%}",
            'Precision': f"{report['weighted avg']['precision']:.2%}",
            'Recall'   : f"{report['weighted avg']['recall']:.2%}",
            'F1'       : f"{report['weighted avg']['f1-score']:.2%}",
        })

    clf_df = pd.DataFrame(rows).sort_values(
        'Accuracy', ascending=False
    ).reset_index(drop=True)
    save_results(clf_df, "classification_results.csv")

    # Plot accuracy comparison
    plt.figure(figsize=(10, 5))
    accs   = [results[n]['acc'] * 100 for n in clf_df['Model']]
    colors = plt.cm.Set3(np.linspace(0, 1, len(clf_df)))
    bars   = plt.barh(clf_df['Model'].tolist(),
                      accs, color=colors, edgecolor='white')
    plt.xlabel("Accuracy (%)")
    plt.title("Classification — Dead vs Alive\nModel Accuracy Comparison")
    plt.xlim(0, 110)

    for bar, acc in zip(bars, accs):
        plt.text(bar.get_width() + 1,
                 bar.get_y() + bar.get_height()/2,
                 f'{acc:.1f}%', va='center')

    save_plot("classification_comparison.png")

    # Confusion matrix
    best_preds = results[best_name]['preds']
    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_test, best_preds),
                annot=True, fmt='d', cmap='Blues',
                xticklabels=['Alive', 'Dead'],
                yticklabels=['Alive', 'Dead'])
    plt.title(f"Confusion Matrix — {best_name}\n"
              f"Acc: {results[best_name]['acc']:.2%}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    save_plot("best_model_confusion.png")

    return results, best_name, best_model, y_status


# ══════════════════════════════════════════
# MAIN STAGE 2 RUNNER
# ══════════════════════════════════════════

def run_stage2(harmful_genes):
    print_section("STAGE 2 — SURVIVAL PREDICTION")

    df = load_survival_data()

    df, X, X_scaled, y_days, y_status, scaler, overlap = preprocess_survival(
        df, harmful_genes
    )

    pca, X_pca = apply_pca(X_scaled)

    run_clustering(X_pca, y_status)

    reg_results, best_reg, best_reg_model = run_regression(
        X_pca, y_days
    )

    clf_results, best_clf, best_clf_model, y_class = run_classification(
        X_pca, y_status, X, df
    )

    print(f"\n Stage 2 Complete!")

    return {
        'df'            : df,
        'X'             : X,
        'X_scaled'      : X_scaled,
        'X_pca'         : X_pca,
        'y'             : y_days,
        'y_status'      : y_status,
        'pca'           : pca,
        'scaler'        : scaler,
        'overlap_genes' : overlap,
        'best_reg'      : best_reg,
        'best_reg_model': best_reg_model,
        'best_clf'      : best_clf,
        'best_clf_model': best_clf_model,
        'reg_results'   : reg_results,
        'clf_results'   : clf_results,
    }