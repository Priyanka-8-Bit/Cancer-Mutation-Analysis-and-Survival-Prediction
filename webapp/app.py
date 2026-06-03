# webapp/app.py
# Flask web application for Cancer Mutation ML Project

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import json
import os
import sys

# Add parent directory so we can import our ml modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)

# ── Helper: load results if they exist ──────────────────────────────
def load_results():
    results = {}
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Classification results
    clf_path = os.path.join(base, 'outputs', 'results', 'classification_results.csv')
    if os.path.exists(clf_path):
        results['classification'] = pd.read_csv(clf_path).to_dict(orient='records')

    # Regression results
    reg_path = os.path.join(base, 'outputs', 'results', 'regression_results.csv')
    if os.path.exists(reg_path):
        results['regression'] = pd.read_csv(reg_path).to_dict(orient='records')

    # Harmful genes
    genes_path = os.path.join(base, 'outputs', 'results', 'harmful_genes.csv')
    if os.path.exists(genes_path):
        results['harmful_genes'] = pd.read_csv(genes_path)['Gene_Symbol'].tolist()

    # Survival data for charts
    surv_path = os.path.join(base, 'data', 'survival.csv')
    if os.path.exists(surv_path):
        df = pd.read_csv(surv_path)
        results['total_patients']   = len(df)
        results['dead_count']       = int((df['Life'] == 'Dead').sum())
        results['alive_count']      = int((df['Life'] == 'Alive').sum())
        results['avg_age']          = round(df['Age at Diagnosis '].mean(), 1)
        results['avg_days']         = round(df['Days Till Death'].mean(), 1)

        # Age distribution
        results['age_dist'] = df['Age at Diagnosis '].dropna().tolist()

        # Days Till Death distribution
        results['days_dist'] = df['Days Till Death'].dropna().tolist()

        # Gender distribution
        gender = df['Gender'].value_counts().to_dict()
        results['gender_dist'] = gender

        # Race distribution
        race = df['Race'].value_counts().to_dict()
        results['race_dist'] = race

        # Mutation score if exists
        if 'Mutation_Score' in df.columns:
            results['mutation_scores'] = df['Mutation_Score'].dropna().tolist()
            results['avg_mutation_score'] = round(df['Mutation_Score'].mean(), 2)

    # Gene data
    gene_path = os.path.join(base, 'data', 'genes.tsv')
    if os.path.exists(gene_path):
        gdf = pd.read_csv(gene_path, sep='\t')
        top10 = gdf.nlargest(10, 'num_mutations')[['symbol', 'num_mutations']]
        results['top_genes']        = top10['symbol'].tolist()
        results['top_genes_counts'] = top10['num_mutations'].tolist()
        results['total_genes']      = len(gdf)
        results['harmful_count']    = len(results.get('harmful_genes', []))

    return results


# ── Routes ───────────────────────────────────────────────────────────
@app.route('/')
def index():
    results = load_results()
    return render_template('index.html', results=results)


@app.route('/predict')
def predict_page():
    # Load overlap genes for the form
    base       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    genes_path = os.path.join(base, 'outputs', 'results', 'harmful_genes.csv')
    overlap_genes = []
    if os.path.exists(genes_path):
        overlap_genes = pd.read_csv(genes_path)['Gene_Symbol'].tolist()[:20]
    return render_template('predict.html', overlap_genes=overlap_genes)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        from ml.utils import load_model
        import joblib

        data = request.json
        age  = float(data.get('age', 55))
        gene_mutations = data.get('gene_mutations', {})

        base       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        genes_path = os.path.join(base, 'outputs', 'results', 'harmful_genes.csv')
        surv_path  = os.path.join(base, 'data', 'survival.csv')

        # Load models
        scaler     = load_model("survival_scaler.pkl")
        pca        = load_model("pca.pkl")
        regressor  = load_model("regression_model.pkl")
        classifier = load_model("classifier.pkl")
        le_gender  = load_model("label_encoder_gender.pkl")
        le_race    = load_model("label_encoder_race.pkl")

        surv_df    = pd.read_csv(surv_path)
        surv_clean = surv_df.drop(columns=['ID', 'Life', 'Days Till Death']).copy()
        surv_clean.fillna(surv_clean.median(numeric_only=True), inplace=True)
        surv_clean['Gender'] = le_gender.transform(surv_clean['Gender'].str.strip())
        surv_clean['Race']   = le_race.transform(surv_clean['Race'].str.strip())

        overlap_df    = pd.read_csv(genes_path)
        overlap_genes = overlap_df['Gene_Symbol'].tolist()
        valid_genes   = [g for g in overlap_genes if g in surv_df.columns]

        mutation_score = sum(int(gene_mutations.get(g, 0)) for g in valid_genes)
        if mutation_score == 0:
            mutation_score = surv_df[valid_genes].sum(axis=1).mean()

        new_row = surv_clean.iloc[0:1].copy() * 0
        new_row['Age at Diagnosis '] = age
        new_row['Mutation_Score']    = mutation_score

        gender_str = data.get('gender', 'Male').capitalize()
        race_str   = data.get('race', 'Caucasian').strip()
        new_row['Gender'] = le_gender.transform([gender_str])[0] if gender_str in le_gender.classes_ else 0
        new_row['Race']   = le_race.transform([race_str])[0]     if race_str   in le_race.classes_   else 0

        new_scaled = scaler.transform(new_row)
        new_pca    = pca.transform(new_scaled)

        age_mean    = surv_df['Age at Diagnosis '].mean()
        age_std     = surv_df['Age at Diagnosis '].std() + 1e-8
        mut_scores  = surv_df[valid_genes].sum(axis=1)
        mut_mean    = mut_scores.mean()
        mut_std     = mut_scores.std() + 1e-8
        age_norm    = (age - age_mean) / age_std
        mut_norm    = (mutation_score - mut_mean) / mut_std
        interaction = np.array([[age_norm * mut_norm]])
        new_pca_enh = np.hstack([new_pca, interaction])

        predicted_days   = max(0, float(regressor.predict(new_pca)[0]))
        predicted_months = predicted_days / 30.44
        group_pred       = int(classifier.predict(new_pca_enh)[0])
        group_names      = {0: 'Short Survival (< 6 months)', 1: 'Long Survival (> 18 months)'}
        group_label      = group_names.get(group_pred, 'Unknown')

        return jsonify({
            'success'        : True,
            'days'           : round(predicted_days, 0),
            'months'         : round(predicted_months, 1),
            'group'          : group_label,
            'mutation_score' : round(mutation_score, 1),
            'age'            : age,
            'gender'         : gender_str,
            'race'           : race_str,
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True, port=5000)