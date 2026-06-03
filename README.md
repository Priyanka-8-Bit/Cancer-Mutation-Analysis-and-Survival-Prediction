# 🧬 Cancer Mutation Analysis & Survival Prediction
### B.Tech TY CSE — Machine Learning Project

A two-stage machine learning pipeline that:
1. Identifies harmful genes from genomic mutation data
2. Predicts cancer patient survival using clinical + gene features

---

## 📁 Project Structure
```
cancer-mutation-ml/
│
├── data/
│   ├── genes.tsv              ← Frequently mutated genes dataset
│   └── survival.csv           ← Glioblastoma patient survival dataset
│
├── ml/
│   ├── __init__.py
│   ├── stage1_genes.py        ← Gene classification pipeline
│   ├── stage2_survival.py     ← Survival prediction pipeline
│   └── utils.py               ← Shared helper functions
│
├── outputs/
│   ├── plots/                 ← All generated graphs
│   ├── models/                ← Saved trained models
│   └── results/               ← CSV result tables
│
├── main.py                    ← Run full pipeline
├── predict.py                 ← Predict for new patient
├── requirements.txt           ← Dependencies
└── README.md
```

---

## 🚀 How To Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline
```bash
python main.py
```

### 3. Predict for a new patient
```bash
python predict.py
```

---

## 🧠 How It Works

### Stage 1 — Gene Classification
- Dataset: 711 frequently mutated genes (TSV)
- Labels genes as harmful or harmless using mutation count,
  frequency and affected cases
- Trains 7 ML models and picks the best one
- Outputs a list of harmful gene symbols

### Bridge — Mutation Score
- Connects Stage 1 and Stage 2 using gene symbols
- For each patient counts how many harmful genes are mutated
- Creates a new feature called `Mutation_Score`

### Stage 2 — Survival Prediction
- Dataset: 300 Glioblastoma patients (CSV)
- 506 features including clinical data and 500+ gene mutations
- PCA reduces dimensions while keeping 95% variance
- Clusters patients using K-Means, K-Medoid, DBSCAN
- Predicts Days Till Death using Ridge/Lasso/Linear Regression
- Classifies survival group using KNN, NB, DT, SVM, RF, Ensemble

---

## 📊 Models Used

| Assignment | Models |
|---|---|
| A1 | Preprocessing, EDA, Encoding, Scaling |
| A2 | KNN, Naive Bayes |
| A3 | Decision Tree, K-Fold Cross Validation |
| A4 | SVM (Linear, RBF, Poly) vs Decision Tree |
| A5 | Random Forest, Bagging, AdaBoost, Stacking |
| A6 | K-Means, K-Medoid, DBSCAN |
| A7 | Linear, Ridge, Lasso Regression |
| Extra | PCA, Audio Output (gTTS) |

---

## 📦 Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- gtts
- joblib

---

## 🎯 Key Features

- ✅ Real clinical + genomic cancer dataset
- ✅ Two-stage connected pipeline
- ✅ Custom Mutation_Score feature engineering
- ✅ PCA on 500+ gene columns
- ✅ All 7 ML lab assignments covered
- ✅ Audio output for predictions
- ✅ Trained models saved for reuse

---

## 👨‍💻 Author

**Priyanka Bhagat**
B.Tech TY CSE | Panel G
ML Lab — Dr. Jayshree Aher
AY 2025-26
