# 💰 Adult Income Predictor

> An end-to-end machine learning web application that predicts whether an individual's annual income exceeds **$50,000**, built on the UCI Adult Census dataset using **XGBoost with GridSearchCV hyperparameter tuning** and deployed via **Streamlit**.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Live Demo](#-live-demo)
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Project Architecture](#-project-architecture)
- [Feature Engineering](#-feature-engineering)
- [ML Pipeline](#-ml-pipeline)
- [Model Comparison](#-model-comparison)
- [Hyperparameter Tuning](#-hyperparameter-tuning)
- [Final Model Results](#-final-model-results)
- [Design Decisions](#-design-decisions)
- [Known Limitations](#-known-limitations)
- [Installation & Usage](#-installation--usage)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

## 🧠 Project Overview

This project tackles a **binary classification** problem in socioeconomic data science: predicting income brackets from demographic and employment features. The end-to-end solution covers data preprocessing, SMOTE-based class balancing, feature engineering, multi-model benchmarking, hyperparameter tuning via GridSearchCV, and deployment as an interactive web application.

Users can input demographic and employment information and receive an instant prediction with a confidence score, making the model accessible to non-technical users.

---

## 🚀 Live Demo

👉 **[Try the App Here](https://adult-income-predictor-4dqfcmscmmw767yk4asft8.streamlit.app)**

---

## 🎯 Problem Statement

**Objective:** Given a set of socioeconomic attributes about an individual, predict whether their annual income is:

- `>50K` — High income
- `<=50K` — Lower income

This is a supervised binary classification task with real-world applications in credit risk assessment, financial profiling, and policy analysis.

---

## 📊 Dataset

| Property        | Details                                                                                           |
| --------------- | ------------------------------------------------------------------------------------------------- |
| Source          | [UCI Machine Learning Repository — Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult) |
| Also known as   | Census Income Dataset                                                                             |
| Instances       | ~48,842 rows                                                                                      |
| Train / Test    | 80% / 20% stratified split → 39,073 train, 9,758 test                                            |
| Features        | 14 raw features                                                                                   |
| Target          | `income` → `>50K` or `<=50K`                                                                     |
| Class Imbalance | ~75% `<=50K` (7,422 test), ~25% `>50K` (2,336 test) — addressed with SMOTE                      |
| Year collected  | 1994 US Census — see [Known Limitations](#-known-limitations)                                     |

### Raw Input Features

| Feature          | Type        | Description                                      |
| ---------------- | ----------- | ------------------------------------------------ |
| `age`            | Numerical   | Age of the individual                            |
| `workclass`      | Categorical | Employment sector                                |
| `fnlwgt`         | Numerical   | Census sampling weight — **dropped** (see below) |
| `education`      | Categorical | Highest education level attained                 |
| `education-num`  | Numerical   | Numeric encoding of education level              |
| `marital-status` | Categorical | Marital status                                   |
| `occupation`     | Categorical | Type of job                                      |
| `relationship`   | Categorical | Household relationship role                      |
| `race`           | Categorical | Race — **dropped** (see below)                   |
| `sex`            | Categorical | Gender                                           |
| `capital-gain`   | Numerical   | Investment capital gains                         |
| `capital-loss`   | Numerical   | Investment capital losses                        |
| `hours-per-week` | Numerical   | Weekly working hours                             |
| `native-country` | Categorical | Country of origin — **dropped** (see below)      |

---

## 🏗️ Project Architecture

```
Raw Census Data
      │
      ▼
Data Cleaning & Preprocessing
      │
      ▼
Feature Engineering (9 derived features)
      │
      ▼
ColumnTransformer Pipeline
  ├── Numerical  → Median Imputation → StandardScaler
  └── Categorical → Mode Imputation  → OrdinalEncoder
      │
      ▼
SMOTE — Oversample minority class (>50K) to 1:1 ratio
      │
      ▼
Multi-Model Benchmarking
  ├── Logistic Regression  (baseline)
  ├── Random Forest
  ├── SVM (RBF Kernel)
  ├── XGBoost              ← best AUC
  └── Gradient Boosting    ← best F1
      │
      ▼
GridSearchCV Hyperparameter Tuning (5-fold Stratified CV, 32 candidates, 160 fits)
      │
      ▼
Final Tuned XGBoost Model
      │
      ▼
Binary Prediction + Confidence Score
      │
      ▼
Streamlit Web Application
```

---

## ⚙️ Feature Engineering

Nine new features are derived from raw census features before model training and inference:

| Engineered Feature | Derivation                           | Rationale                                                   |
| ------------------ | ------------------------------------ | ----------------------------------------------------------- |
| `age_group`        | Age bucketed into 6 life stages      | Captures non-linear income patterns across age              |
| `capital_net`      | `capital_gain - capital_loss`        | Net investment position is more informative than raw values |
| `has_capital`      | Binary flag for any capital activity | Separates active investors from non-investors               |
| `hours_category`   | Hours bucketed (Part-time → Extreme) | Work intensity has non-linear income effects                |
| `is_married`       | Binary flag for married status       | Strong proxy for household income stability                 |
| `education_level`  | Education mapped to 5 ordinal tiers  | Groups similar education outcomes meaningfully              |
| `work_life_ratio`  | `hours_per_week / 40`                | Normalized effort metric relative to standard working week  |
| `is_high_skill`    | Flag for high-skill occupations      | Exec, Prof, Tech, Protective roles correlate with >50K      |
| `age_x_edu`        | `age × education_num`                | Interaction term capturing experience + education synergy   |

**Features dropped before model inference:**

| Feature          | Reason for Dropping                                                                           |
| ---------------- | --------------------------------------------------------------------------------------------- |
| `fnlwgt`         | Census sampling weight — not a demographic predictor, adds noise                              |
| `education`      | Replaced by ordinal `education_level`; `education_num` retained for interactions             |
| `native-country` | ~90% of values are "United-States"; low signal, high cardinality noise                        |
| `marital-status` | Replaced by binary `is_married` which captures the key income-related signal                 |
| `capital-gain`   | Replaced by `capital_net` and `has_capital`                                                   |
| `capital-loss`   | Replaced by `capital_net` and `has_capital`                                                   |
| `race`           | Dropped to avoid encoding protected attributes — see [Known Limitations](#-known-limitations) |

---

## 🔧 ML Pipeline

### Class Imbalance — SMOTE

The training set had a ~3:1 imbalance (≤50K : >50K). SMOTE (Synthetic Minority Over-sampling Technique) was applied to the **training set only** to produce a balanced 1:1 ratio before model fitting. The test set was left untouched to reflect real-world distribution.

### Preprocessing

A `ColumnTransformer` applies separate pipelines to numerical and categorical features:

**Numerical Pipeline** (`age`, `hours-per-week`, `capital_net`, `has_capital`, `is_married`, `work_life_ratio`, `is_high_skill`, `age_x_edu`):

```
SimpleImputer(strategy='median') → StandardScaler()
```

**Categorical Pipeline** (`workclass`, `occupation`, `relationship`, `sex`, `age_group`, `hours_category`, `education_level`):

```
SimpleImputer(strategy='most_frequent') → OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
```

---

## 📊 Model Comparison

Five models were benchmarked on the same preprocessed + SMOTE-balanced training set and evaluated on the held-out test set:

| Model                | Accuracy | Precision | Recall | F1 Score | ROC-AUC | CV F1 (5-fold)    |
| -------------------- | -------- | --------- | ------ | -------- | -------- | ----------------- |
| Logistic Regression  | 79.73%   | 55.04%    | 83.65% | 66.39%   | 89.64%   | 0.8201 ± 0.0026   |
| Random Forest        | 82.15%   | 59.07%    | 82.83% | 68.96%   | 91.29%   | 0.8787 ± 0.0016   |
| SVM (RBF Kernel)     | 84.69%   | 73.08%    | 57.06% | 64.09%   | 90.17%   | 0.6466 ± 0.0042   |
| **XGBoost**          | **85.52%** | **68.13%** | **74.23%** | **71.05%** | **92.01%** | **0.9028 ± 0.0038** |
| Gradient Boosting    | 85.22%   | 66.20%    | 78.21% | 71.70%   | 92.35%   | 0.8976 ± 0.0035   |

> **XGBoost was selected for tuning** — it delivered the best balance of accuracy, AUC-ROC, and CV F1 stability, making it the most reliable model for production deployment.

---

## 🔬 Hyperparameter Tuning

GridSearchCV with 5-fold Stratified Cross-Validation was used to tune XGBoost across **32 candidate combinations (160 total fits)**:

**Search Space:**

| Parameter          | Values Searched  |
| ------------------ | ---------------- |
| `n_estimators`     | 200, 300         |
| `max_depth`        | 4, 6             |
| `learning_rate`    | 0.05, 0.1        |
| `subsample`        | 0.8, 1.0         |
| `colsample_bytree` | 0.8, 1.0         |

**Best Parameters Found:**

| Parameter          | Best Value |
| ------------------ | ---------- |
| `n_estimators`     | 300        |
| `max_depth`        | 6          |
| `learning_rate`    | 0.1        |
| `subsample`        | 0.8        |
| `colsample_bytree` | 0.8        |

**Best CV F1 Score: 0.9028**

---

## 🎯 Final Model Results

**Tuned XGBoost — Test Set Performance:**

| Metric                      | Score      |
| --------------------------- | ---------- |
| Accuracy                    | **85.52%** |
| F1 Score (>50K class)       | **71.05%** |
| ROC-AUC                     | **92.01%** |
| Precision (>50K class)      | **68.13%** |
| Recall (>50K class)         | **74.23%** |
| CV F1 (5-fold, best params) | **0.9028** |

**Classification Report (Test Set — 9,758 samples):**

| Class  | Precision | Recall | F1-Score | Support |
| ------ | --------- | ------ | -------- | ------- |
| <=50K  | 0.92      | 0.89   | 0.90     | 7,422   |
| >50K   | 0.68      | 0.74   | 0.71     | 2,336   |
| **Macro Avg** | **0.80** | **0.82** | **0.81** | **9,758** |
| **Weighted Avg** | **0.86** | **0.86** | **0.86** | **9,758** |

> The dataset has a ~75/25 class imbalance. **Recall on the >50K class (74.23%)** is the primary metric — a naive model predicting <=50K for everyone achieves 76% accuracy while being completely useless. SMOTE + XGBoost achieves strong minority-class recall without sacrificing overall accuracy.

---

## 💡 Design Decisions

**Why XGBoost over Gradient Boosting (which had a slightly higher F1)?**
Gradient Boosting achieved F1 = 0.717 vs XGBoost's 0.7105, but XGBoost had a higher ROC-AUC (0.9201 vs 0.9235 — negligible difference), better CV F1 stability (0.9028 vs 0.8976), and significantly faster training time. For a production app where inference speed and reliability matter, XGBoost is the better choice.

**Why SMOTE on training data only?**
Applying SMOTE to the test set would contaminate the evaluation — the model would appear to perform better on synthetic samples than it would on real-world data. SMOTE is applied exclusively to training data to simulate what the model would see in deployment.

**Why drop `race` but keep `sex`?**
Both are protected attributes. `race` was dropped due to sparse representation across categories and the risk of encoding historical bias. `sex` was retained as it is a strong predictor in this dataset, but this is an acknowledged tradeoff — see [Known Limitations](#-known-limitations).

**Why `OrdinalEncoder` for categoricals?**
XGBoost can handle ordinal-encoded categoricals efficiently. `OneHotEncoder` would be more theoretically correct for nominal features but produces a wider feature matrix. The current approach is a pragmatic choice that performed well in benchmarking.

**Why `StandardScaler` with XGBoost?**
Tree-based models are invariant to monotonic feature scaling, so `StandardScaler` does not affect XGBoost's split decisions. It is included for forward-compatibility — if the pipeline is swapped to a linear model in future experiments, scaling is already in place.

**Why hardcode `native_country = "United-States"`?**
Over 90% of training data is from the United States. Rather than silently produce unreliable predictions for other countries, the app restricts scope to US residents.

---

## ⚠️ Known Limitations

This section is included deliberately. A model without acknowledged limitations is more dangerous than one that is transparent about its gaps.

- **Outdated data:** The dataset reflects 1994 US Census records. Income thresholds, workforce composition, and economic conditions have changed significantly. The $50K cutoff is not adjusted for inflation (~$100K in 2024 dollars).
- **Geographic scope:** The app is only validated for US residents. `native_country` is hardcoded to "United-States" — non-US predictions are unreliable.
- **Protected attributes:** `sex` is used as a predictive feature. The model's differential accuracy across demographic groups has not been formally audited. This model should **not** be used in any consequential decision-making (credit, hiring, etc.) without a full fairness evaluation.
- **Education input inconsistency:** The app requires users to input both an education category and a numeric education value manually. Mismatched inputs (e.g., "Doctorate" with education years = 1) will produce unreliable predictions.
- **SMOTE limitations:** SMOTE generates synthetic samples by interpolation, which may not fully represent the true minority distribution. It can introduce noise near class boundaries.
- **No data drift monitoring:** The model has no mechanism to detect when incoming prediction requests differ significantly from the training distribution.

---

## 🖥️ Installation & Usage

### Prerequisites

- Python 3.8+
- pip

### 1. Clone the Repository

```bash
git clone (https://github.com/sagarsingh-analyst/Adult-Income-Predictor)

```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the App

```bash
streamlit run app.py
```


---

## 📁 Project Structure

```
income-predictor/
│
├── app.py                        # Streamlit web application
├── income.ipynb                  # Full EDA, feature engineering, model training & evaluation
├── requirements.txt              # Pinned Python dependencies
├── README.md                     # Project documentation
│
└── models/
    ├── xgboost_model.pkl  # Tuned XGBoost classifier (saved as gradient_boost_model)
    ├── preprocessor.pkl          # Fitted ColumnTransformer pipeline
    ├── cat_features.pkl          # Categorical feature name list
    └── num_features.pkl          # Numerical feature name list
```

---

## 🛠️ Tech Stack

| Layer             | Technology                              |
| ----------------- | --------------------------------------- |
| Language          | Python 3.8+                             |
| ML Framework      | scikit-learn, XGBoost, imbalanced-learn |
| Web Framework     | Streamlit                               |
| Data Processing   | pandas, NumPy                           |
| Explainability    | SHAP                                    |
| Model Persistence | joblib                                  |
| Deployment        | Streamlit Community Cloud               |

---

## 🔮 Future Improvements

Listed in priority order:

- [ ] **SHAP explainability dashboard** — per-prediction feature importance already imported in notebook; expose in app UI
- [ ] **Fairness audit** — evaluate accuracy, precision, and recall across `sex` and other demographic splits before any real-world use
- [ ] **Fix education input** — auto-populate `education_num` from the education dropdown to eliminate inconsistent user inputs
- [ ] **Model performance dashboard** — expose confusion matrix, ROC curve, and calibration plots within the app
- [ ] **Docker containerization** — portable deployment independent of Streamlit Community Cloud
- [ ] **MLflow experiment tracking** — versioned model registry for reproducibility across runs
- [ ] **Optuna tuning** — replace GridSearchCV with Optuna for more efficient hyperparameter search over a larger space

---

## 👤 Author

**Sagar Singh**

- GitHub: (https://github.com/sagarsingh-analyst)
- LinkedIn: (https://linkedin.com/in/sagarsingh-analyst)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/adult) for the dataset
- [Streamlit](https://streamlit.io) for the deployment framework
- [XGBoost](https://xgboost.readthedocs.io/) for the model library
- [imbalanced-learn](https://imbalanced-learn.org/) for SMOTE implementation
