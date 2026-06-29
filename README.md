# 💰 Adult Income Predictor

> A production-ready machine learning web application that predicts whether an individual's annual income exceeds **$50,000**, built on the UCI Adult Census dataset using **XGBoost** and deployed via **Streamlit**.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Live Demo](#-live-demo)
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Project Architecture](#-project-architecture)
- [Feature Engineering](#-feature-engineering)
- [ML Pipeline](#-ml-pipeline)
- [Model Details](#-model-details)
- [Installation & Usage](#-installation--usage)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

## 🧠 Project Overview

This project tackles a classic **binary classification** problem in socioeconomic data science: predicting income brackets from demographic and employment features. The end-to-end solution covers data preprocessing, feature engineering, model training, and deployment as an interactive web application.

The app allows users to input demographic and employment information and receive an instant prediction with a confidence score — making the model interpretable and accessible to non-technical users.

---

## 🚀 Live Demo

👉 **https://adult-income-predictor-4dqfcmscmmw767yk4asft8.streamlit.app/**

---

## 🎯 Problem Statement

**Objective:** Given a set of socioeconomic attributes about an individual, predict whether their annual income is:
- `>50K` — High income
- `<=50K` — Lower income

This is a supervised binary classification task. It has real-world applications in credit risk assessment, financial profiling, and targeted policy-making.

---

## 📊 Dataset

| Property | Details |
|---|---|
| Source | [UCI Machine Learning Repository — Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult) |
| Also known as | Census Income Dataset |
| Instances | ~48,842 rows |
| Features | 14 raw features |
| Target | `income` → `>50K` or `<=50K` |
| Class Imbalance | ~75% `<=50K`, ~25% `>50K` |

### Raw Input Features

| Feature | Type | Description |
|---|---|---|
| `age` | Numerical | Age of the individual |
| `workclass` | Categorical | Employment sector |
| `fnlwgt` | Numerical | Census sampling weight (dropped) |
| `education` | Categorical | Highest education level attained |
| `education-num` | Numerical | Numeric encoding of education |
| `marital-status` | Categorical | Marital status |
| `occupation` | Categorical | Type of job |
| `relationship` | Categorical | Household relationship role |
| `race` | Categorical | Race (not used in this model) |
| `sex` | Categorical | Gender |
| `capital-gain` | Numerical | Investment capital gains |
| `capital-loss` | Numerical | Investment capital losses |
| `hours-per-week` | Numerical | Weekly working hours |
| `native-country` | Categorical | Country of origin (dropped) |

---

## 🏗️ Project Architecture

```
Raw Census Data
      │
      ▼
Data Cleaning & Preprocessing
      │
      ▼
Feature Engineering (9 new features)
      │
      ▼
ColumnTransformer Pipeline
  ├── Numerical → Median Imputation → StandardScaler
  └── Categorical → Mode Imputation → OrdinalEncoder
      │
      ▼
XGBoost Classifier (300 estimators, depth=6, lr=0.1)
      │
      ▼
Binary Prediction + Confidence Score
      │
      ▼
Streamlit Web Application
```

---

## ⚙️ Feature Engineering

One of the core strengths of this project is the **domain-driven feature engineering** applied on top of raw census features. Nine new features are derived before model inference:

| Engineered Feature | Derivation | Rationale |
|---|---|---|
| `age_group` | Age bucketed into 6 life stages | Captures non-linear income patterns across age |
| `capital_net` | `capital_gain - capital_loss` | Net investment activity is more informative than raw values |
| `has_capital` | Binary flag for any capital activity | Separates investors from non-investors |
| `hours_category` | Hours bucketed (Part-time → Extreme) | Work intensity has non-linear income effects |
| `is_married` | Binary flag for married status | Strong proxy for dual-income household |
| `education_level` | Education mapped to 5 ordinal tiers | Groups similar education outcomes together |
| `work_life_ratio` | `hours_per_week / 40` | Normalized effort metric relative to standard hours |
| `is_high_skill` | Flag for high-skill occupations | Exec, Prof, Tech, Protective roles correlate strongly with >50K |
| `age_x_edu` | `age × education_num` | Interaction term capturing experience + education synergy |

The following raw features are **dropped** before model inference as they either leak information, add noise, or are represented by engineered alternatives:

`fnlwgt`, `education`, `native-country`, `marital-status`, `capital-gain`, `capital-loss`, `education-num`

---

## 🔧 ML Pipeline

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

## 🤖 Model Details

| Parameter | Value |
|---|---|
| Algorithm | XGBoost (`XGBClassifier`) |
| Objective | `binary:logistic` |
| Number of Estimators | 300 |
| Max Tree Depth | 6 |
| Learning Rate | 0.1 |
| Output | Class label + probability scores |

**Why XGBoost?**
- Handles mixed feature types (numerical + categorical) efficiently
- Built-in regularization reduces overfitting on imbalanced datasets
- Superior performance on tabular data compared to linear models
- Fast inference — critical for real-time web app predictions

---

## 🖥️ Installation & Usage

### Prerequisites
- Python 3.8+
- pip

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/income-predictor.git
cd income-predictor
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the App
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

### requirements.txt
```
streamlit
pandas
numpy
joblib
scikit-learn==1.9.0
xgboost
```

---

## 📁 Project Structure

```
income-predictor/
│
├── app.py                        # Streamlit web application
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
│
└── models/
    ├── gradient_boost_model.pkl  # Trained XGBoost classifier
    ├── preprocessor.pkl          # Fitted ColumnTransformer pipeline
    ├── cat_features.pkl          # Categorical feature name list
    └── num_features.pkl          # Numerical feature name list
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.8+ |
| ML Framework | scikit-learn, XGBoost |
| Web Framework | Streamlit |
| Data Processing | pandas, NumPy |
| Model Persistence | joblib |
| Deployment | Streamlit Community Cloud |

---

## 🔮 Future Improvements

- [ ] Add SHAP values for per-prediction explainability
- [ ] Implement model performance dashboard with confusion matrix and ROC curve
- [ ] Add batch prediction via CSV file upload
- [ ] Experiment with class imbalance techniques (SMOTE, class weights)
- [ ] Hyperparameter tuning with Optuna or GridSearchCV
- [ ] Add model versioning and experiment tracking with MLflow
- [ ] Containerize with Docker for portable deployment

---

## 👤 Author

**Sagar Singh**
- GitHub: (https://github.com/sagarsingh-analyst)
- LinkedIn:(https://linkedin.com/in/sagarsingh-analyst)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/adult) for the dataset
- [Streamlit](https://streamlit.io) for the deployment framework
- [XGBoost](https://xgboost.readthedocs.io/) for the model library
