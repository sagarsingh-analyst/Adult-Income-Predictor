import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Adult Income Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL & PREPROCESSOR
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/xgboost_model.pkl')
        preprocessor = joblib.load('models/preprocessor.pkl')
        return model, preprocessor, True
    except FileNotFoundError:
        return None, None, False

model, preprocessor, model_loaded = load_model()

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def engineer_features(input_dict: dict) -> pd.DataFrame:
    df = pd.DataFrame([input_dict])

    # 1. Age group
    df['age_group'] = pd.cut(
        df['age'],
        bins=[0, 25, 35, 45, 55, 65, 100],
        labels=['Young', 'Early Career', 'Mid Career',
                'Senior', 'Pre-Retirement', 'Retirement']
    ).astype(str)

    # 2. Capital net
    df['capital_net'] = df['capital_gain'] - df['capital_loss']

    # 3. Has capital
    df['has_capital'] = (
        (df['capital_gain'] > 0) | (df['capital_loss'] > 0)
    ).astype(int)

    # 4. Hours category
    df['hours_category'] = pd.cut(
        df['hours-per-week'],
        bins=[0, 20, 34, 40, 55, 100],
        labels=['Part-time', 'Below Full', 'Full-time', 'Overtime', 'Extreme']
    ).astype(str)

    # 5. Is married
    df['is_married'] = df['marital_status'].apply(
        lambda x: 1 if x in ['Married-civ-spouse', 'Married-AF-spouse'] else 0
    )

    # 6. Education level
    edu_map = {
        'Preschool': 'Low',      '1st-4th': 'Low',    '5th-6th': 'Low',
        '7th-8th': 'Low',        '9th': 'Low',         '10th': 'Medium-Low',
        '11th': 'Medium-Low',    '12th': 'Medium-Low', 'HS-grad': 'Medium',
        'Some-college': 'Medium','Assoc-voc': 'Medium-High',
        'Assoc-acdm': 'Medium-High', 'Bachelors': 'High',
        'Masters': 'Very High',  'Prof-school': 'Very High',
        'Doctorate': 'Very High'
    }
    df['education_level'] = df['education'].map(edu_map).fillna('Medium')

    # 7. Work-life ratio
    df['work_life_ratio'] = (df['hours-per-week'] / 40.0).round(3)

    # 8. High skill flag
    high_skill = ['Exec-managerial', 'Prof-specialty',
                  'Tech-support', 'Protective-serv']
    df['is_high_skill'] = df['occupation'].apply(
        lambda x: 1 if x in high_skill else 0
    )

    # 9. Age x Education interaction
    df['age_x_edu'] = df['age'] * df['education_num']

    # Drop columns not used in model
    DROP = ['income', 'fnlwgt', 'education', 'native_country',
            'marital_status', 'capital_gain', 'capital_loss',
            'income_encoded']
    for col in DROP:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: MAKE PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
def make_prediction(input_dict):
    df_input = engineer_features(input_dict)
    X_proc = preprocessor.transform(df_input)
    pred = model.predict(X_proc)[0]
    prob = model.predict_proba(X_proc)[0]
    return pred, prob, df_input


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR - USER INPUTS
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Enter Person Details")
    st.markdown("---")

    # Personal Info
    st.markdown("### Personal Info")
    age = st.slider("Age", min_value=18, max_value=90, value=35, step=1)
    sex = st.selectbox("Sex", options=['Male', 'Female'])

    # Education
    st.markdown("### Education")
    education = st.selectbox(
        "Education",
        options=[
            'HS-grad', 'Some-college', 'Bachelors', 'Masters',
            'Doctorate', 'Prof-school', 'Assoc-acdm', 'Assoc-voc',
            '11th', '10th', '7th-8th', '12th', '9th',
            '5th-6th', '1st-4th', 'Preschool'
        ],
        index=0
    )
    education_num = st.slider(
        "Education Years (Numeric)", min_value=1, max_value=16, value=9, step=1
    )

    # Work Info
    st.markdown("### Work Info")
    workclass = st.selectbox(
        "Work Class",
        options=[
            'Private', 'Self-emp-not-inc', 'Self-emp-inc',
            'Federal-gov', 'Local-gov', 'State-gov',
            'Without-pay', 'Never-worked'
        ]
    )
    occupation = st.selectbox(
        "Occupation",
        options=[
            'Exec-managerial', 'Prof-specialty', 'Tech-support',
            'Sales', 'Craft-repair', 'Machine-op-inspct',
            'Transport-moving', 'Adm-clerical', 'Other-service',
            'Protective-serv', 'Handlers-cleaners', 'Farming-fishing',
            'Priv-house-serv', 'Armed-Forces'
        ]
    )
    hours_per_week = st.slider(
        "hours-Per-Week", min_value=1, max_value=99, value=40, step=1
    )

    # Family Info
    st.markdown("### Family Info")
    marital_status = st.selectbox(
        "Marital Status",
        options=[
            'Married-civ-spouse', 'Never-married', 'Divorced',
            'Separated', 'Widowed', 'Married-spouse-absent',
            'Married-AF-spouse'
        ]
    )
    relationship = st.selectbox(
        "Relationship",
        options=[
            'Husband', 'Not-in-family', 'Wife', 'Own-child',
            'Unmarried', 'Other-relative'
        ]
    )

    # Financial Info
    st.markdown("### Financial Info")
    capital_gain = st.number_input(
        "Capital Gain ($)", min_value=0, max_value=99999, value=0, step=100
    )
    capital_loss = st.number_input(
        "Capital Loss ($)", min_value=0, max_value=99999, value=0, step=100
    )

    st.markdown("---")
    predict_clicked = st.button("Predict Income", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────────────────────
st.title("Adult Income Predictor")
st.markdown("Predict whether income exceeds $50K/year based on census data.")

# Model not loaded warning
if not model_loaded:
    st.error("Model not found! Please run `python save_model.py` first to train and save your model, then restart the app.")
    st.stop()

if predict_clicked:
    # Build input dict
    input_data = {
        'age': age,
        'workclass': workclass,
        'fnlwgt': 0,
        'education': education,
        'education_num': education_num,
        'marital_status': marital_status,
        'occupation': occupation,
        'relationship': relationship,
        'sex': sex,
        'capital_gain': capital_gain,
        'capital_loss': capital_loss,
        'hours-per-week': hours_per_week,
        'native_country': 'United-States',
    }

    # Make prediction
    with st.spinner("Predicting..."):
        try:
            pred, prob, df_engineered = make_prediction(input_data)
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()

    # Display result
    if pred == 1:
        st.success(f"Predicted Income: >50K (Confidence: {prob[1]*100:.1f}%)")
    else:
        st.error(f"Predicted Income: <=50K (Confidence: {prob[0]*100:.1f}%)")

    # Show probabilities
    st.write(f"Probability <=50K: {prob[0]*100:.1f}%")
    st.write(f"Probability >50K: {prob[1]*100:.1f}%")

    # Show engineered features
    st.markdown("---")
    st.subheader("Engineered Features")
    st.write(df_engineered.to_dict(orient='records')[0])

else:
    st.info("Fill in the person's details in the sidebar and click 'Predict Income' to get the prediction.")