import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ---------------------------------------------------------
# Loan Approve / Decline Prediction App
# Compatible with Loan_Approve_Decline_Deployment_Model.ipynb
# ---------------------------------------------------------

st.set_page_config(
    page_title="Loan Pre-Screening",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = Path(__file__).parent / "loan_approve_decline_pipeline.joblib"

# Clean, light, professional theme
st.markdown(
    """
    <style>
    :root {
        --primary: #2457C5;
        --primary-dark: #1946A3;
        --ink: #172033;
        --muted: #64748B;
        --line: #E2E8F0;
        --surface: #FFFFFF;
        --page: #F5F7FB;
    }

    .stApp {
        background: var(--page);
        color: var(--ink);
    }
    [data-testid="stHeader"] {
        background: rgba(245, 247, 251, 0.92);
    }
    [data-testid="stToolbar"] { right: 1rem; }
    .block-container {
        max-width: 1180px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }
    h1, h2, h3, p, label, [data-testid="stMarkdownContainer"] {
        color: var(--ink);
    }
    h1 {
        font-weight: 800 !important;
        letter-spacing: -0.035em;
        line-height: 1.15;
    }
    h2, h3 { letter-spacing: -0.02em; }
    [data-testid="stCaptionContainer"] p {
        color: var(--muted);
        font-size: 0.96rem;
    }
    [data-testid="stForm"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 1.4rem 1.5rem 1.2rem;
        box-shadow: 0 8px 28px rgba(15, 23, 42, 0.045);
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 16px;
    }
    [data-testid="stAlert"] {
        border-radius: 12px;
        border-width: 1px;
    }
    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 1rem 1.1rem;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.035);
    }
    div[data-testid="stMetricLabel"] p {
        color: #64748B;
        font-weight: 600;
    }
    div[data-testid="stMetricValue"] {
        color: #172033;
        font-weight: 750;
    }
    .stButton > button, .stFormSubmitButton > button {
        background: var(--primary);
        color: white;
        border: 1px solid var(--primary);
        border-radius: 10px;
        min-height: 3rem;
        font-weight: 700;
        transition: all 0.15s ease;
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover {
        background: var(--primary-dark);
        border-color: var(--primary-dark);
        color: white;
    }
    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div,
    div[data-baseweb="textarea"] > div {
        background: #FFFFFF;
        border-color: #CBD5E1;
        border-radius: 9px;
    }
    [data-testid="stRadio"] label,
    [data-testid="stCheckbox"] label {
        color: var(--ink);
    }
    hr {
        border-color: var(--line);
        margin-top: 1.6rem;
        margin-bottom: 1.6rem;
    }
    .hero-card {
        background: linear-gradient(120deg, #FFFFFF 0%, #EEF4FF 100%);
        border: 1px solid #DCE7FA;
        border-radius: 20px;
        padding: 1.6rem 1.8rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 10px 30px rgba(36, 87, 197, 0.06);
    }
    .hero-eyebrow {
        color: #2457C5;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }
    .hero-subtitle {
        color: #64748B;
        font-size: 1rem;
        margin-top: 0.55rem;
        margin-bottom: 0;
    }
    .section-note {
        color: #64748B;
        margin-top: -0.45rem;
        margin-bottom: 1rem;
    }
    footer { visibility: hidden; }
    @media (max-width: 700px) {
        .block-container { padding: 1rem 1rem 2rem; }
        [data-testid="stForm"] { padding: 1rem; }
        .hero-card { padding: 1.2rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-eyebrow">Credit decision support · ML demo</div>
        <h1 style="margin:0;">🏦 Loan Pre-Screening</h1>
        <p class="hero-subtitle">
            Explore how applicant and affordability details relate to a model's
            prediction of a historical loan decision.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "Educational prototype only. This model predicts historical bank decisions, not actual repayment "
    "or default. Do not use its output as the sole basis for a real lending decision. "
    "Human review, independent validation, and appropriate policy/compliance checks are required."
)

@st.cache_resource
def load_model(path_string: str):
    return joblib.load(path_string)

if not MODEL_PATH.exists():
    st.error("Trained model file not found.")
    st.markdown(
        "Run all cells in **Loan_Approve_Decline_Deployment_Model.ipynb** first. "
        "That notebook should create `loan_approve_decline_pipeline.joblib`. "
        "Place the `.joblib` file in the same folder as `app.py`."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Could not load the model artifact: {exc}")
    st.stop()

st.subheader("Applicant details")
st.markdown(
    '<p class="section-note">Enter the applicant information below, then run the pre-screening prediction.</p>',
    unsafe_allow_html=True,
)

with st.form("loan_application_form"):
    st.markdown("#### Personal and income profile")
    left, right = st.columns(2, gap="large")

    with left:
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=30, step=1)
        employment_type = st.selectbox(
            "Employment type",
            ["Salaried", "Self-Employed Professional", "Business Owner", "Gig / Contract"],
        )
        years_in_current_job = st.number_input(
            "Years in current job/business", min_value=0.0, max_value=70.0, value=4.0, step=0.5
        )
        monthly_income = st.number_input(
            "Monthly income (₹)", min_value=0.0, max_value=100_000_000.0, value=65000.0, step=1000.0
        )
        existing_emi = st.number_input(
            "Existing monthly EMIs (₹)", min_value=0.0, max_value=100_000_000.0, value=8000.0, step=500.0
        )
        has_credit_score = st.radio(
            "Does the applicant have a credit score?", ["Yes", "No"], horizontal=True
        )
        if has_credit_score == "Yes":
            credit_score = st.number_input(
                "Credit score (300–900)", min_value=300, max_value=900, value=735, step=1
            )
        else:
            credit_score = np.nan
        credit_history_months = st.number_input(
            "Credit history (months)",
            min_value=0,
            max_value=1200,
            value=(48 if has_credit_score == "Yes" else 0),
            step=1,
        )

    with right:
        num_active_loans = st.number_input(
            "Number of active loans", min_value=0, max_value=100, value=1, step=1
        )
        delinquencies_24m = st.number_input(
            "Late/missed repayments in last 24 months", min_value=0, max_value=100, value=0, step=1
        )
        loan_purpose = st.selectbox(
            "Loan purpose",
            [
                "Medical", "Debt Consolidation", "Education", "Home Improvement",
                "Travel", "Wedding", "Business", "Other",
            ],
        )
        loan_amount = st.number_input(
            "Requested loan amount (₹)",
            min_value=1.0,
            max_value=100_000_000.0,
            value=300000.0,
            step=5000.0,
        )
        tenure_months = st.selectbox(
            "Requested tenure (months)", [6, 12, 18, 24, 36, 48, 60, 72, 84, 96]
        )
        proposed_emi = st.number_input(
            "Proposed monthly EMI (₹)",
            min_value=0.0,
            max_value=100_000_000.0,
            value=10250.0,
            step=250.0,
        )
        city_tier = st.selectbox(
            "City tier",
            [1, 2, 3],
            format_func=lambda x: {
                1: "Tier 1 — Metro",
                2: "Tier 2 — Large city",
                3: "Tier 3 — Smaller city",
            }[x],
        )

    submitted = st.form_submit_button(
        "🔎  Predict decision", type="primary", use_container_width=True
    )

if submitted:
    # Match the training notebook's feature engineering.
    corrected_income = monthly_income
    income_corrected = False
    if corrected_income > 600_000:
        corrected_income = corrected_income / 12
        income_corrected = True

    if corrected_income <= 0:
        st.error("Monthly income must be greater than ₹0 so affordability ratios can be calculated.")
        st.stop()

    foir = (existing_emi + proposed_emi) / corrected_income
    loan_to_income = loan_amount / (corrected_income * 12)
    is_ntc = int(pd.isna(credit_score) or credit_history_months == 0)

    features = pd.DataFrame(
        [{
            "credit_score": credit_score,
            "is_ntc": is_ntc,
            "foir": foir,
            "loan_to_income": loan_to_income,
            "delinquencies_24m": delinquencies_24m,
            "num_active_loans": num_active_loans,
            "years_in_current_job": years_in_current_job,
            "age": age,
            "tenure_months": tenure_months,
            "city_tier": city_tier,
            "employment_type": employment_type,
            "loan_purpose": loan_purpose,
        }]
    )

    expected_features = [
        "credit_score", "is_ntc", "foir", "loan_to_income",
        "delinquencies_24m", "num_active_loans", "years_in_current_job",
        "age", "tenure_months", "city_tier", "employment_type", "loan_purpose",
    ]
    features = features[expected_features]

    try:
        prediction = str(model.predict(features)[0])
        class_labels = [str(c) for c in model.named_steps["classifier"].classes_]
        probabilities = model.predict_proba(features)[0]
        probability_map = {
            label: float(prob) for label, prob in zip(class_labels, probabilities)
        }
    except Exception as exc:
        st.error(
            "Prediction failed. Ensure the model artifact was created by the supplied notebook "
            "and uses the same feature schema."
        )
        st.exception(exc)
        st.stop()

    st.divider()
    st.subheader("Prediction result")
    st.caption("The result below reflects the model's learned historical decision patterns.")

    if prediction.lower() == "approved":
        st.success(f"Model prediction: {prediction}")
    else:
        st.warning(f"Model prediction: {prediction}")

    metric_cols = st.columns(2, gap="large")
    with metric_cols[0]:
        st.metric("Probability of Approved", f"{100 * probability_map.get('Approved', 0.0):.1f}%")
    with metric_cols[1]:
        st.metric("Probability of Declined", f"{100 * probability_map.get('Declined', 0.0):.1f}%")

    st.subheader("Calculated affordability indicators")
    col1, col2, col3 = st.columns(3, gap="medium")
    col1.metric("FOIR", f"{foir:.1%}")
    col2.metric("Loan / annual income", f"{loan_to_income:.2f}×")
    col3.metric("New-to-credit", "Yes" if is_ntc else "No")

    if income_corrected:
        st.warning(
            "The entered monthly income exceeded ₹6,00,000, so the app divided it by 12 "
            "using the same correction rule as the training notebook. Please verify this value."
        )

    if is_ntc:
        st.warning(
            "This applicant is marked new-to-credit. The model may be less reliable for this group; "
            "route the application for human review rather than relying on the prediction alone."
        )

    with st.expander("View model probabilities and input features"):
        st.json({k: round(v, 4) for k, v in probability_map.items()})
        display_features = features.copy()
        display_features["foir"] = display_features["foir"].map(lambda x: f"{x:.4f}")
        display_features["loan_to_income"] = display_features["loan_to_income"].map(lambda x: f"{x:.4f}")
        st.dataframe(display_features, use_container_width=True)

st.divider()
st.caption(
    "Model target: historical `decision` (Approved/Declined). It is not a validated default-risk model. "
    "Predictions are informational and should be reviewed by a qualified human."
)
